//! Puente hacia el proceso persistente de herramientas de Python.
//!
//! Antes, cada llamada a una herramienta lanzaba un interprete de Python nuevo.
//! Medido: 7,5 s por llamada, siempre, contra un timeout de 10 s en el frontend.
//! Importar llama_index y chromadb cuesta ~3,7 s y validar el modelo de
//! embeddings anade una peticion de red; eso no se arregla afinando el
//! subproceso. Ver H-12.
//!
//! Ahora se arranca `TOOLS/server.py` una sola vez y se le habla por tuberias
//! con JSON delimitado por lineas. El coste de arranque se paga mientras el
//! usuario todavia esta conectando.
//!
//! Se eligieron tuberias en lugar de un servidor HTTP en localhost porque el
//! servidor seria alcanzable por cualquier proceso de la maquina, y una de esas
//! herramientas controla el PC. Las tuberias son privadas del proceso padre.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;

use serde_json::{json, Value};
use tauri::{AppHandle, Manager};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;
use tokio::time::{timeout, Duration};

/// Red de seguridad, no presupuesto de trabajo: con el proceso ya caliente las
/// respuestas tardan milisegundos. Solo salta si Python se ha colgado de verdad.
const TIMEOUT_HERRAMIENTA: Duration = Duration::from_secs(30);

/// El arranque incluye importar llama_index y chromadb.
const TIMEOUT_ARRANQUE: Duration = Duration::from_secs(60);

pub struct ProcesoHerramientas {
    proceso: Child,
    entrada: ChildStdin,
    salida: Lines<BufReader<ChildStdout>>,
    contador: u64,
}

/// Estado compartido de Tauri. `None` mientras no se haya arrancado o despues
/// de que el proceso muera: la siguiente llamada lo levanta de nuevo.
#[derive(Default)]
pub struct EstadoPuente(pub Arc<Mutex<Option<ProcesoHerramientas>>>);

/// Localiza la carpeta TOOLS sin rutas absolutas cableadas.
///
/// Antes estaba escrito literalmente `C:\Users\<usuario>\Perseo\TOOLS\runner.py`,
/// asi que el proyecto solo funcionaba en una maquina y una carpeta. Ver H-15.
fn ruta_tools(app: &AppHandle) -> Result<PathBuf, String> {
    // 1. Variable de entorno, para instalaciones no estandar.
    if let Ok(ruta) = std::env::var("PERSEO_TOOLS_DIR") {
        let ruta = PathBuf::from(ruta);
        if ruta.is_dir() {
            return Ok(ruta);
        }
    }

    // 2. Recursos empaquetados junto al binario (compilacion de produccion).
    if let Ok(recursos) = app.path().resource_dir() {
        let candidata = recursos.join("TOOLS");
        if candidata.is_dir() {
            return Ok(candidata);
        }
    }

    // 3. Arbol de fuentes, resuelto en tiempo de compilacion (desarrollo).
    let desarrollo = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("TOOLS");
    if desarrollo.is_dir() {
        return Ok(desarrollo);
    }

    Err("No se encuentra la carpeta TOOLS. Defina PERSEO_TOOLS_DIR.".into())
}

/// Interprete de Python. Configurable para poder apuntar a un entorno virtual.
fn interprete() -> String {
    std::env::var("PERSEO_PYTHON").unwrap_or_else(|_| "python".to_string())
}

impl ProcesoHerramientas {
    async fn arrancar(app: &AppHandle) -> Result<Self, String> {
        let tools = ruta_tools(app)?;
        let script = tools.join("server.py");
        if !script.is_file() {
            return Err(format!("No se encuentra {}", script.display()));
        }

        let mut proceso = Command::new(interprete())
            .arg("-X")
            .arg("utf8") // el interprete trabaja en UTF-8 de principio a fin
            .arg(&script)
            .current_dir(&tools)
            // Sin esto, Python en Windows escribe stdout en cp1252 y aqui se lee
            // como UTF-8: los acentos de las respuestas llegaban corruptos al
            // modelo. Ver H-13.
            .env("PYTHONIOENCODING", "utf-8")
            .env("PYTHONUNBUFFERED", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()) // los registros de Python, a la consola
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| format!("No se pudo arrancar el servidor de herramientas: {e}"))?;

        let entrada = proceso.stdin.take().ok_or("Sin stdin en el proceso hijo")?;
        let salida = proceso.stdout.take().ok_or("Sin stdout en el proceso hijo")?;
        let mut salida = BufReader::new(salida).lines();

        // El servidor saluda cuando ha cargado los modulos de herramientas.
        match timeout(TIMEOUT_ARRANQUE, salida.next_line()).await {
            Ok(Ok(Some(_))) => {}
            Ok(Ok(None)) => return Err("El servidor de herramientas murio al arrancar".into()),
            Ok(Err(e)) => return Err(format!("Error leyendo del servidor: {e}")),
            Err(_) => return Err("El servidor de herramientas no respondio al arrancar".into()),
        }

        Ok(Self { proceso, entrada, salida, contador: 0 })
    }

    async fn pedir(&mut self, herramienta: &str, argumentos: Value) -> Result<String, String> {
        self.contador += 1;
        let peticion = json!({
            "id": self.contador,
            "tool": herramienta,
            "args": argumentos,
        });

        let mut linea = peticion.to_string();
        linea.push('\n');
        self.entrada
            .write_all(linea.as_bytes())
            .await
            .map_err(|e| format!("No se pudo escribir al servidor: {e}"))?;
        self.entrada
            .flush()
            .await
            .map_err(|e| format!("No se pudo vaciar el buffer: {e}"))?;

        let respuesta = match timeout(TIMEOUT_HERRAMIENTA, self.salida.next_line()).await {
            Ok(Ok(Some(linea))) => linea,
            Ok(Ok(None)) => return Err("El servidor de herramientas cerro la conexion".into()),
            Ok(Err(e)) => return Err(format!("Error leyendo la respuesta: {e}")),
            Err(_) => {
                return Err(format!(
                    "La herramienta '{herramienta}' no respondio en {} s",
                    TIMEOUT_HERRAMIENTA.as_secs()
                ))
            }
        };

        let valor: Value = serde_json::from_str(&respuesta)
            .map_err(|e| format!("Respuesta ilegible del servidor: {e}"))?;

        if valor.get("ok").and_then(Value::as_bool).unwrap_or(false) {
            Ok(valor
                .get("result")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string())
        } else {
            Err(valor
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("Error desconocido en la herramienta")
                .to_string())
        }
    }

    async fn matar(&mut self) {
        let _ = self.proceso.kill().await;
    }
}

/// Ejecuta una herramienta en el proceso persistente de Python.
///
/// Si algo falla, el proceso se mata y se descarta: la siguiente llamada
/// arranca uno limpio. Antes el timeout vivia en JavaScript, que abandonaba la
/// promesa pero dejaba el proceso de Python vivo y trabajando para un
/// consumidor que ya no existia. Ver H-11.
#[tauri::command]
pub async fn ejecutar_herramienta_python(
    app: AppHandle,
    estado: tauri::State<'_, EstadoPuente>,
    tool_name: String,
    argumentos: String,
) -> Result<String, String> {
    let args: Value = serde_json::from_str(&argumentos)
        .map_err(|e| format!("Argumentos JSON invalidos: {e}"))?;

    let puente = estado.0.clone();
    let mut guarda = puente.lock().await;

    if guarda.is_none() {
        *guarda = Some(ProcesoHerramientas::arrancar(&app).await?);
    }

    let proceso = guarda.as_mut().expect("acaba de asignarse");
    match proceso.pedir(&tool_name, args).await {
        Ok(resultado) => Ok(resultado),
        Err(e) => {
            // Un fallo de transporte deja el proceso en estado dudoso: se
            // reemplaza. Los errores de la propia herramienta llegan como
            // respuesta JSON valida y no pasan por aqui.
            if let Some(mut viejo) = guarda.take() {
                viejo.matar().await;
            }
            Err(e)
        }
    }
}

/// Arranca el proceso por adelantado, sin esperar a la primera herramienta.
/// El frontend lo llama al conectar, para que el precalentado del indice
/// vectorial ocurra mientras el usuario todavia esta saludando.
#[tauri::command]
pub async fn precalentar_herramientas(
    app: AppHandle,
    estado: tauri::State<'_, EstadoPuente>,
) -> Result<(), String> {
    let puente = estado.0.clone();
    let mut guarda = puente.lock().await;
    if guarda.is_none() {
        *guarda = Some(ProcesoHerramientas::arrancar(&app).await?);
    }
    Ok(())
}
