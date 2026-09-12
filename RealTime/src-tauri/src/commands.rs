use xcap::Monitor;
use image::codecs::jpeg::JpegEncoder;
use base64::{Engine, engine::general_purpose::STANDARD};
use std::io::Cursor;
use tauri::AppHandle;
use tauri_plugin_store::StoreExt;

/// Archivo del almacen cifrado-por-plataforma donde vive la configuracion
/// sensible. Nunca se empaqueta en el bundle del frontend.
const ARCHIVO_AJUSTES: &str = "perseo-ajustes.json";
const CLAVE_API: &str = "gemini_api_key";

/// Fichero marcador que deja el detector de aplausos para pedir que la
/// aplicacion entre en llamada sola. Se borra al leerlo.
const MARCADOR_AUTOLLAMADA: &str = ".perseo-autollamada";
/// Marcador que solo pide sacar la ventana del escondite, sin entrar en llamada.
const MARCADOR_MOSTRAR: &str = ".perseo-mostrar";

/// Tamano al que se reduce la captura antes de mandarla al modelo. El modelo ve
/// **esta** imagen, no la pantalla: sus coordenadas hay que traducirlas.
const ANCHO_CAPTURA: u32 = 1280;
const ALTO_CAPTURA: u32 = 720;

/// La pantalla principal, que es la unica donde el raton sabe clicar.
///
/// Antes se cogia `monitors.first()`, que en un equipo con dos pantallas puede
/// ser cualquiera de las dos: el modelo veia una pantalla y el clic caia en la
/// otra. `pyautogui` mide en la principal (ver `_dentro_de_la_pantalla` en
/// `perseo_core/agentes/pc.py`), asi que se comparte esa y no otra.
fn pantalla_principal() -> Result<Monitor, String> {
    let monitores = Monitor::all().map_err(|e| e.to_string())?;
    monitores
        .iter()
        .find(|m| m.is_primary())
        .or_else(|| monitores.first())
        .cloned()
        .ok_or_else(|| "No hay ninguna pantalla".to_string())
}

/// Cuanto mide la imagen que ve el modelo y cuanto mide la pantalla de verdad.
///
/// Existe por: el modelo senala sobre la imagen y `pc.py` clica en pixeles
/// de pantalla, y nadie traducia entre las dos cosas.
#[derive(serde::Serialize)]
pub struct GeometriaPantalla {
    pub ancho_imagen: u32,
    pub alto_imagen: u32,
    pub ancho_pantalla: u32,
    pub alto_pantalla: u32,
}

#[tauri::command]
pub fn geometria_pantalla() -> Result<GeometriaPantalla, String> {
    let monitor = pantalla_principal()?;
    Ok(GeometriaPantalla {
        ancho_imagen: ANCHO_CAPTURA,
        alto_imagen: ALTO_CAPTURA,
        ancho_pantalla: monitor.width(),
        alto_pantalla: monitor.height(),
    })
}

#[tauri::command]
pub async fn capture_screen_base64(quality: u8) -> Result<String, String> {
    let monitor = pantalla_principal()?;
    let image = monitor.capture_image().map_err(|e| e.to_string())?;

    // Resize to reduce bandwidth (1280x720 is plenty for Gemini)
    let resized = image::imageops::resize(
        &image, ANCHO_CAPTURA, ALTO_CAPTURA, image::imageops::FilterType::Triangle
    );

    let mut jpeg_bytes = Vec::new();
    let mut cursor = Cursor::new(&mut jpeg_bytes);
    JpegEncoder::new_with_quality(&mut cursor, quality)
        .encode_image(&resized)
        .map_err(|e| e.to_string())?;
    
    Ok(STANDARD.encode(&jpeg_bytes))
}

/// Devuelve la clave de API de Gemini.
///
/// Tres sitios, en este orden:
///
/// 1. **El almacen local**, donde la deja el usuario desde el boton de Ajustes.
/// 2. **`GEMINI_API_KEY`** del sistema, leida en tiempo de ejecucion.
/// 3. **`perseo_core/datos/gemini.txt`**, que es de donde la lee el nucleo.
///
/// El tercero se anadio el 2026-08-16 y no es un capricho: la clave estaba
/// puesta para el nucleo y la app seguia pidiendola por pantalla, asi que en una
/// misma maquina habia que escribirla dos veces y recordar rotarla en dos
/// sitios. Ese directorio esta fuera de git, igual que el almacen.
///
/// El motivo de que esto viva en Rust y no en el frontend: cualquier variable
/// con prefijo VITE_ la incrusta Vite dentro del JavaScript compilado, asi que
/// la clave quedaba en claro dentro del .exe.
#[tauri::command]
pub fn obtener_api_key(app: AppHandle) -> Result<String, String> {
    let store = app.store(ARCHIVO_AJUSTES).map_err(|e| e.to_string())?;

    if let Some(valor) = store.get(CLAVE_API) {
        if let Some(clave) = valor.as_str() {
            if !clave.is_empty() {
                return Ok(clave.to_string());
            }
        }
    }

    let del_entorno = std::env::var("GEMINI_API_KEY").unwrap_or_default();
    if !del_entorno.trim().is_empty() {
        return Ok(del_entorno.trim().to_string());
    }

    Ok(clave_del_nucleo(&app))
}

/// La clave que usa el nucleo, si esta puesta. Cadena vacia si no.
///
/// Se busca donde `nucleo.rs` busca el token, y por lo mismo: sin rutas
/// absolutas cableadas, para que valga en `tauri dev` y en el binario instalado.
fn clave_del_nucleo(app: &AppHandle) -> String {
    for ruta in crate::nucleo::rutas_datos(app) {
        if let Ok(contenido) = std::fs::read_to_string(ruta.join("gemini.txt")) {
            let clave = contenido.trim().to_string();
            if !clave.is_empty() {
                return clave;
            }
        }
    }
    String::new()
}

// La clave ya no se guarda desde la interfaz (2026-08-22). Se pone donde la
// pone el nucleo —`<datos>/gemini.txt`, o la variable GEMINI_API_KEY— y aqui
// solo se lee: un secreto que se escribe en una ventana es un secreto que se
// ensena en una ventana. El comando `guardar_api_key` se fue con el campo.

/// Lee un ajuste cualquiera del almacen local.
///
/// Los Ajustes solo mutaban un objeto en memoria, asi que la voz y el prompt se
/// perdian al cerrar la aplicacion.
#[tauri::command]
pub fn obtener_ajuste(app: AppHandle, clave: String) -> Result<Option<serde_json::Value>, String> {
    let store = app.store(ARCHIVO_AJUSTES).map_err(|e| e.to_string())?;
    Ok(store.get(&clave))
}

/// Guarda un ajuste cualquiera y lo persiste en disco.
#[tauri::command]
pub fn guardar_ajuste(
    app: AppHandle,
    clave: String,
    valor: serde_json::Value,
) -> Result<(), String> {
    let store = app.store(ARCHIVO_AJUSTES).map_err(|e| e.to_string())?;
    store.set(&clave, valor);
    store.save().map_err(|e| e.to_string())?;
    Ok(())
}

/// Consume la senal de autollamada dejada por el detector de aplausos o por
/// los subagentes.
///
/// Devuelve el **motivo** escrito en el marcador (cadena vacia si no lo habia,
/// o si quien lo dejo —el detector— no escribio ninguno): la app se lo cuenta
/// a Perseo para que sepa por que llama. El fichero marcador se borra al
/// leerlo. Antes esto era `src/autocall.json`, importado estaticamente por
/// React, con dos problemas: Vite congela el valor al compilar (asi que en
/// produccion el disparo por aplausos no funcionaba) y nadie lo devolvia a
/// false, de modo que toda apertura manual entraba en llamada sola.
#[tauri::command]
pub fn consumir_autollamada(app: AppHandle) -> Result<String, String> {
    for ruta in rutas_marcador_autollamada(&app) {
        if ruta.is_file() {
            let motivo = std::fs::read_to_string(&ruta)
                .map(|t| t.trim().to_string())
                .unwrap_or_default();
            let _ = std::fs::remove_file(&ruta);
            return Ok(motivo);
        }
    }
    Ok(String::new())
}

pub fn rutas_marcador_autollamada(app: &AppHandle) -> Vec<std::path::PathBuf> {
    rutas_de_marcador(app, MARCADOR_AUTOLLAMADA)
}

/// Donde se busca el marcador que solo pide **enseniar la ventana**.
///
/// Existe porque cerrar la ventana no cierra Perseo: la esconde en la bandeja.
/// Con la app escondida, `perseo` desde la terminal decia "[ya estaba] la app de
/// voz" y no pasaba nada en pantalla, que desde fuera se ve exactamente igual
/// que una app rota. Ahora deja este fichero y la ventana vuelve.
///
/// Es un marcador aparte y no el de autollamada porque sacar la ventana y
/// entrar en llamada son dos cosas distintas, y solo una la pidio el usuario.
pub fn rutas_marcador_mostrar(app: &AppHandle) -> Vec<std::path::PathBuf> {
    rutas_de_marcador(app, MARCADOR_MOSTRAR)
}

fn rutas_de_marcador(app: &AppHandle, nombre: &str) -> Vec<std::path::PathBuf> {
    use tauri::Manager;

    let mut rutas = Vec::new();
    if let Ok(dir) = app.path().app_config_dir() {
        rutas.push(dir.join(nombre));
    }
    // Arbol de fuentes, para `tauri dev`.
    rutas.push(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join(nombre),
    );
    rutas
}

// Las herramientas ya no viven en la app. `nucleo.rs` encola trabajos en
// perseo-core y espera el resultado: una sola memoria, una sola cola, y lo que
// se pide por voz aparece tambien en la web del movil. Antes esto era un puente
// de tuberias hacia TOOLS/, y antes de eso un interprete de Python nuevo en cada
// llamada: 7,5 s medidos contra un timeout de 10 s. Ver a.

/// Cuaderno de la llamada: lo que la interfaz apunta para poder mirarlo luego.
///
/// Una compilacion de release no lleva devtools, asi que lo que la ventana
/// escribe en la consola se va con la ventana y no hay forma de mirar por que
/// se oyo entrecortada una respuesta. Estas lineas caen en un fichero de texto
/// junto a los datos del nucleo, se leen desde fuera y se pisan solas cuando el
/// fichero se hace grande: es un cuaderno de depuracion, no un historial.
const TOPE_DIAGNOSTICO: u64 = 512 * 1024;

fn ruta_diagnostico() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("perseo_core")
        .join("datos")
        .join("llamada.log")
}

#[tauri::command]
pub fn anotar_diagnostico(lineas: Vec<String>) -> Result<(), String> {
    use std::io::Write;

    let ruta = ruta_diagnostico();
    if let Some(dir) = ruta.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if std::fs::metadata(&ruta).map(|m| m.len()).unwrap_or(0) > TOPE_DIAGNOSTICO {
        let _ = std::fs::remove_file(&ruta);
    }

    let mut fichero = std::fs::OpenOptions::new()
.create(true)
.append(true)
.open(&ruta)
.map_err(|e| e.to_string())?;
    for linea in lineas {
        writeln!(fichero, "{linea}").map_err(|e| e.to_string())?;
    }
    Ok(())
}
