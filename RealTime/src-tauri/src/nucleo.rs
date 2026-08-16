//! Cliente del nucleo. Sustituye al puente de tuberias hacia `TOOLS/`.
//!
//! Hasta la Fase D, la app llevaba dentro sus propias herramientas: arrancaba
//! `TOOLS/server.py` y le hablaba por tuberias. Eso significaba dos sistemas con
//! dos memorias y dos formas de hacer lo mismo, y que lo que pidieras por voz no
//! existiera para la web del movil.
//!
//! Ahora la app **encola trabajos en el nucleo** y espera el resultado. Es la
//! regla del plan aplicada a la ultima cara que faltaba: *las caras no piensan*.
//! Lo que se pide por voz aparece en la cola de la web, sobrevive a que se
//! cuelgue la llamada, y lo ejecuta el mismo agente que atiende al movil.
//!
//! Sobre la restriccion que la Fase 3 protegia: **la API no expone
//! herramientas**. Aqui no se manda "ejecuta esto", se manda "encola un trabajo
//! para el agente `pc`", y quien decide que hace ese agente vive en el nucleo,
//! detras de su lista blanca. El token va en cada peticion y **no pasa por el
//! frontend**: lo lee este modulo del disco.

use std::path::PathBuf;
use std::time::Duration;

use serde_json::{json, Value};
use tauri::{AppHandle, Manager};
use tokio::time::{sleep, Instant};

/// Cuanto se espera a que un trabajo termine. Es red de seguridad, no
/// presupuesto: `memoria` y `pc` responden en milisegundos. Si salta, el trabajo
/// sigue vivo en la cola y se ve en la web — no se pierde, solo deja de
/// esperarse.
const ESPERA_MAXIMA: Duration = Duration::from_secs(30);

/// Cada cuanto se pregunta por el estado del trabajo.
const SONDEO: Duration = Duration::from_millis(250);

fn base_url() -> String {
    std::env::var("PERSEO_CORE_URL").unwrap_or_else(|_| "http://127.0.0.1:8787".to_string())
}

/// Localiza `perseo_core/datos/token.txt` sin rutas absolutas cableadas, igual
/// que hacia el puente con la carpeta TOOLS. Ver H-15.
fn ruta_token(app: &AppHandle) -> Option<PathBuf> {
    if let Ok(datos) = std::env::var("PERSEO_CORE_DATOS") {
        let candidata = PathBuf::from(datos).join("token.txt");
        if candidata.is_file() {
            return Some(candidata);
        }
    }

    if let Ok(recursos) = app.path().resource_dir() {
        let candidata = recursos.join("perseo_core").join("datos").join("token.txt");
        if candidata.is_file() {
            return Some(candidata);
        }
    }

    let desarrollo = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("perseo_core")
        .join("datos")
        .join("token.txt");
    if desarrollo.is_file() {
        return Some(desarrollo);
    }

    None
}

/// El token de acceso al nucleo. La variable de entorno manda sobre el fichero,
/// igual que en `almacen.py`.
fn token(app: &AppHandle) -> Result<String, String> {
    if let Ok(del_entorno) = std::env::var("PERSEO_TOKEN") {
        let limpio = del_entorno.trim().to_string();
        if !limpio.is_empty() {
            return Ok(limpio);
        }
    }

    let ruta = ruta_token(app).ok_or(
        "No se encuentra el token del nucleo. Arranque `python -m perseo_core` una vez, \
         o defina PERSEO_TOKEN.",
    )?;
    std::fs::read_to_string(&ruta)
        .map(|t| t.trim().to_string())
        .map_err(|e| format!("No se pudo leer {}: {e}", ruta.display()))
}

/// Traduce la herramienta que pide el modelo al agente que la hace.
///
/// Esta tabla es la unica parte del cambio que sabe de las dos partes a la vez.
/// El modelo sigue viendo `guardar_recuerdo` y `consultar_base_vectorial`
/// —cambiar los nombres obligaria a reescribir las instrucciones de la sesion—
/// y el nucleo ve trabajos para `memoria` y para `pc`.
fn traducir(herramienta: &str, args: &Value) -> Result<(String, Value), String> {
    match herramienta {
        "consultar_base_vectorial" => {
            let texto = args
                .get("query")
                .or_else(|| args.get("texto"))
                .and_then(Value::as_str)
                .unwrap_or_default();
            Ok(("memoria".into(), json!({ "accion": "buscar", "texto": texto })))
        }
        "guardar_recuerdo" => {
            let entidad = args.get("entidad").and_then(Value::as_str).unwrap_or_default();
            let visual = args
                .get("descripcion_visual")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let contexto = args.get("contexto").and_then(Value::as_str).unwrap_or_default();
            if entidad.is_empty() {
                return Err("Falta la entidad del recuerdo".into());
            }
            let mut cuerpo = String::new();
            if !contexto.is_empty() {
                cuerpo.push_str(contexto);
            }
            if !visual.is_empty() {
                if !cuerpo.is_empty() {
                    cuerpo.push_str("\n\n");
                }
                cuerpo.push_str("Descripcion visual: ");
                cuerpo.push_str(visual);
            }
            Ok((
                "memoria".into(),
                json!({ "accion": "anotar", "titulo": entidad, "texto": cuerpo }),
            ))
        }
        "guardar_conversacion" => {
            // El titulo lo pone la fecha, y eso lo sabe el agente: una
            // conversacion no tiene nombre hasta que la lees.
            let mensajes = args.get("mensajes").cloned().unwrap_or_else(|| json!([]));
            Ok((
                "memoria".into(),
                json!({ "accion": "conversacion", "mensajes": mensajes }),
            ))
        }
        "controlar_pc" => {
            let accion = args.get("accion").and_then(Value::as_str).unwrap_or_default();
            let parametro = args
                .get("parametro")
                .and_then(Value::as_str)
                .unwrap_or_default();
            Ok(("pc".into(), json!({ "accion": accion, "parametro": parametro })))
        }
        otra => Err(format!("Herramienta desconocida: {otra}")),
    }
}

/// Convierte el resultado de un trabajo en la frase que lee el modelo.
fn resumir(resultado: &Value) -> String {
    if let Some(texto) = resultado.get("texto").and_then(Value::as_str) {
        return texto.to_string();
    }

    if let Some(notas) = resultado.get("notas").and_then(Value::as_array) {
        if notas.is_empty() {
            return "No hay ninguna nota sobre eso en la memoria.".into();
        }
        let lineas: Vec<String> = notas
            .iter()
            .map(|n| {
                format!(
                    "- {}: {}",
                    n.get("titulo").and_then(Value::as_str).unwrap_or("(sin titulo)"),
                    n.get("extracto").and_then(Value::as_str).unwrap_or("")
                )
            })
            .collect();
        return lineas.join("\n");
    }

    if let Some(ruta) = resultado.get("ruta").and_then(Value::as_str) {
        return format!("Guardado en {ruta}.");
    }

    resultado.to_string()
}

async fn pedir_json(
    peticion: reqwest::RequestBuilder,
) -> Result<Value, String> {
    let respuesta = peticion
        .send()
        .await
        .map_err(|e| format!("No se pudo hablar con el nucleo: {e}. ¿Esta arrancado?"))?;

    let estado = respuesta.status();
    let cuerpo: Value = respuesta
        .json()
        .await
        .map_err(|e| format!("Respuesta ilegible del nucleo: {e}"))?;

    if !estado.is_success() {
        let detalle = cuerpo
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("sin detalle");
        return Err(format!("El nucleo respondio {estado}: {detalle}"));
    }
    Ok(cuerpo)
}

/// Encola un trabajo en el nucleo y espera su resultado.
///
/// El origen es `voz` porque esta cara es la de la llamada: asi se distingue en
/// la cola de la web lo que pediste hablando de lo que escribiste.
#[tauri::command]
pub async fn ejecutar_herramienta(
    app: AppHandle,
    tool_name: String,
    argumentos: String,
) -> Result<String, String> {
    let args: Value =
        serde_json::from_str(&argumentos).map_err(|e| format!("Argumentos JSON invalidos: {e}"))?;
    let (agente, peticion) = traducir(&tool_name, &args)?;

    let token = token(&app)?;
    let cliente = reqwest::Client::new();
    let base = base_url();

    let creado = pedir_json(
        cliente
            .post(format!("{base}/trabajos"))
            .bearer_auth(&token)
            .json(&json!({ "agente": agente, "peticion": peticion, "origen": "voz" })),
    )
    .await?;

    let id = creado
        .get("id")
        .and_then(Value::as_i64)
        .ok_or("El nucleo no devolvio el identificador del trabajo")?;

    let limite = Instant::now() + ESPERA_MAXIMA;
    loop {
        let trabajo = pedir_json(
            cliente
                .get(format!("{base}/trabajos/{id}"))
                .bearer_auth(&token),
        )
        .await?;

        match trabajo.get("estado").and_then(Value::as_str).unwrap_or("") {
            "hecho" => {
                let vacio = json!({});
                return Ok(resumir(trabajo.get("resultado").unwrap_or(&vacio)));
            }
            "fallido" => {
                return Err(trabajo
                    .get("error")
                    .and_then(Value::as_str)
                    .unwrap_or("El trabajo fallo")
                    .to_string())
            }
            "cancelado" | "rechazado" => return Err("El trabajo se cerro sin ejecutarse".into()),
            "esperando" => {
                // El agente ha parado a pedir un si. No se espera aqui: la
                // pregunta esta en la web y en Telegram, y el modelo tiene que
                // poder seguir hablando mientras tanto.
                let pregunta = trabajo
                    .get("confirmacion")
                    .and_then(|c| c.get("resumen"))
                    .and_then(Value::as_str)
                    .unwrap_or("una confirmacion");
                return Ok(format!(
                    "Queda pendiente de que lo confirmes: {pregunta} (trabajo #{id})."
                ));
            }
            _ => {}
        }

        if Instant::now() >= limite {
            return Ok(format!(
                "Sigue en marcha; te lo cuento cuando termine (trabajo #{id})."
            ));
        }
        sleep(SONDEO).await;
    }
}

/// Comprueba que el nucleo esta vivo **y que el token vale** antes de la primera
/// herramienta.
///
/// Antes esto arrancaba el proceso de Python y pagaba por adelantado los 3,7 s
/// de importar llama_index y chromadb (H-12). Ahora el nucleo ya esta
/// encendido —es su razon de ser— y esto solo sirve para avisar pronto si algo
/// no esta, en vez de descubrirlo a mitad de una frase.
///
/// Se pregunta por `/trabajos` y no por `/salud`: **`/salud` es publica**, asi
/// que contestaba 200 con un token caducado o equivocado y la comprobacion
/// dejaba pasar justo el fallo que existe para detectar. `?limite=1` la hace
/// tan barata como la otra.
#[tauri::command]
pub async fn precalentar_herramientas(app: AppHandle) -> Result<(), String> {
    let token = token(&app)?;
    let cliente = reqwest::Client::new();
    pedir_json(
        cliente
            .get(format!("{}/trabajos?limite=1", base_url()))
            .bearer_auth(&token),
    )
    .await?;
    Ok(())
}
