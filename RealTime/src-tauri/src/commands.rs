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

#[tauri::command]
pub async fn capture_screen_base64(quality: u8) -> Result<String, String> {
    let monitors = Monitor::all().map_err(|e| e.to_string())?;
    let monitor = monitors.first().ok_or("No monitor found")?;
    let image = monitor.capture_image().map_err(|e| e.to_string())?;
    
    // Resize to reduce bandwidth (1280x720 is plenty for Gemini)
    let resized = image::imageops::resize(
        &image, 1280, 720, image::imageops::FilterType::Triangle
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
/// Orden de busqueda: el almacen local primero (donde la deja el usuario desde
/// la pantalla de Ajustes) y, si esta vacio, la variable de entorno
/// GEMINI_API_KEY del sistema, leida en tiempo de ejecucion.
///
/// El motivo de que esto viva en Rust y no en el frontend: cualquier variable
/// con prefijo VITE_ la incrusta Vite dentro del JavaScript compilado, asi que
/// la clave quedaba en claro dentro del .exe. Ver H-17.
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

    Ok(std::env::var("GEMINI_API_KEY").unwrap_or_default())
}

/// Guarda la clave de API en el almacen local y la persiste en disco.
#[tauri::command]
pub fn guardar_api_key(app: AppHandle, clave: String) -> Result<(), String> {
    guardar_ajuste(app, CLAVE_API.to_string(), serde_json::Value::String(clave))
}

/// Lee un ajuste cualquiera del almacen local.
///
/// Los Ajustes solo mutaban un objeto en memoria, asi que la voz y el prompt se
/// perdian al cerrar la aplicacion. Ver H-08.
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

/// Consume la senal de autollamada dejada por el detector de aplausos.
///
/// Devuelve true una sola vez: el fichero marcador se borra al leerlo. Antes
/// esto era `src/autocall.json`, importado estaticamente por React, con dos
/// problemas: Vite congela el valor al compilar (asi que en produccion el
/// disparo por aplausos no funcionaba) y nadie lo devolvia a false, de modo que
/// toda apertura manual entraba en llamada sola. Ver H-09.
#[tauri::command]
pub fn consumir_autollamada(app: AppHandle) -> bool {
    for ruta in rutas_marcador_autollamada(&app) {
        if ruta.is_file() {
            let _ = std::fs::remove_file(&ruta);
            return true;
        }
    }
    false
}

pub fn rutas_marcador_autollamada(app: &AppHandle) -> Vec<std::path::PathBuf> {
    use tauri::Manager;

    let mut rutas = Vec::new();
    if let Ok(dir) = app.path().app_config_dir() {
        rutas.push(dir.join(MARCADOR_AUTOLLAMADA));
    }
    // Arbol de fuentes, para `tauri dev`.
    rutas.push(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join(MARCADOR_AUTOLLAMADA),
    );
    rutas
}

// `ejecutar_herramienta_python` vive ahora en `puente.rs`, sobre un proceso de
// Python persistente. La version anterior lanzaba un interprete nuevo en cada
// llamada: 7,5 s medidos, contra un timeout de 10 s. Ver H-10 a H-15.
