use xcap::Monitor;
use image::codecs::jpeg::JpegEncoder;
use base64::{Engine, engine::general_purpose::STANDARD};
use std::io::Cursor;
use std::process::Command;
use tauri::AppHandle;
use tauri_plugin_store::StoreExt;

/// Archivo del almacen cifrado-por-plataforma donde vive la configuracion
/// sensible. Nunca se empaqueta en el bundle del frontend.
const ARCHIVO_AJUSTES: &str = "perseo-ajustes.json";
const CLAVE_API: &str = "gemini_api_key";

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
    let store = app.store(ARCHIVO_AJUSTES).map_err(|e| e.to_string())?;
    store.set(CLAVE_API, serde_json::Value::String(clave));
    store.save().map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub async fn ejecutar_herramienta_python(tool_name: String, argumentos: String) -> Result<String, String> {
    // Usamos la ruta absoluta al script runner de Python para evitar problemas de directorios de trabajo
    let runner_path = "C:\\Users\\<usuario>\\Perseo\\TOOLS\\runner.py";
    
    let output = Command::new("python")
        .arg(runner_path)
        .arg(&tool_name)
        .arg(&argumentos)
        .output()
        .map_err(|e| e.to_string())?;

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        Ok(stdout)
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        Err(format!("Error en script Python: {}", stderr))
    }
}
