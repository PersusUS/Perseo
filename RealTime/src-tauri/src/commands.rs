use xcap::Monitor;
use image::codecs::jpeg::JpegEncoder;
use base64::{Engine, engine::general_purpose::STANDARD};
use std::io::Cursor;
use std::process::Command;

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
