use xcap::Monitor;
use image::codecs::jpeg::JpegEncoder;
use base64::{Engine, engine::general_purpose::STANDARD};
use std::io::Cursor;

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
