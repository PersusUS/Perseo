mod commands;
mod puente;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_store::Builder::new().build())
        .plugin(tauri_plugin_screenshots::init())
        .plugin(tauri_plugin_opener::init())
        .manage(puente::EstadoPuente::default())
        .invoke_handler(tauri::generate_handler![
            commands::capture_screen_base64,
            commands::obtener_api_key,
            commands::guardar_api_key,
            commands::obtener_ajuste,
            commands::guardar_ajuste,
            commands::consumir_autollamada,
            puente::ejecutar_herramienta_python,
            puente::precalentar_herramientas
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
