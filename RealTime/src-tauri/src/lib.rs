mod autollamada;
mod bandeja;
mod commands;
mod puente;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            bandeja::instalar(app.handle())?;
            autollamada::vigilar(app.handle().clone());
            Ok(())
        })
        // Cerrar la ventana esconde Perseo, no lo mata. Sin esto, la bandeja no
        // serviria de nada: el proceso terminaria en cuanto se cerrara la
        // ventana y el detector de palabra clave no tendria a quien invocar.
        .on_window_event(|ventana, evento| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = evento {
                if bandeja::ocultar_en_vez_de_cerrar(ventana) {
                    api.prevent_close();
                }
            }
        })
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
