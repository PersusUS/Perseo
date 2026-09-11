//! El armado de la aplicación: qué módulos hay y qué comandos ve el frontend.
//!
//! Este fichero no tiene lógica propia a propósito. Es el índice: declara los
//! módulos, construye la ventana y registra en `invoke_handler` todo lo que el
//! JavaScript puede llamar. Si un comando existe en `commands.rs`, `panel.rs` o
//! `nucleo.rs` pero no está en esa lista, el frontend recibe un error de
//! comando desconocido y el fallo parece del lado de JavaScript.
//!
//! Dónde vive cada cosa:
//!
//! - `nucleo.rs`   — el cliente del núcleo: traduce herramientas en trabajos.
//! - `panel.rs`    — los comandos `panel_*` y `chat_*` del panel y del chat.
//! - `commands.rs` — pantalla, ajustes cifrados y biometría.
//! - `bandeja.rs`  — el icono de la bandeja; cerrar la ventana la esconde.
//! - `autollamada.rs` y `presencia.rs` — los dos marcadores en disco que
//!   comunican esta app con los scripts de Python de `commands/`.

mod autollamada;
mod bandeja;
mod commands;
mod nucleo;
mod panel;
mod presencia;

use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            bandeja::instalar(app.handle())?;
            // Deja el PID en un fichero para que el detector de aplausos sepa
            // que Perseo ya esta abierto sin tener que adivinar el nombre del
            // ejecutable.
            presencia::anunciar(app.handle());
            autollamada::vigilar(app.handle().clone());
            Ok(())
        })
        // Cerrar la ventana esconde Perseo, no lo mata. Sin esto, la bandeja no
        // serviria de nada: el proceso terminaria en cuanto se cerrara la
        // ventana y el detector de palabra clave no tendria a quien invocar.
        .on_window_event(|ventana, evento| {
            match evento {
                tauri::WindowEvent::CloseRequested { api, .. } => {
                    if bandeja::ocultar_en_vez_de_cerrar(ventana) {
                        api.prevent_close();
                    }
                }
                // Una corteza muerta deja su ficha huerfana: sin esto, cada
                // apertura de proyecto acumularia una entrada diminuta para
                // siempre. Las etiquetas de corteza empiezan por proyecto- o
                // grafo-; las de Perseo, no.
                tauri::WindowEvent::Destroyed => {
                    let etiqueta = ventana.label();
                    if etiqueta.starts_with("proyecto-") || etiqueta.starts_with("grafo-") {
                        if let Some(estado) = ventana
                            .app_handle()
                            .try_state::<panel::EstadoCorteza>()
                        {
                            if let Ok(mut mapa) = estado.0.lock() {
                                mapa.remove(&etiqueta.to_string());
                            }
                        }
                    }
                }
                _ => {}
            }
        })
        .plugin(tauri_plugin_store::Builder::new().build())
        .plugin(tauri_plugin_screenshots::init())
        .plugin(tauri_plugin_opener::init())
        .manage(panel::EstadoCorteza::default())
        .invoke_handler(tauri::generate_handler![
            commands::capture_screen_base64,
            commands::geometria_pantalla,
            commands::obtener_api_key,
            commands::obtener_ajuste,
            commands::guardar_ajuste,
            commands::consumir_autollamada,
            commands::anotar_diagnostico,
            nucleo::ejecutar_herramienta,
            nucleo::precalentar_herramientas,
            nucleo::biometria_estado,
            nucleo::biometria_voz,
            nucleo::biometria_cara,
            nucleo::biometria_enrolar,
            nucleo::biometria_renombrar,
            nucleo::biometria_borrar,
            panel::panel_estado,
            panel::panel_trabajos,
            panel::panel_trabajo,
            panel::panel_actividad,
            panel::panel_responder,
            panel::panel_encolar,
            panel::panel_confianza,
            panel::habitos_espejo,
            panel::tareas_espejo,
            panel::tareas_recoger,
            panel::panel_correos,
            panel::panel_marcar_correo,
            panel::panel_proyectos,
            panel::panel_abrir_proyecto,
            panel::ventana_proyecto,
            panel::ventana_grafo,
            panel::corteza_parametros,
            panel::panel_mensaje,
            panel::chat_sesiones,
            panel::chat_crear,
            panel::chat_sesion,
            panel::chat_borrar,
            panel::chat_hablar
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
