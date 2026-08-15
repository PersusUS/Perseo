//! Icono en la bandeja del sistema.
//!
//! Perseo deja de ser una app que se abre y se cierra para pasar a estar
//! encendido. Es lo que pide la Fase C: si el detector de palabra clave tiene
//! que poder invocarlo, el proceso no puede morir cada vez que se cierra la
//! ventana — arrancar de cero cuesta segundos, y la promesa es responder en
//! menos de uno.
//!
//! Por eso cerrar la ventana la esconde en vez de terminar el programa. Salir
//! de verdad se hace desde el menu de la bandeja, que es donde el usuario
//! espera encontrarlo.

use tauri::{
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    AppHandle, Manager, Window,
};

/// Identificador de la ventana principal, tal y como la nombra Tauri por
/// defecto en `tauri.conf.json`.
const VENTANA: &str = "main";

/// Instala el icono y su menu. Se llama una sola vez, al arrancar.
pub fn instalar(app: &AppHandle) -> tauri::Result<()> {
    let mostrar = MenuItem::with_id(app, "mostrar", "Mostrar Perseo", true, None::<&str>)?;
    let ocultar = MenuItem::with_id(app, "ocultar", "Ocultar", true, None::<&str>)?;
    let salir = MenuItem::with_id(app, "salir", "Salir", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&mostrar, &ocultar, &salir])?;

    TrayIconBuilder::with_id("perseo")
        .icon(app.default_window_icon().cloned().ok_or_else(|| {
            tauri::Error::AssetNotFound("no hay icono por defecto para la bandeja".into())
        })?)
        .tooltip("Perseo")
        .menu(&menu)
        // El menu solo con el boton derecho: el izquierdo se reserva para
        // mostrar la ventana, que es lo que espera cualquiera en Windows.
        .show_menu_on_left_click(false)
        .on_menu_event(|app, evento| match evento.id.as_ref() {
            "mostrar" => mostrar_ventana(app),
            "ocultar" => {
                if let Some(ventana) = app.get_webview_window(VENTANA) {
                    let _ = ventana.hide();
                }
            }
            // Aqui si se sale de verdad. Es la unica via, y por eso esta en el
            // menu: cerrar la ventana ya no termina el proceso.
            "salir" => {
                // Se quita la marca de presencia antes de irse: si se quedara,
                // el detector tendria que descubrir por su cuenta que el PID ya
                // no existe. Funciona igual, pero tarda un instante mas.
                crate::presencia::retirar(app);
                app.exit(0)
            }
            _ => {}
        })
        .on_tray_icon_event(|bandeja, evento| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = evento
            {
                mostrar_ventana(bandeja.app_handle());
            }
        })
        .build(app)?;

    Ok(())
}

/// Saca la ventana del escondite y le da el foco.
///
/// `show` sola no basta en Windows: si la ventana estaba minimizada se queda
/// ahi, y sin `set_focus` aparece detras de lo que hubiera delante.
pub fn mostrar_ventana(app: &AppHandle) {
    if let Some(ventana) = app.get_webview_window(VENTANA) {
        let _ = ventana.unminimize();
        let _ = ventana.show();
        let _ = ventana.set_focus();
    }
}

/// Esconde la ventana en lugar de dejar que se cierre.
///
/// Devuelve `true` si se ha ocultado, para que quien llame sepa que tiene que
/// cancelar el cierre.
/// `Window` y no `WebviewWindow`: es lo que entrega `on_window_event`.
pub fn ocultar_en_vez_de_cerrar(ventana: &Window) -> bool {
    if ventana.label() != VENTANA {
        return false;
    }
    ventana.hide().is_ok()
}
