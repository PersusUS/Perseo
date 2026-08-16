//! La ventana del panel: el mismo tablero que se ve en el movil, dentro de la app.
//!
//! Perseo tenia dos caras que no se hablaban. En el PC, la ventana de la llamada
//! —avatar, ola, camara— y nada mas: para saber que habia en la cola, si el
//! correo se habia triado o si Ollama seguia en pie, habia que abrir el
//! navegador. En el movil, justo lo contrario.
//!
//! Esto las junta, y lo hace **hospedando en vez de reescribiendo**. Se abre una
//! ventana con `perseo_core/interfaz/index.html`, servida por el propio nucleo.
//! Es la misma interfaz, el mismo fichero y el mismo codigo que en el iPhone.
//!
//! La alternativa era portar las cuatro pestañas a React aqui dentro, y es
//! justo lo que no se hace: serian dos paneles haciendo lo mismo, que divergen a
//! la primera prisa. Este proyecto ya pago esa factura una vez —dos memorias,
//! dos formas de hacer lo mismo, y lo que pedias por voz no existia para el
//! movil— y de ahi sale la regla de que las caras no piensan.
//!
//! Esta ventana **no tiene ningun permiso de Tauri**, y no por descuido:
//! `capabilities/default.json` se aplica solo a `windows: ["main"]`, asi que la
//! del panel carga contenido HTTP sin acceso a `invoke` ni a ningun comando. Es
//! lo que debe ser — ahi dentro solo hay una pagina web hablando con el nucleo
//! por HTTP, igual que en el navegador del movil.
//!
//! Sobre el token: **no se inyecta desde aqui**. La primera vez se pega en la
//! propia pantalla, igual que en el movil, y el nucleo lo canjea por una cookie
//! que la ventana conserva entre arranques. Meterlo por `initialization_script`
//! seria dejar el token dentro del JavaScript de una ventana, que es
//! exactamente lo que `nucleo.rs` evita al leerlo del disco desde Rust.

use tauri::{AppHandle, Manager, WebviewUrl, WebviewWindowBuilder};

/// Etiqueta de la ventana. Distinta de `main` a proposito: `bandeja.rs` solo
/// esconde la principal al cerrarla, asi que cerrar el panel lo cierra de
/// verdad, que es lo que uno espera de un panel.
pub const VENTANA: &str = "panel";

/// Abre el panel, o lo trae al frente si ya estaba abierto.
#[tauri::command]
pub fn abrir_panel(app: AppHandle) -> Result<(), String> {
    if let Some(ventana) = app.get_webview_window(VENTANA) {
        let _ = ventana.unminimize();
        ventana.show().map_err(|e| e.to_string())?;
        return ventana.set_focus().map_err(|e| e.to_string());
    }

    let destino = crate::nucleo::base_url();
    let url = destino
        .parse()
        .map_err(|e| format!("La direccion del nucleo no vale ({destino}): {e}"))?;

    WebviewWindowBuilder::new(&app, VENTANA, WebviewUrl::External(url))
        .title("Perseo — panel")
        // Estrecha y alta: la interfaz esta pensada para el movil primero, y con
        // una ventana ancha las tarjetas se estiran hasta quedar ilegibles.
        .inner_size(460.0, 860.0)
        .min_inner_size(360.0, 480.0)
        .build()
        .map_err(|e| format!("No se pudo abrir el panel: {e}"))?;

    Ok(())
}
