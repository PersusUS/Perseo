//! Vigilante del marcador de autollamada.
//!
//! El detector de palabra clave avisa dejando un fichero. Hasta ahora eso valia
//! porque la app se lanzaba en ese momento y lo leia al arrancar; con la app
//! viviendo en la bandeja, el arranque ya paso hace horas y nadie mira el
//! fichero. La palabra clave dejaria de servir justo al hacer permanente la
//! aplicacion.
//!
//! Asi que se vigila. Cuando aparece el marcador: se borra, se saca la ventana
//! del escondite y se avisa al frontend por evento. La comprobacion es un
//! `is_file` cada medio segundo, que no se nota.
//!
//! El comando `consumir_autollamada` se queda como estaba, para el caso de que
//! el marcador ya existiera antes de arrancar.

use std::time::Duration;

use tauri::{AppHandle, Emitter};

use crate::{bandeja, commands};

/// Evento que escucha el frontend para entrar en llamada.
pub const EVENTO: &str = "perseo://autollamada";

/// Cada cuanto se mira si ha aparecido el marcador.
const INTERVALO: Duration = Duration::from_millis(500);

/// Arranca la vigilancia en segundo plano. No termina hasta que muere la app.
pub fn vigilar(app: AppHandle) {
    tauri::async_runtime::spawn(async move {
        loop {
            tokio::time::sleep(INTERVALO).await;

            let encontrado = commands::rutas_marcador_autollamada(&app)
                .into_iter()
                .find(|ruta| ruta.is_file());

            let Some(ruta) = encontrado else { continue };

            // Se borra antes de avisar: si el frontend tardara en responder, no
            // se debe disparar dos veces por el mismo aplauso.
            if std::fs::remove_file(&ruta).is_err() {
                continue;
            }

            bandeja::mostrar_ventana(&app);
            if let Err(e) = app.emit(EVENTO, ()) {
                eprintln!("[autollamada] no se pudo avisar al frontend: {e}");
            }
        }
    });
}
