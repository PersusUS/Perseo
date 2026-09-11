//! Marca de "Perseo esta abierto", para que nadie tenga que adivinarlo.
//!
//! El detector de aplausos necesita saber si la aplicacion ya esta viva: si lo
//! esta, basta con despertarla, y si no, hay que lanzarla. Hasta ahora lo
//! averiguaba buscando `temp-app.exe` en el `tasklist`, lo que ataba dos cosas
//! que no deberian estarlo — **el nombre del binario y el funcionamiento del
//! detector**. Renombrar el paquete rompia la deteccion en silencio: la app
//! seguia arrancando, y de pronto cada aplauso abria una segunda instancia. Ver
//!.
//!
//! Ahora la aplicacion deja un fichero con su PID al arrancar y lo borra al
//! salir. Quien quiera saber si esta viva lee el numero y pregunta al sistema
//! por ese proceso. Da igual como se llame el ejecutable.
//!
//! **Un PID rancio no enga~na a nadie**: si la app muere de mala manera el
//! fichero se queda, y por eso lo que se comprueba es el proceso, no el fichero.

use std::path::PathBuf;

use tauri::{AppHandle, Manager};

/// Nombre del fichero. Vive junto al marcador de autollamada, y por lo mismo:
/// es estado efimero, no configuracion.
pub const FICHERO: &str = ".perseo-app.pid";

/// Donde se escribe. Se prefiere el directorio de configuracion de la
/// aplicacion, y en `tauri dev` tambien el arbol de fuentes — que es donde mira
/// el detector cuando se trabaja sin instalar.
pub fn rutas(app: &AppHandle) -> Vec<PathBuf> {
    let mut rutas = Vec::new();
    if let Ok(dir) = app.path().app_config_dir() {
        let _ = std::fs::create_dir_all(&dir);
        rutas.push(dir.join(FICHERO));
    }
    rutas.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join(FICHERO),
    );
    rutas
}

/// Deja la marca. Se llama al arrancar.
pub fn anunciar(app: &AppHandle) {
    let pid = std::process::id().to_string();
    for ruta in rutas(app) {
        if let Err(e) = std::fs::write(&ruta, &pid) {
            // Que no se pueda escribir una de las rutas no es motivo para no
            // arrancar: el detector tiene la otra, y sin ninguna solo pierde la
            // optimizacion de "ya esta abierta".
            eprintln!("[presencia] No se pudo escribir {}: {e}", ruta.display());
        }
    }
}

/// Quita la marca. Se llama al salir por el menu de la bandeja.
pub fn retirar(app: &AppHandle) {
    for ruta in rutas(app) {
        let _ = std::fs::remove_file(ruta);
    }
}
