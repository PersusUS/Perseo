//! El panel, hablando con el nucleo desde Rust.
//!
//! Perseo tenia dos caras que no se hablaban: en el PC la ventana de la llamada
//! y nada mas, y para ver la cola habia que abrir el navegador.
//!
//! El primer intento fue **hospedar** la interfaz del nucleo en una ventana
//! aparte. No vale, y por dos motivos que no se arreglan con codigo:
//!
//! 1. **La CSP.** `tauri.conf.json` declara `script-src 'self'`, y esa interfaz
//!    es un solo fichero con su `<script>` en linea. Pagina cargada, script
//!    bloqueado, pantalla en blanco.
//! 2. **La cookie.** La sesion del nucleo es `SameSite=Strict`. Metida en un
//!    `<iframe>` dentro de la app, el navegador trata como cross-site hasta las
//!    peticiones que la pagina hace a su propio origen, asi que la cookie no
//!    viaja y no hay forma de autenticarse.
//!
//! Asi que el panel del escritorio es una vista de React, y habla con el nucleo
//! **por aqui**. Eso cuesta tener dos implementaciones de la misma pantalla —la
//! de React y el fichero suelto del movil— y es un coste real: lo que se cambie
//! en una hay que llevarlo a la otra. A cambio se gana lo que el usuario pedia,
//! una sola ventana, y algo que no estaba en la lista: **en el PC no se pega
//! ningun token**, porque el token lo lee Rust del disco y no pasa por el
//! frontend.
//!
//! Los comandos son concretos y no un proxy generico a `/{ruta}`: un proxy
//! dejaria que cualquier cosa del frontend llamara a cualquier ruta del nucleo,
//! y la lista de lo que el panel necesita cabe en una pantalla.

use serde_json::{json, Value};
use tauri::AppHandle;

use crate::nucleo::{base_url, pedir_json, token};

/// GET autenticado contra el nucleo.
async fn traer(app: &AppHandle, ruta: &str) -> Result<Value, String> {
    let token = token(app)?;
    let cliente = reqwest::Client::new();
    pedir_json(cliente.get(format!("{}{ruta}", base_url())).bearer_auth(&token)).await
}

/// POST autenticado contra el nucleo.
async fn mandar(app: &AppHandle, ruta: &str, cuerpo: Value) -> Result<Value, String> {
    let token = token(app)?;
    let cliente = reqwest::Client::new();
    pedir_json(
        cliente
            .post(format!("{}{ruta}", base_url()))
            .bearer_auth(&token)
            .json(&cuerpo),
    )
    .await
}

/// De que esta capado el sistema hoy: piezas, cuota, disparadores.
#[tauri::command]
pub async fn panel_estado(app: AppHandle) -> Result<Value, String> {
    traer(&app, "/estado").await
}

/// Los ultimos trabajos de la cola.
#[tauri::command]
pub async fn panel_trabajos(app: AppHandle, limite: u32) -> Result<Value, String> {
    traer(&app, &format!("/trabajos?limite={limite}")).await
}

/// Uno concreto, para seguir un trabajo recien encolado.
#[tauri::command]
pub async fn panel_trabajo(app: AppHandle, id: i64) -> Result<Value, String> {
    traer(&app, &format!("/trabajos/{id}")).await
}

/// Aprobar, rechazar o cancelar.
///
/// La decision se valida aqui: sin esto, `decision` seria un trozo de URL que
/// elige el frontend, y eso es una ruta abierta con otro nombre.
#[tauri::command]
pub async fn panel_responder(app: AppHandle, id: i64, decision: String) -> Result<Value, String> {
    if !["aprobar", "rechazar", "cancelar"].contains(&decision.as_str()) {
        return Err(format!("Decision desconocida: {decision}"));
    }
    mandar(&app, &format!("/trabajos/{id}/{decision}"), json!({})).await
}

/// Que se ha hecho con cada correo triado. Lo que no salga esta pendiente.
#[tauri::command]
pub async fn panel_correos(app: AppHandle) -> Result<Value, String> {
    traer(&app, "/correos").await
}

/// Marca un correo como atendido, descartado, o de vuelta a pendiente.
///
/// El estado se valida aqui por el mismo motivo que la decision de
/// `panel_responder`: si lo elige el frontend entero, la ruta la escribe el
/// frontend.
#[tauri::command]
pub async fn panel_marcar_correo(
    app: AppHandle,
    id: String,
    estado: String,
) -> Result<Value, String> {
    if !["atendido", "descartado", "pendiente"].contains(&estado.as_str()) {
        return Err(format!("Estado desconocido: {estado}"));
    }
    // El id de Gmail es hexadecimal, asi que en vez de arrastrar una
    // dependencia para escapar la URL se comprueba que sea lo que dice ser.
    // Cualquier otra cosa seria un trozo de ruta escrito desde el frontend.
    if id.is_empty() || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
        return Err(format!("Id de correo no valido: {id}"));
    }
    mandar(&app, &format!("/correos/{id}/estado"), json!({ "estado": estado })).await
}

/// Los otros proyectos que se pueden abrir desde el panel.
#[tauri::command]
pub async fn panel_proyectos(app: AppHandle) -> Result<Value, String> {
    traer(&app, "/proyectos").await
}

/// Abre uno. Por aqui viaja **cual**, nunca que ejecutar: la lista vive en
/// `<datos>/proyectos.json` y la valida el nucleo.
#[tauri::command]
pub async fn panel_abrir_proyecto(app: AppHandle, id: String) -> Result<Value, String> {
    if id.is_empty() || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
        return Err(format!("Id de proyecto no valido: {id}"));
    }
    mandar(&app, &format!("/proyectos/{id}/abrir"), json!({})).await
}

/// Encola un trabajo para un agente. Lo usa la pestana de memoria.
#[tauri::command]
pub async fn panel_encolar(
    app: AppHandle,
    agente: String,
    peticion: Value,
) -> Result<Value, String> {
    mandar(
        &app,
        "/trabajos",
        json!({ "agente": agente, "peticion": peticion, "origen": "texto" }),
    )
    .await
}

/// Escribirle a Perseo por texto. El router decide si contesta o encola.
///
/// Es la misma puerta que usa el movil, no un atajo del escritorio: por eso va
/// a `/mensaje` y no a `/trabajos`. Hablar y escribir acaban en la misma cola.
#[tauri::command]
pub async fn panel_mensaje(app: AppHandle, texto: String) -> Result<Value, String> {
    mandar(&app, "/mensaje", json!({ "texto": texto, "origen": "texto" })).await
}

/// Enciende o apaga el modo confianza.
#[tauri::command]
pub async fn panel_confianza(app: AppHandle, minutos: Option<f64>) -> Result<Value, String> {
    let cuerpo = match minutos {
        Some(m) => json!({ "minutos": m }),
        None => json!({ "activo": false }),
    };
    mandar(&app, "/confianza", cuerpo).await
}
