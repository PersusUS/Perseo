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

use serde::Serialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Mutex;
use tauri::{AppHandle, Manager};

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

/// El paso a paso de un encargo de codigo: el suyo y el de sus subagentes.
///
/// Es lo que convierte un "HECHO (16 vueltas)" en algo que se puede depurar:
/// que herramienta uso, que contesto cada una y que hizo cada subagente.
#[tauri::command]
pub async fn panel_actividad(app: AppHandle, id: i64) -> Result<Value, String> {
    traer(&app, &format!("/trabajos/{id}/actividad")).await
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

/// Lo que la corteza de una ventana necesita para montarse.
///
/// La «corteza» es la explicación de esta sección entera: las ventanas de
/// proyecto y del grafo cargan páginas **remotas** (Armario en el 8123,
/// CVScraper en el 5173, MAGI en el 8050, el grafo en el núcleo), y a una
/// página remota no se le puede colgar una barra de la casa — lo intentado el
/// 2026-08-24 y desmontado: ni el puente IPC llega (trampa del handoff §6.17)
/// ni hay forma honesta de pulsar un botón que minimice. Así que la ventana
/// carga **nuestra** página —una vista mínima que solo existe para esto— y el
/// proyecto va dentro de un `<iframe>`. La barra es local, con IPC de sobra:
/// se arrastra, minimiza, pone pantalla completa y cierra, y se pinta del
/// color que el proyecto declare en `proyectos.json`.
#[derive(Clone, Serialize)]
pub struct CortezaDatos {
    pub url: String,
    pub titulo: String,
    /// #rrggbb o cadena vacía = monocromo de la casa.
    pub color: String,
}

/// Los datos de cada corteza viva, por etiqueta de ventana.
///
/// Viajan por aquí y no por la URL a propósito: el grafo lleva el token del
/// núcleo dentro, y meterlo en la dirección de la ventana lo dejaría en el
/// historial y al alcance de cualquier JS remoto que mirara `location`. Así
/// solo lo lee el comando que la propia corteza pregunta, y acaba dentro del
/// `src` de su iframe — que es donde ya estaba cuando la ventana entera
/// navegaba directa.
#[derive(Default)]
pub struct EstadoCorteza(pub Mutex<HashMap<String, CortezaDatos>>);

fn _color_valido(crudo: &str) -> Result<String, String> {
    let limpio = crudo.trim();
    if limpio.is_empty() {
        return Ok(String::new());
    }
    let bien = limpio.len() == 7
        && limpio.starts_with('#')
        && limpio[1..].chars().all(|c| c.is_ascii_hexdigit());
    if bien {
        Ok(limpio.to_ascii_lowercase())
    } else {
        Err(format!("Color no válido: {crudo}"))
    }
}

/// Crea una ventana-corteza con los datos dados. Es el cuerpo común de
/// `ventana_proyecto` y `ventana_grafo`: misma barra, mismo ciclo, otra ficha.
async fn _ventana_corteza(
    app: &AppHandle,
    raiz_etiqueta: &str,
    titulo: &str,
    url: String,
    color: String,
    ancho: f64,
    alto: f64,
) -> Result<(), String> {
    let destino: tauri::Url = url.parse().map_err(|e| format!("URL invalida: {e}"))?;
    // Dentro de un iframe un esquema raro es peor que fuera: `javascript:` o
    // `file:` en el `src` de nuestra propia página ya no sería «mirar otra
    // dirección», sería ejecutar código local. Solo http/https, como el núcleo.
    if !matches!(destino.scheme(), "http" | "https") {
        return Err(format!(
            "Esquema no permitido: {} (solo http/https)",
            destino.scheme()
        ));
    }

    // Etiqueta única por apertura: abrir dos veces el mismo proyecto son dos
    // ventanas, no un choque de nombres.
    let etiqueta = format!(
        "{raiz_etiqueta}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or_default()
    );

    let estado = app.state::<EstadoCorteza>();
    estado
        .0
        .lock()
        .map_err(|e| e.to_string())?
        .insert(etiqueta.clone(), CortezaDatos {
            url: destino.to_string(),
            titulo: titulo.to_string(),
            color,
        });

    let resultado = tauri::WebviewWindowBuilder::new(
        app,
        &etiqueta,
        tauri::WebviewUrl::App("index.html".into()),
    )
    .title(titulo)
    .inner_size(ancho, alto)
    .decorations(false)
    .resizable(true)
    .focused(true)
    .build();

    if let Err(e) = resultado {
        // Sin ventana no hay corteza que la pida: fuera, para no acumular.
        if let Ok(mut mapa) = estado.0.lock() {
            mapa.remove(&etiqueta);
        }
        return Err(e.to_string());
    }
    Ok(())
}

/// Abre la ventana de un proyecto: la corteza de la casa con su app dentro.
///
/// El señor Persus pidio «en su propia ventana individual», delante de todo y
/// sin navegadores; y luego, viendo la barra gris de Windows encima de cada
/// app, que llevara **el marco de Perseo y el color del proyecto**. Por HTTP
/// viaja cuál (el `url`/`titulo` que el núcleo ya validó contra
/// `proyectos.json`) y su color; qué ejecutar sigue viviendo en el disco.
#[tauri::command]
pub async fn ventana_proyecto(
    app: AppHandle,
    url: String,
    titulo: String,
    color: String,
    ancho: f64,
    alto: f64,
) -> Result<(), String> {
    let color = _color_valido(&color)?;
    _ventana_corteza(&app, "proyecto", &titulo, url, color, ancho, alto).await
}

/// La ventana del grafo del segundo cerebro: el vault como constelación.
///
/// Misma corteza que los proyectos — el marco de la casa, monocromo, que es su
/// color de identidad al no ser un proyecto del disco. El token viaja en los
/// datos internos de la corteza, nunca por la URL de la ventana (ver
/// `EstadoCorteza`).
#[tauri::command]
pub async fn ventana_grafo(app: AppHandle, ancho: f64, alto: f64) -> Result<(), String> {
    let url = format!("{}/grafo?t={}", base_url(), token(&app)?);
    _ventana_corteza(
        &app,
        "grafo",
        "Segundo cerebro · Grafo",
        url,
        String::new(),
        ancho,
        alto,
    )
    .await
}

/// Lo que la corteza de ESTA ventana tiene que montar.
///
/// La página local de la corteza lo pregunta nada más arrancar, y Rust contesta
/// con la ficha de SU etiqueta — nunca con la de otra ventana. `None` significa
/// «esta ventana no tiene ficha»: solo pasa si algo fue mal al crearla, y la
/// corteza lo dice en pantalla en vez de quedarse negra.
#[tauri::command]
pub async fn corteza_parametros(
    webview: tauri::WebviewWindow,
    app: AppHandle,
) -> Result<Option<CortezaDatos>, String> {
    let estado = app.state::<EstadoCorteza>();
    let mapa = estado.0.lock().map_err(|e| e.to_string())?;
    Ok(mapa.get(webview.label()).cloned())
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

/// Deja en el nucleo una copia del seguimiento de habitos.
///
/// Los habitos viven en el `localStorage` de esta ventana —marcar una casilla no
/// puede depender de que el nucleo este encendido— y eso dejaba fuera a media
/// casa: el Perseo de la llamada corre AQUI y podia leerlos, pero el chat
/// escrito y los agentes son Python y no ven dentro de un navegador.
///
/// Lo que viaja es el texto ya redactado, no las casillas: quien cuenta es
/// `src/lib/habitos.ts` y nadie mas (ver `perseo_core/habitos.py`).
///
/// Un fallo aqui no es un fallo de la pantalla: si el nucleo esta apagado, el
/// señor Persus sigue marcando sus habitos igual y la copia se manda con el
/// cambio siguiente. Por eso el frontend se traga el error en vez de enseñarlo.
#[tauri::command]
pub async fn habitos_espejo(app: AppHandle, texto: String, foto: Value) -> Result<Value, String> {
    mandar(&app, "/habitos", json!({ "texto": texto, "foto": foto })).await
}

/// Deja en el nucleo una copia del tablero de tareas.
///
/// Mismo reparto que los habitos y por el mismo motivo: el corcho vive en el
/// `localStorage` de esta ventana —mover una nota no puede depender de que el
/// nucleo este encendido— y el chat escrito, el triaje y los agentes son
/// Python y no ven dentro de un navegador.
///
/// Lo que viaja es el texto ya redactado, no las notas: quien cuenta es
/// `src/lib/tareas.ts` y nadie mas (ver `perseo_core/tareas.py`).
///
/// Un fallo aqui tampoco es un fallo de la pantalla: con el nucleo apagado el
/// señor Persus sigue moviendo sus notas y la copia sale con el cambio
/// siguiente. Por eso el frontend se traga el error en vez de enseñarlo.
#[tauri::command]
pub async fn tareas_espejo(app: AppHandle, texto: String, foto: Value) -> Result<Value, String> {
    mandar(&app, "/tareas", json!({ "texto": texto, "foto": foto })).await
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

// --------------------------------------------------------------------------- //
// El chat escrito
//
// La misma escuela que el resto del panel: comandos concretos y no un proxy
// a `/{ruta}`. El chat vive en `/chat*` del núcleo; aquí solo se traduce lo
// que la vista necesita — listar sesiones, leer una, hablar, borrar — con los
// mismos validadores de siempre en los campos que viajan dentro de una URL.
// --------------------------------------------------------------------------- //

/// DELETE autenticado contra el nucleo.
async fn quitar(app: &AppHandle, ruta: &str) -> Result<Value, String> {
    let token = token(app)?;
    let cliente = reqwest::Client::new();
    pedir_json(
        cliente
            .delete(format!("{}{ruta}", base_url()))
            .bearer_auth(&token),
    )
    .await
}

fn _id_sesion(id: i64) -> Result<i64, String> {
    if id <= 0 {
        return Err(format!("Identificador de conversación no válido: {id}"));
    }
    Ok(id)
}

/// Las conversaciones, la más reciente primero.
#[tauri::command]
pub async fn chat_sesiones(app: AppHandle) -> Result<Value, String> {
    traer(&app, "/chat").await
}

/// Una conversación nueva. Sin título: lo pone el primer mensaje.
#[tauri::command]
pub async fn chat_crear(app: AppHandle) -> Result<Value, String> {
    mandar(&app, "/chat", json!({})).await
}

/// Una conversación entera, con su semáforo de turno. Es lo que sondea la
/// vista mientras Perseo está escribiendo.
#[tauri::command]
pub async fn chat_sesion(app: AppHandle, id: i64) -> Result<Value, String> {
    let id = _id_sesion(id)?;
    traer(&app, &format!("/chat/{id}")).await
}

/// Borra una conversación y sus mensajes. El núcleo rechaza borrar una que
/// esté a mitad de turno; ese error llega tal cual a la vista.
#[tauri::command]
pub async fn chat_borrar(app: AppHandle, id: i64) -> Result<Value, String> {
    let id = _id_sesion(id)?;
    quitar(&app, &format!("/chat/{id}")).await
}

/// Le escribe al Perseo del texto. Encola un turno del agente `chat` y
/// contesta al momento con los identificadores; el contenido va creciendo en
/// `GET /chat/{id}`, que es lo que esta misma vista estará sondeando.
#[tauri::command]
pub async fn chat_hablar(app: AppHandle, id: i64, texto: String) -> Result<Value, String> {
    let id = _id_sesion(id)?;
    if texto.trim().is_empty() {
        return Err("No hay nada que enviar".into());
    }
    mandar(
        &app,
        &format!("/chat/{id}/hablar"),
        json!({ "texto": texto.trim() }),
    )
    .await
}
