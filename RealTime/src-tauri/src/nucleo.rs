//! Cliente del nucleo. Sustituye al puente de tuberias hacia `TOOLS/`.
//!
//! Hasta la Fase D, la app llevaba dentro sus propias herramientas: arrancaba
//! `TOOLS/server.py` y le hablaba por tuberias. Eso significaba dos sistemas con
//! dos memorias y dos formas de hacer lo mismo, y que lo que pidieras por voz no
//! existiera para la web del movil.
//!
//! Ahora la app **encola trabajos en el nucleo** y espera el resultado. Es la
//! regla del plan aplicada a la ultima cara que faltaba: *las caras no piensan*.
//! Lo que se pide por voz aparece en la cola de la web, sobrevive a que se
//! cuelgue la llamada, y lo ejecuta el mismo agente que atiende al movil.
//!
//! Sobre la restriccion que la Fase 3 protegia: **la API no expone
//! herramientas**. Aqui no se manda "ejecuta esto", se manda "encola un trabajo
//! para el agente `pc`", y quien decide que hace ese agente vive en el nucleo,
//! detras de su lista blanca. El token va en cada peticion y **no pasa por el
//! frontend**: lo lee este modulo del disco.

use std::path::PathBuf;
use std::time::Duration;

use serde_json::{json, Value};
use tauri::{AppHandle, Manager};
use tokio::time::{sleep, Instant};

/// Cuanto se espera a que un trabajo termine. Es red de seguridad, no
/// presupuesto: `memoria` y `pc` responden en milisegundos. Si salta, el trabajo
/// sigue vivo en la cola y se ve en la web — no se pierde, solo deja de
/// esperarse.
const ESPERA_MAXIMA: Duration = Duration::from_secs(30);

/// Cuanto se espera a un encargo de código. El agente `dev` tiene su propio
/// tope en el núcleo (900 s por defecto) y su propio carril; aquí se espera
/// parecido para poder contar el resultado real cuando termina.
const ESPERA_DEV: Duration = Duration::from_secs(870);

/// Cada cuanto se pregunta por el estado del trabajo.
const SONDEO: Duration = Duration::from_millis(250);

/// Donde vive el nucleo. Lo usa tambien `panel.rs`, que abre su interfaz.
pub(crate) fn base_url() -> String {
    std::env::var("PERSEO_CORE_URL").unwrap_or_else(|_| "http://127.0.0.1:8787".to_string())
}

/// Donde puede estar `perseo_core/datos/`, en orden. Sin rutas absolutas
/// cableadas, igual que hacia el puente con la carpeta TOOLS: la primera vale en
/// pruebas, la segunda en el binario instalado y la tercera en `tauri dev`.
/// Ver H-15. Lo usa tambien `commands.rs` para la clave de Gemini.
pub(crate) fn rutas_datos(app: &AppHandle) -> Vec<PathBuf> {
    let mut rutas = Vec::new();
    if let Ok(datos) = std::env::var("PERSEO_CORE_DATOS") {
        rutas.push(PathBuf::from(datos));
    }
    if let Ok(recursos) = app.path().resource_dir() {
        rutas.push(recursos.join("perseo_core").join("datos"));
    }
    rutas.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("perseo_core")
            .join("datos"),
    );
    rutas
}

/// Localiza `perseo_core/datos/token.txt`.
fn ruta_token(app: &AppHandle) -> Option<PathBuf> {
    rutas_datos(app)
        .into_iter()
        .map(|d| d.join("token.txt"))
        .find(|c| c.is_file())
}

/// El token de acceso al nucleo. La variable de entorno manda sobre el fichero,
/// igual que en `almacen.py`.
pub(crate) fn token(app: &AppHandle) -> Result<String, String> {
    if let Ok(del_entorno) = std::env::var("PERSEO_TOKEN") {
        let limpio = del_entorno.trim().to_string();
        if !limpio.is_empty() {
            return Ok(limpio);
        }
    }

    let ruta = ruta_token(app).ok_or(
        "No se encuentra el token del nucleo. Arranque `python -m perseo_core` una vez, \
         o defina PERSEO_TOKEN.",
    )?;
    std::fs::read_to_string(&ruta)
        .map(|t| t.trim().to_string())
        .map_err(|e| format!("No se pudo leer {}: {e}", ruta.display()))
}

/// Traduce la herramienta que pide el modelo al agente que la hace.
///
/// Esta tabla es la unica parte del cambio que sabe de las dos partes a la vez.
/// El modelo sigue viendo `guardar_recuerdo` y `consultar_base_vectorial`
/// —cambiar los nombres obligaria a reescribir las instrucciones de la sesion—
/// y el nucleo ve trabajos para `memoria` y para `pc`.
fn traducir(herramienta: &str, args: &Value) -> Result<(String, Value), String> {
    match herramienta {
        // `consultar_base_vectorial` es el nombre de la v1 y se acepta todavia
        // por si una sesion vieja se reanuda con el nombre antiguo en su
        // historial. El bueno es `buscar_en_memoria`: no hay ninguna base
        // vectorial detras desde el 2026-08-15, y el modelo se creia el nombre
        // — llego a explicarle al usuario que funcionaba "con un RAG".
        "buscar_en_memoria" | "consultar_base_vectorial" => {
            let texto = args
                .get("texto")
                .or_else(|| args.get("query"))
                .and_then(Value::as_str)
                .unwrap_or_default();
            Ok(("memoria".into(), json!({ "accion": "buscar", "texto": texto })))
        }
        "leer_nota" => {
            let ruta = args.get("ruta").and_then(Value::as_str).unwrap_or_default();
            if ruta.is_empty() {
                return Err("Falta la ruta de la nota. Sale de buscar_en_memoria.".into());
            }
            Ok(("memoria".into(), json!({ "accion": "leer", "ruta": ruta })))
        }
        "guardar_recuerdo" => {
            let entidad = args.get("entidad").and_then(Value::as_str).unwrap_or_default();
            let visual = args
                .get("descripcion_visual")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let contexto = args.get("contexto").and_then(Value::as_str).unwrap_or_default();
            if entidad.is_empty() {
                return Err("Falta la entidad del recuerdo".into());
            }
            if visual.is_empty() && contexto.is_empty() {
                // Antes `descripcion_visual` era obligatoria y el modelo se
                // inventaba una descripcion de camara para guardar un dato sin
                // ninguna imagen delante. Ahora vale cualquiera de los dos,
                // pero algo tiene que quedar escrito.
                return Err("El recuerdo necesita contexto o descripcion_visual".into());
            }
            let mut cuerpo = String::new();
            if !contexto.is_empty() {
                cuerpo.push_str(contexto);
            }
            if !visual.is_empty() {
                if !cuerpo.is_empty() {
                    cuerpo.push_str("\n\n");
                }
                cuerpo.push_str("Descripcion visual: ");
                cuerpo.push_str(visual);
            }
            Ok((
                "memoria".into(),
                json!({ "accion": "anotar", "titulo": entidad, "texto": cuerpo }),
            ))
        }
        "guardar_conversacion" => {
            // El titulo lo pone la fecha, y eso lo sabe el agente: una
            // conversacion no tiene nombre hasta que la lees.
            let mensajes = args.get("mensajes").cloned().unwrap_or_else(|| json!([]));
            Ok((
                "memoria".into(),
                json!({ "accion": "conversacion", "mensajes": mensajes }),
            ))
        }
        "controlar_pc" => {
            let accion = args.get("accion").and_then(Value::as_str).unwrap_or_default();
            let parametro = args
                .get("parametro")
                .and_then(Value::as_str)
                .unwrap_or_default();
            Ok(("pc".into(), json!({ "accion": accion, "parametro": parametro })))
        }
        // N-2: lo que el núcleo ya sabía hacer y la voz no podía pedir.
        "consultar_agenda" => {            // Sin horas es «hoy», que es casi siempre lo que se pregunta.
            let mut peticion = json!({ "accion": "proximos" });
            if let Some(horas) = args.get("horas").and_then(Value::as_f64) {
                if horas > 0.0 {
                    peticion["horas"] = json!(horas);
                }
            }
            Ok(("agenda".into(), peticion))
        }
        "buscar_en_web" => {
            let texto = args
                .get("texto")
                .or_else(|| args.get("consulta"))
                .and_then(Value::as_str)
                .unwrap_or_default();
            if texto.is_empty() {
                return Err("Falta lo que hay que buscar".into());
            }
            Ok(("web".into(), json!({ "accion": "buscar", "texto": texto })))
        }
        "leer_pagina" => {
            let url = args.get("url").and_then(Value::as_str).unwrap_or_default();
            if url.is_empty() {
                return Err("Falta la URL de la página. Sale de buscar_en_web.".into());
            }
            Ok(("web".into(), json!({ "accion": "leer", "url": url })))
        }
        // N-3: los servidores MCP configurados en `<datos>/mcp.json`. La
        // llamada no sabe nada de protocolos: encola un trabajo para el agente
        // `mcp`, que es quien lanza y habla con los procesos.
        // El subagente de código (§4.2): Claude Code por `claude -p`, con su
        // carril propio en el núcleo. Tarda minutos; el modelo queda mejor
        // esperando la respuesta real que con un «ya te contaré», y el tope
        // largo de abajo lo permite sin bloquear a nadie más.
        "encargar_codigo" => {
            let texto = args
                .get("texto")
                .or_else(|| args.get("orden"))
                .and_then(Value::as_str)
                .unwrap_or_default();
            if texto.is_empty() {
                return Err("Falta la descripción de lo que hay que hacer en el código.".into());
            }
            let mut peticion = json!({ "texto": texto });
            if let Some(directorio) = args.get("directorio").and_then(Value::as_str) {
                if !directorio.is_empty() {
                    peticion["directorio"] = json!(directorio);
                }
            }
            Ok(("dev".into(), peticion))
        }
        "listar_mcp" => Ok(("mcp".into(), json!({ "accion": "servidores" }))),
        "usar_mcp" => {
            let servidor = args.get("servidor").and_then(Value::as_str).unwrap_or_default();
            let herramienta = args.get("herramienta").and_then(Value::as_str).unwrap_or_default();
            if servidor.is_empty() || herramienta.is_empty() {
                return Err(
                    "Faltan 'servidor' y 'herramienta'. Con listar_mcp ves cuáles hay.".into(),
                );
            }
            let argumentos = args.get("argumentos").cloned().unwrap_or_else(|| json!({}));
            Ok((
                "mcp".into(),
                json!({
                    "accion": "llamar",
                    "servidor": servidor,
                    "herramienta": herramienta,
                    "argumentos": argumentos,
                }),
            ))
        }
        otra => Err(format!("Herramienta desconocida: {otra}")),
    }
}

/// Convierte el resultado de un trabajo en la frase que lee el modelo.
fn resumir(resultado: &Value) -> String {
    // Una pagina de la web (`leer_pagina`): trae titulo y url junto al texto.
    // Va LO PRIMERO, antes del caso generico del `texto` de abajo, porque
    // volcar una pagina entera al prompt seria tirar el contexto por la
    // ventana.
    if let (Some(titulo), Some(url), Some(texto)) = (
        resultado.get("titulo").and_then(Value::as_str),
        resultado.get("url").and_then(Value::as_str),
        resultado.get("texto").and_then(Value::as_str),
    ) {
        let trozo = recortar(texto, 3500);
        let aviso = if texto.chars().count() > 3500 {
            "\n[…la página sigue; pide más si hace falta]"
        } else {
            ""
        };
        return format!("{titulo} ({url}):\n{trozo}{aviso}");
    }

    if let Some(texto) = resultado.get("texto").and_then(Value::as_str) {
        return texto.to_string();
    }
    if let Some(notas) = resultado.get("notas").and_then(Value::as_array) {
        if notas.is_empty() {
            return "No hay ninguna nota sobre eso en el vault.".into();
        }
        // **La ruta va dentro a proposito.** Sin ella, el modelo veia titulos y
        // extractos y no tenia con que abrir ninguno: en una llamada real
        // encontro tres notas sobre un proyecto y acabo diciendo que no habia
        // encontrado nada especifico, porque no podia leer ni una. La ruta es lo
        // que le pasa a `leer_nota`.
        let lineas: Vec<String> = notas
            .iter()
            .map(|n| {
                format!(
                    "- {} (ruta: {}): {}",
                    n.get("titulo").and_then(Value::as_str).unwrap_or("(sin titulo)"),
                    n.get("ruta").and_then(Value::as_str).unwrap_or("?"),
                    n.get("extracto").and_then(Value::as_str).unwrap_or("")
                )
            })
            .collect();
        return format!(
            "{} nota(s). Para citar lo que pone, usa leer_nota con su ruta:\n{}",
            notas.len(),
            lineas.join("\n")
        );
    }

    // Una nota leida entera. Va antes que el respaldo del JSON en crudo: si no,
    // el modelo recibiria el objeto entero y leeria en voz alta las llaves.
    if let Some(contenido) = resultado.get("contenido").and_then(Value::as_str) {
        return contenido.to_string();
    }

    // Lo que viene de la agenda (`consultar_agenda`): una lista de eventos con
    // su hora. Sin esto, el modelo recibira el JSON en crudo y leera llaves.
    if let Some(eventos) = resultado.get("eventos").and_then(Value::as_array) {
        if eventos.is_empty() {
            return "No hay nada en la agenda para ese plazo.".into();
        }
        let lineas: Vec<String> = eventos
            .iter()
            .map(|e| {
                let titulo = e.get("titulo").and_then(Value::as_str).unwrap_or("(sin titulo)");
                let inicio = e.get("inicio").and_then(Value::as_str).unwrap_or("");
                let cuando: String = inicio.chars().take(16).collect();
                let lugar = e.get("lugar").and_then(Value::as_str).unwrap_or_default();
                if lugar.is_empty() {
                    format!("- {titulo}: {cuando}")
                } else {
                    format!("- {titulo}: {cuando} ({lugar})")
                }
            })
            .collect();
        return lineas.join("\n");
    }

    // Resultados de `buscar_en_web`: titulo, URL y extracto de cada uno.
    if let Some(hallazgos) = resultado.get("resultados").and_then(Value::as_array) {
        if hallazgos.is_empty() {
            return "La busqueda no devolvio nada.".into();
        }
        let lineas: Vec<String> = hallazgos
            .iter()
            .map(|p| {
                let titulo = p.get("titulo").and_then(Value::as_str).unwrap_or("(sin titulo)");
                let url = p.get("url").and_then(Value::as_str).unwrap_or("");
                let extracto = p
                    .get("texto")
                    .and_then(Value::as_str)
                    .map(|t| recortar(t, 200))
                    .unwrap_or_default();
                format!("- {titulo} — {url}\n  {extracto}")
            })
            .collect();
        return format!(
            "{} resultado(s). Para leer uno entero, usa leer_pagina con su URL:\n{}",
            hallazgos.len(),
            lineas.join("\n")
        );
    }

    // Una pagina de la web ya se ha atendido arriba: aqui solo puede llegar un
    // `texto` suelto, que se devuelve tal cual.
    if let Some(texto) = resultado.get("texto").and_then(Value::as_str) {
        return texto.to_string();
    }

    if let Some(ruta) = resultado.get("ruta").and_then(Value::as_str) {
        return format!("Guardado en {ruta}.");
    }

    resultado.to_string()
}

/// Recorta por caracteres y no por bytes, que es como se parte un caracter del
/// espanol por la mitad — y como un limite de UTF-8 acaba en panico.
fn recortar(texto: &str, maximo: usize) -> String {
    if texto.chars().count() <= maximo {
        return texto.to_string();
    }
    let cortado: String = texto.chars().take(maximo).collect();
    format!("{cortado}…")
}

pub(crate) async fn pedir_json(
    peticion: reqwest::RequestBuilder,
) -> Result<Value, String> {
    let respuesta = peticion
        .send()
        .await
        .map_err(|e| format!("No se pudo hablar con el nucleo: {e}. ¿Esta arrancado?"))?;

    let estado = respuesta.status();
    let cuerpo: Value = respuesta
        .json()
        .await
        .map_err(|e| format!("Respuesta ilegible del nucleo: {e}"))?;

    if !estado.is_success() {
        let detalle = cuerpo
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("sin detalle");
        return Err(format!("El nucleo respondio {estado}: {detalle}"));
    }
    Ok(cuerpo)
}

/// Encola un trabajo en el nucleo y espera su resultado.
///
/// El origen es `voz` porque esta cara es la de la llamada: asi se distingue en
/// la cola de la web lo que pediste hablando de lo que escribiste.
#[tauri::command]
pub async fn ejecutar_herramienta(
    app: AppHandle,
    tool_name: String,
    argumentos: String,
) -> Result<String, String> {
    let args: Value =
        serde_json::from_str(&argumentos).map_err(|e| format!("Argumentos JSON invalidos: {e}"))?;

    // La confirmacion no encola nada: resuelve un trabajo que ya esta parado.
    // Va antes de `traducir` porque no es un trabajo nuevo para un agente, es
    // una decision sobre uno que existe. Ver N-1 en bitacora/11_HISTORIA.md §12:
    // desde que Telegram dejo de tener botones, el si se da aqui, de viva voz.
    if tool_name == "responder_confirmacion" {
        return responder_confirmacion(&app, &args).await;
    }
    // Lo mismo para la de N-2 que habla con rutas de la API y no con la cola:
    // lee lo que hay delante ahora mismo. Abrir proyectos se quedó fuera de la
    // llamada (2026-08-23): para eso están el panel y el servidor MCP
    // 'subagentes', que además trabaja en ellos, no solo los abre.
    if tool_name == "situacion_actual" {
        return situacion_actual(&app).await;
    }

    let (agente, peticion) = traducir(&tool_name, &args)?;

    let token = token(&app)?;
    let cliente = reqwest::Client::new();
    let base = base_url();

    let creado = pedir_json(
        cliente
            .post(format!("{base}/trabajos"))
            .bearer_auth(&token)
            .json(&json!({ "agente": agente, "peticion": peticion, "origen": "voz" })),
    )
    .await?;

    let id = creado
        .get("id")
        .and_then(Value::as_i64)
        .ok_or("El nucleo no devolvio el identificador del trabajo")?;

    let tope = if agente == "dev" { ESPERA_DEV } else { ESPERA_MAXIMA };
    esperar_trabajo(&cliente, &token, &base, id, tope).await
}

/// Resuelve por voz un trabajo parado esperando un si.
///
/// El flujo entero es hablado: una herramienta devolvio «pendiente de que lo
/// confirmes», Perseo pregunta en voz alta, el señor Persus contesta, y el
/// modelo llama aqui con su decision. Nadie pulsa nada durante la llamada; los
/// botones del panel y de la pantalla siguen para cuando no hay voz delante.
///
/// Tras aprobar se espera al resultado con el mismo plazo que cualquier otra
/// herramienta: lo normal es que el señor Persus pregunte «¿y?» justo despues,
/// y asi hay respuesta de verdad y no un «esta en ello» para todo.
async fn responder_confirmacion(app: &AppHandle, args: &Value) -> Result<String, String> {
    let id = args
        .get("id")
        .and_then(Value::as_i64)
        .ok_or("Falta el numero del trabajo a confirmar")?;
    let decision = args
        .get("decision")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if !["aprobar", "rechazar"].contains(&decision) {
        return Err(format!("Decision desconocida: {decision}"));
    }

    let token = token(app)?;
    let cliente = reqwest::Client::new();
    let base = base_url();

    pedir_json(
        cliente
            .post(format!("{base}/trabajos/{id}/{decision}"))
            .bearer_auth(&token)
            .json(&json!({})),
    )
    .await?;

    if decision == "rechazar" {
        // Rechazar no tiene mas recorrido: el trabajo se cierra sin ejecutarse,
        // y no hay nada que esperar.
        return Ok(format!(
            "Hecho: el trabajo #{id} queda rechazado y no se ejecuta."
        ));
    }

    esperar_trabajo(&cliente, &token, &base, id, ESPERA_MAXIMA).await
}

/// Responde a «¿qué hay ahora mismo?» con lo que ya sabe el núcleo.
///
/// Rehecho el 2026-08-23: antes decía "trabaja en un asunto de 'correo'", que
/// no dice nada. El briefing ahora trae lo que de verdad se pregunta en voz:
/// **qué** está haciendo Perseo (agente y encargo), **qué espera un sí** —con
/// la pregunta literal, para que la decisión sea contestarla y no ir a
/// buscarla—, **qué falló por última vez**, el buzón por cajones y la batería.
///
/// Lee `/estado`, como el panel, y `/trabajos`, que es de donde salen las
/// preguntas pendientes.
async fn situacion_actual(app: &AppHandle) -> Result<String, String> {
    let token = token(app)?;
    let cliente = reqwest::Client::new();
    let base = base_url();
    let estado = pedir_json(
        cliente
            .get(format!("{base}/estado"))
            .bearer_auth(&token),
    )
    .await?;
    let trabajos = pedir_json(
        cliente
            .get(format!("{base}/trabajos?limite=15"))
            .bearer_auth(&token),
    )
    .await?;

    let presencia = estado.get("presencia").cloned().unwrap_or_else(|| json!({}));
    let mut partes: Vec<String> = Vec::new();

    // Qué está haciendo AHORA, con el encargo delante y no solo el nombre del
    // agente: "un asunto de 'dev'" no dice nada; "el código de mi web", sí.
    let lista_vacia: Vec<Value> = Vec::new();
    let filas = trabajos.get("trabajos").and_then(Value::as_array).unwrap_or(&lista_vacia);
    if let Some(en_curso) = filas.iter().find(|t| t.get("estado").and_then(Value::as_str) == Some("en_curso")) {
        let agente = en_curso.get("agente").and_then(Value::as_str).unwrap_or("?");
        let peticion = en_curso
            .get("peticion")
            .and_then(|p| p.get("texto").or_else(|| p.get("consulta")).or_else(|| p.get("titulo")))
            .and_then(Value::as_str)
            .unwrap_or("");
        if peticion.is_empty() {
            partes.push(format!("ahora mismo trabaja en un asunto de '{agente}'"));
        } else {
            partes.push(format!(
                "ahora mismo trabaja en '{}' ({agente})",
                recortar(peticion, 80)
            ));
        }
    }

    // Lo que espera un sí, con su pregunta. Es lo único parado esperando al
    // señor Persus: si hay algo, esto va primero aunque lo demás callara.
    let esperando: Vec<String> = filas
        .iter()
        .filter(|t| t.get("estado").and_then(Value::as_str) == Some("esperando"))
        .filter_map(|t| {
            let id = t.get("id").and_then(Value::as_i64)?;
            let pregunta = t
                .get("confirmacion")
                .and_then(|c| c.get("resumen"))
                .and_then(Value::as_str)
                .unwrap_or("una confirmación");
            Some(format!("#{id} {pregunta}"))
        })
        .collect();
    match esperando.len() {
        0 => {}
        1 => partes.insert(0, format!("espera tu sí sobre: {}", esperando[0])),
        _ => partes.insert(
            0,
            format!(
                "esperan tu sí {} asuntos: {}",
                esperando.len(),
                esperando.join("; ")
            ),
        ),
    }

    // El último fallo, con su primera línea. Saber QUE falló algo cambia la
    // conversación; saber POR QUÉ suele ahorrar la pregunta de seguimiento.
    if let Some(fallido) = filas.iter().find(|t| t.get("estado").and_then(Value::as_str) == Some("fallido")) {
        let agente = fallido.get("agente").and_then(Value::as_str).unwrap_or("?");
        let error = fallido
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("sin detalle")
            .lines()
            .next()
            .unwrap_or("sin detalle");
        partes.push(format!("falló por última vez un asunto de '{agente}': {}", recortar(error, 100)));
    }

    match presencia.get("correo").and_then(Value::as_object) {
        Some(cajones) if !cajones.is_empty() => {
            // Los nombres de los cajones son los del triaje; al señor Persus
            // se le dicen como se le dicen en la web.
            for (clase, cuantos) in cajones {
                let como = match clase.as_str() {
                    "requiere_accion" => "piden acción",
                    "interesante" => "interesante(s)",
                    "no_seguro" => "sin decidir",
                    otra => otra,
                };
                partes.push(format!("el buzón tiene {} correo(s) que {como}", cuantos));
            }
        }
        _ => partes.push("el buzón está al día".into()),
    }

    // La batería es la pregunta de máquina más frecuente y aquí vive su
    // respuesta corta; el resto de telemetría queda para el panel.
    let bateria = estado
        .get("maquina")
        .and_then(|m| m.get("bateria"))
        .cloned()
        .unwrap_or_else(|| json!({}));
    if let Some(pct) = bateria.get("porcentaje").and_then(Value::as_i64) {
        let enchufado = bateria
            .get("enchufado")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        partes.push(format!(
            "la batería va al {pct}% {}",
            if enchufado { "(enchufada)" } else { "sin enchufar" }
        ));
    }

    if partes.is_empty() {
        return Ok("Todo tranquilo: nada en marcha y el buzón al día.".into());
    }
    Ok(format!("{}.", partes.join("; ")))
}

/// Sonda un trabajo hasta que termina o se agota el plazo.
///
/// Extraido de `ejecutar_herramienta` para compartirlo con
/// `responder_confirmacion`: en los dos casos el modelo queda mejor con el
/// resultado real que con un «sigue en marcha», pero el techo es el mismo.
async fn esperar_trabajo(
    cliente: &reqwest::Client,
    token: &str,
    base: &str,
    id: i64,
    tope: Duration,
) -> Result<String, String> {
    let limite = Instant::now() + tope;
    loop {
        let trabajo = pedir_json(
            cliente
                .get(format!("{base}/trabajos/{id}"))
                .bearer_auth(token),
        )
        .await?;

        match trabajo.get("estado").and_then(Value::as_str).unwrap_or("") {
            "hecho" => {
                let vacio = json!({});
                return Ok(resumir(trabajo.get("resultado").unwrap_or(&vacio)));
            }
            "fallido" => {
                return Err(trabajo
                    .get("error")
                    .and_then(Value::as_str)
                    .unwrap_or("El trabajo fallo")
                    .to_string())
            }
            "cancelado" | "rechazado" => return Err("El trabajo se cerro sin ejecutarse".into()),
            "esperando" => {
                // El agente ha parado a pedir un si — otra vez, o la primera.
                // No se espera aqui: el modelo pregunta en voz alta, y quien
                // quiera resolverlo sin hablar tiene el panel.
                let pregunta = trabajo
                    .get("confirmacion")
                    .and_then(|c| c.get("resumen"))
                    .and_then(Value::as_str)
                    .unwrap_or("una confirmacion");
                return Ok(format!(
                    "Queda pendiente de que lo confirmes: {pregunta} (trabajo #{id})."
                ));
            }
            _ => {}
        }

        if Instant::now() >= limite {
            return Ok(format!(
                "Sigue en marcha; te lo cuento cuando termine (trabajo #{id})."
            ));
        }
        sleep(SONDEO).await;
    }
}

/// Comprueba que el nucleo esta vivo **y que el token vale** antes de la primera
/// herramienta.
///
/// Antes esto arrancaba el proceso de Python y pagaba por adelantado los 3,7 s
/// de importar llama_index y chromadb (H-12). Ahora el nucleo ya esta
/// encendido —es su razon de ser— y esto solo sirve para avisar pronto si algo
/// no esta, en vez de descubrirlo a mitad de una frase.
///
/// Se pregunta por `/trabajos` y no por `/salud`: **`/salud` es publica**, asi
/// que contestaba 200 con un token caducado o equivocado y la comprobacion
/// dejaba pasar justo el fallo que existe para detectar. `?limite=1` la hace
/// tan barata como la otra.
#[tauri::command]
pub async fn precalentar_herramientas(app: AppHandle) -> Result<(), String> {
    let token = token(&app)?;
    let cliente = reqwest::Client::new();
    pedir_json(
        cliente
            .get(format!("{}/trabajos?limite=1", base_url()))
            .bearer_auth(&token),
    )
    .await?;
    Ok(())
}

// --------------------------------------------------------------------------- //
// Biometría
//
// El reconocimiento de quién habla y quién sale por la cámara vive en el núcleo
// (perseo_core/biometria.py); la llamada solo transporta los mismos trozos que
// ya le manda a Gemini y pinta la etiqueta que vuelve. Estos comandos son rutas
// CONCRETAS y no un proxy genérico a /{ruta}, por el mismo motivo que panel.rs:
// si el frontend elige la ruta entera, la ruta la escribe el frontend.
//
// No pasan por la cola de trabajos a propósito: identificar voz son varias
// peticiones por segundo durante toda la llamada, y ensuciar la cola que ve el
// móvil con un trabajo cada dos segundos la haría ilegible.
// --------------------------------------------------------------------------- //

async fn traer_biometria(app: &AppHandle, ruta: &str) -> Result<Value, String> {
    let token = token(app)?;
    let cliente = reqwest::Client::new();
    pedir_json(
        cliente
            .get(format!("{}{ruta}", base_url()))
            .bearer_auth(&token),
    )
    .await
}

async fn mandar_biometria(
    app: &AppHandle,
    metodo: &str,
    ruta: &str,
    cuerpo: Value,
) -> Result<Value, String> {
    let token = token(app)?;
    let cliente = reqwest::Client::new();
    let peticion = match metodo {
        "POST" => cliente.post(format!("{}{ruta}", base_url())).json(&cuerpo),
        "DELETE" => cliente.delete(format!("{}{ruta}", base_url())),
        otra => return Err(format!("Método interno desconocido: {otra}")),
    };
    pedir_json(peticion.bearer_auth(&token)).await
}

/// Perfiles guardados, progreso de aprendizaje y qué motores hay hoy.
#[tauri::command]
pub async fn biometria_estado(app: AppHandle) -> Result<Value, String> {
    traer_biometria(&app, "/biometria").await
}

/// Un trozo de PCM 16k mono (base64): ¿de quién es la voz?
#[tauri::command]
pub async fn biometria_voz(app: AppHandle, audio: String) -> Result<Value, String> {
    mandar_biometria(&app, "POST", "/biometria/voz", json!({ "audio": audio })).await
}

/// Un JPEG (base64): qué caras salen, con nombre y caja.
#[tauri::command]
pub async fn biometria_cara(app: AppHandle, imagen: String) -> Result<Value, String> {
    mandar_biometria(&app, "POST", "/biometria/cara", json!({ "imagen": imagen })).await
}

/// Crea o refuerza un perfil con una muestra grabada a propósito.
#[tauri::command]
pub async fn biometria_enrolar(
    app: AppHandle,
    nombre: String,
    audio: Option<String>,
    imagen: Option<String>,
) -> Result<Value, String> {
    mandar_biometria(
        &app,
        "POST",
        "/biometria/perfiles",
        json!({ "nombre": nombre, "audio": audio, "imagen": imagen }),
    )
    .await
}

/// Le pone el nombre real a un «Desconocido N».
#[tauri::command]
pub async fn biometria_renombrar(
    app: AppHandle,
    nombre: String,
    nuevo_nombre: String,
) -> Result<Value, String> {
    mandar_biometria(
        &app,
        "POST",
        &format!("/biometria/perfiles/{}", encode_ruta(&nombre)),
        json!({ "nuevo_nombre": nuevo_nombre }),
    )
    .await
}

/// Borra el perfil y sus vectores. Sin copia, que es lo pedido.
#[tauri::command]
pub async fn biometria_borrar(app: AppHandle, nombre: String) -> Result<Value, String> {
    mandar_biometria(
        &app,
        "DELETE",
        &format!("/biometria/perfiles/{}", encode_ruta(&nombre)),
        json!({}),
    )
    .await
}

/// Codifica un nombre para meterlo en la ruta. Los perfiles pueden llamarse
/// «Desconocido 1» o «Fátima»: espacios y acentos no pueden ir crudos en una URL.
/// Lo codifica el propio parser de URL que trae reqwest, sin dependencia nueva.
///
/// Devuelve SOLO el segmento, no la ruta entera: quien llama ya escribe
/// `/biometria/perfiles/{}`. Cuando esto devolvía la ruta completa salía
/// `/biometria/perfiles//biometria/perfiles/Desconocido%201`, y el núcleo
/// contestaba 404 a todo borrado y a todo renombrado (H-69).
fn encode_ruta(nombre: &str) -> String {
    let mut url = reqwest::Url::parse("http://localhost/")
        .expect("la URL semilla es fija y válida");
    url.path_segments_mut()
        .expect("la URL semilla no puede ser «cannot-be-a-base»")
        .push(nombre);
    url.path().trim_start_matches('/').to_string()
}

#[cfg(test)]
mod pruebas_ruta {
    use super::encode_ruta;

    #[test]
    fn codifica_un_solo_segmento_sin_prefijo() {
        assert_eq!(encode_ruta("Desconocido 1"), "Desconocido%201");
        assert_eq!(encode_ruta("Fátima"), "F%C3%A1tima");
        assert_eq!(
            format!("/biometria/perfiles/{}", encode_ruta("Desconocido 1")),
            "/biometria/perfiles/Desconocido%201"
        );
    }
}
