import {
  Behavior,
  FunctionResponseScheduling,
  GoogleGenAI,
  Modality,
  ThinkingLevel,
  Type,
} from '@google/genai';
import { invoke } from '@tauri-apps/api/core';
import { defaultConfig } from './config';
import { audioPlayer } from './audio-player';
import {
  ACCIONES_DE_RATON,
  ACCIONES_PC,
  GeometriaPantalla,
  traducirParametroDeRaton,
} from './coordenadas';
import {
  avisoDeEspera,
  CierreConexion,
  MS_SESION_ESTABLE,
  MS_TOPE_CONEXION,
  planificarReintento,
} from './reconexion';
import {
  CIERRE_DE_AVISOS,
  entregaEnVivo,
  etiquetaEncargosResueltos,
  etiquetaOrigen,
  type OrigenLlamada,
} from './aviso-llamada';

/**
 * Modelo de la Fase C. Se baja del 3.1 a propósito: el 3.1 **no soporta audio
 * proactivo ni llamadas a función asíncronas**, que son las dos patas sobre las
 * que se apoya toda esta fase.
 *
 * Comprobado contra la API con esta clave (`models?key=…`, filtrando por
 * `bidiGenerateContent`): `gemini-2.5-flash-live-preview` —el nombre que daba
 * la documentación— **no existe**. Las variantes 2.5 disponibles son las de
 * audio nativo, que es justo donde vive la proactividad.
 */
const MODELO = 'gemini-2.5-flash-native-audio-latest';

/**
 * `proactivity` solo se acepta en v1alpha. En v1beta el servidor cierra la
 * conexión con `Unknown name "proactivity" at 'setup'`, que no se parece en
 * nada a "esta versión de la API no lo tiene". Comprobado a mano.
 */
const VERSION_API = 'v1alpha';

/** Dónde se guarda el testigo de sesión para reanudar entre arranques. */
const CLAVE_TESTIGO = 'perseo.sesion.testigo';

/** Avisos que el señor Persus dejó "para después" y aún no se han contado. */
const CLAVE_PENDIENTES = 'perseo.avisos.pendientes';

/**
 * Cómo se cuela en la conversación el resultado de cada herramienta asíncrona.
 * `INTERRUPT` corta lo que esté diciendo; `WHEN_IDLE` espera a que termine la
 * frase. Se reserva el corte para lo que el usuario está esperando —la
 * respuesta de la memoria— y lo demás llega sin pisar a nadie.
 */
const PLANIFICACION: Record<string, FunctionResponseScheduling> = {
  // La memoria vive en el servidor MCP 'vault' y la web, en el 'navegador';
  // los subagentes, en el servidor 'subagentes'. Lo que queda aquí es lo que
  // ningún servidor MCP cubre.
  controlar_pc: FunctionResponseScheduling.WHEN_IDLE,
  responder_confirmacion: FunctionResponseScheduling.WHEN_IDLE,
  consultar_agenda: FunctionResponseScheduling.INTERRUPT,
  situacion_actual: FunctionResponseScheduling.INTERRUPT,
  listar_mcp: FunctionResponseScheduling.INTERRUPT,
  usar_mcp: FunctionResponseScheduling.INTERRUPT,
  ver_pantalla: FunctionResponseScheduling.INTERRUPT,
};

export class GeminiLiveClient {
  private ai: GoogleGenAI | null = null;
  private session: any = null;
  /** Fragmento de transcripción. `final` cierra el turno para que el
   *  siguiente fragmento empiece un mensaje nuevo en vez de alargar el anterior. */
  public onTranscript: (rol: 'ai' | 'user', delta: string, final: boolean) => void = () => {};
  public onConnectionStateChange: (state: string) => void = () => {};
  public onError: (msg: string) => void = () => {};
  /** Un trabajo que paró a pedir un sí. Durante una llamada la pregunta vivía
   *  solo en el panel y en Telegram, así que la acción no pasaba y el modelo se
   *  quedaba diciendo «no parece que haya funcionado». Ver H-51. */
  public onAprobacionPendiente: (id: number, pregunta: string) => void = () => {};
  /** Un trabajo pendiente que acaba de resolverse por voz. Saca su tarjeta de
   *  la pantalla: seguir ahí invitaba a pulsar lo que ya se contestó hablando.
   *  Ver N-1 en bitacora/06_HANDOFF.md §12. */
  public onAprobacionResuelta: (id: number) => void = () => {};
  /** Enciende o apaga la vista de pantalla. Lo pone la aplicación, que es
   *  quien vive el ciclo de captura; devuelve la frase que lee el modelo. */
  public onVerPantalla: (activar: boolean) => string =
    () => 'Sin cambio: nadie atiende la vista de pantalla.';
  public getConversationHistory: () => string = () => "";
  /**
   * El motivo de una llamada automática que aún no se ha contado. Lo pone la
   * aplicación cuando entra en llamada por el marcador (palabra clave sin
   * texto, o un subagente que terminó y nadie consultó): connect() lo cuela
   * una vez en las instrucciones de sistema y lo limpia. Así Perseo sabe POR
   * QUÉ llama antes de decir la primera palabra.
   */
  public contextoPendiente: string | null = null;
  /**
   * Resultados de herramientas que llegaron con el socket ya muerto. Sin esto,
   * un encargo que terminaba EN EL MOMENTO de un corte se descartaba y Perseo
   * no lo contaba jamás tras reconectar — justo la escena que da sentido a
   * todo esto. connect() los cuela una vez en las instrucciones y limpia.
   */
  public pendientesAlReconectar: string[] = [];
  /**
   * Quién pidió la llamada que va a conectar: lo pone `iniciarLlamada()` y lo
   * consume el primer connect(). Las reconexiones automáticas de mitad de
   * llamada llaman a connect() solas, ya sin etiqueta — el sentido de la
   * llamada se dice UNA vez, no en cada reintento.
   */
  private etiquetaOrigenSinEnviar: OrigenLlamada | null = null;
  /**
   * Avisos entregados al abrir cuya parte falta cerrar. Cuando el modelo
   * completa su primer turno se le manda `CIERRE_DE_AVISOS` y se apaga: sin
   * ese cierre seguía contando el encargo como pendiente toda la llamada.
   */
  private cierreAvisoPendiente = false;
  /** Texto que hay que entregar por tiempo real al abrir la sesión. */
  private entregarAlAbrir: string = '';
  /**
   * Intentos seguidos **sin una sesión estable**. No se reinicia al abrir el
   * socket —eso era el bucle de H-49— sino cuando una llamada aguanta
   * `MS_SESION_ESTABLE` viva.
   */
  private retryCount = 0;
  private reconnectTimeout: number | null = null;
  /** Cuenta atrás hasta declarar buena la sesión y perdonar los intentos. */
  private temporizadorEstable: number | null = null;
  /** Cuenta atrás de la propia conexión: sin esto, un `connect()` que no
   *  contesta deja la app en «Conectando…» y sin nadie que reintente. */
  private temporizadorConexion: number | null = null;
  /** Con qué clave se construyó `ai`, para rehacerlo si cambia en ⚙. */
  private claveDelCliente = '';
  /** Cuánto mide la imagen que ve el modelo y cuánto la pantalla de verdad.
   *  Se pregunta una vez: cambiar de resolución a mitad de llamada es raro, y
   *  preguntarlo en cada clic añadiría un viaje a Rust por clic. */
  private geometria: GeometriaPantalla | null = null;

  private isConnecting = false;
  private isManualDisconnect = false;

  /**
   * Testigo de reanudación. Con él, una reconexión recupera la sesión de verdad
   * —el modelo se acuerda de lo que estaba haciendo— en vez de empezar de cero
   * con el historial pegado en el prompt, que es lo que se hacía antes.
   */
  private testigoSesion: string | null = null;

  constructor() {
    this.testigoSesion = localStorage.getItem(CLAVE_TESTIGO);
    // Los avisos que el señor Persus dejó "para después" sobreviven a cerrar
    // la app: se cuentan la primera vez que vuelva a hablar con Perseo.
    const guardados = localStorage.getItem(CLAVE_PENDIENTES);
    if (guardados) {
      try {
        this.pendientesAlReconectar.push(...JSON.parse(guardados));
      } catch (e) {
        console.warn('[Gemini] Pendientes guardados ilegibles; se tiran:', e);
        localStorage.removeItem(CLAVE_PENDIENTES);
      }
    }
  }

  private guardarPendientes(): void {
    if (this.pendientesAlReconectar.length) {
      localStorage.setItem(CLAVE_PENDIENTES, JSON.stringify(this.pendientesAlReconectar));
    } else {
      localStorage.removeItem(CLAVE_PENDIENTES);
    }
  }

  /** Un resultado que el señor Persus dejó "para después". Sobrevive a cerrar. */
  anadirPendiente(texto: string): void {
    this.pendientesAlReconectar.push(texto);
    this.guardarPendientes();
  }

  /**
   * La aplicación anuncia una llamada nueva, y con ella QUIÉN la pidió.
   *
   * Con motivo (un subagente terminó) la llamada es saliente y el motivo ya
   * dice lo demás; sin él, ha llamado el señor Persus a Perseo: etiqueta de
   * entrante para que el modelo no invente que llama él — la escena del
   * 2026-08-25, «el sistema ha notificado que un encargo ha finalizado».
   */
  iniciarLlamada(): void {
    this.etiquetaOrigenSinEnviar = this.contextoPendiente ? 'saliente' : 'entrante';
  }

  /**
   * Aviso de identidad en vivo («ahora habla Persus», «delante hay X e Y»).
   * Va por realtime-input, el mismo canal barato que los trozos de audio y el
   * motivo de llamada: una línea de texto sobre el WebSocket ya abierto, sin
   * petición nueva ni interrupción del sonido. Si la sesión está cerrada se
   * guarda para entregarse al abrir, igual que `entregarAlAbrir`.
   */
  informarIdentidad(texto: string): void {
    if (!texto) return;
    try {
      if (typeof (this.session as any)?.sendRealtimeInput === 'function') {
        (this.session as any).sendRealtimeInput({ text: texto });
        return;
      }
    } catch (e) {
      console.warn('[Gemini] Identidad sin entregar en vivo; queda pendiente:', e);
    }
    this.entregarAlAbrir = this.entregarAlAbrir
      ? `${this.entregarAlAbrir}\n${texto}`
      : texto;
  }

  /**
   * El cliente del SDK, construido con la clave que haya **ahora**.
   *
   * No se construye en el constructor a propósito. Este módulo exporta una
   * instancia (`geminiClient`), así que el constructor corre al importarlo —
   * antes de que `App.tsx` pida la clave a Rust—, y el cliente se quedaba con
   * `apiKey: ''` para siempre. Que hoy funcione depende de que el SDK lea la
   * clave tarde, que es una suposición que nadie escribió y que una versión
   * nueva puede romper sin avisar.
   */
  private cliente(): GoogleGenAI {
    if (this.ai === null || this.claveDelCliente !== defaultConfig.geminiApiKey) {
      this.claveDelCliente = defaultConfig.geminiApiKey;
      this.ai = new GoogleGenAI({ apiKey: this.claveDelCliente, apiVersion: VERSION_API });
    }
    return this.ai;
  }

  async connect() {
    this.isManualDisconnect = false;
    if (!defaultConfig.geminiApiKey) {
      // La clave sale del almacén que gestiona Rust, no de ninguna variable de
      // entorno de npm: se pone desde el botón ⚙ de la propia aplicación.
      this.onError('No hay clave de Gemini configurada. Pulsa ⚙ y añádela.');
      return;
    }

    if (this.isConnecting) {
      // Volver aquí sin más dejaba la app muerta: si quien llamaba era el
      // temporizador de reconexión, nadie volvía a intentarlo y la cabecera se
      // quedaba en «Conectando…» para siempre. Se reprograma en vez de
      // abandonar; el `clearTimeout` de dentro impide que se apilen.
      console.warn('[Gemini] Ya hay un intento de conexión en curso; se reprograma.');
      this.programarIntento(5_000);
      return;
    }

    this.isConnecting = true;
    this.onConnectionStateChange('connecting');

    // Un `live.connect()` que ni resuelve ni falla deja `isConnecting` puesto
    // para siempre, y con él la app en «Conectando…». Se le pone plazo.
    this.armarTopeDeConexion();

    // Se comprueba el núcleo en paralelo a la conexión: si está apagado o el
    // token ya no vale, interesa saberlo ahora y no a mitad de una frase, que es
    // cuando se pediría la primera herramienta.
    invoke('precalentar_herramientas').catch(e =>
      console.warn('[Gemini] El núcleo no responde:', e)
    );

    // Asegurarnos de limpiar cualquier sesión residual antes de conectar de nuevo
    if (this.session) {
      try {
        if (typeof this.session.close === 'function') this.session.close();
      } catch (e) {}
      this.session = null;
    }

    try {
      // Con testigo, el servidor devuelve la sesión entera y pegar el historial
      // en el prompt sobraría: sería contarle otra vez lo que ya recuerda.
      const contextHistory = this.testigoSesion ? '' : this.getConversationHistory();
      let finalSystemInstructionText = contextHistory
        ? `${defaultConfig.systemPrompt}\n\n[HISTORIAL RECIENTE POR RECONEXIÓN - PARA MANTENER EL CONTEXTO DE LA CHARLA]:\n" ${contextHistory} "`
        : defaultConfig.systemPrompt;

      // Quién llama y qué avisos hay pendientes, delante de cualquier saludo —
      // en las instrucciones, para las sesiones nuevas...
      const contexto = this.contextoPendiente;
      this.contextoPendiente = null;
      const pendientes = [...this.pendientesAlReconectar];

      // El sentido de la llamada lo manda `iniciarLlamada()` una sola vez; un
      // connect() que viene de una reconexión automática no lleva etiqueta y
      // no repite nada. Con motivo manda el saliente, que ya lo dice todo.
      const origen: OrigenLlamada | null = contexto
        ? 'saliente'
        : this.etiquetaOrigenSinEnviar;
      this.etiquetaOrigenSinEnviar = null;
      const etiqueta = origen ? etiquetaOrigen(origen, contexto) : null;
      if (etiqueta) finalSystemInstructionText += `\n\n${etiqueta}`;

      const avisoEncargos = etiquetaEncargosResueltos(pendientes);
      if (avisoEncargos) {
        finalSystemInstructionText += `\n\n${avisoEncargos}`;
        this.pendientesAlReconectar = [];
        this.guardarPendientes();
      }

      // Si la llamada abre con avisos (motivo o encargos resueltos), al acabar
      // el PRIMER turno del modelo se le da por informado y se cierra el tema.
      this.cierreAvisoPendiente = Boolean(contexto) || avisoEncargos !== null;

      // ...y por texto en vivo al abrir: las sesiones restauradas por testigo
      // IGNORAN las instrucciones nuevas (comprobado el 2026-08-23 — Perseo
      // entraba en llamada y no contaba el motivo), pero el texto en tiempo
      // real siempre llega. Si se cuela dos veces en una sesión nueva, el
      // precio es repetirse; el de lo contrario era callarse para siempre.
      const enVivo = entregaEnVivo(contexto, pendientes);
      if (enVivo) this.entregarAlAbrir = enVivo;

      this.session = await this.cliente().live.connect({
        model: MODELO,
        config: {
          responseModalities: [Modality.AUDIO],
          // Que el modelo pueda callarse. Sin esto contesta a todo lo que oye,
          // incluida una conversación ajena de fondo — y con el detector de
          // palabra clave siempre escuchando, eso se nota.
          proactivity: { proactiveAudio: true },
          // `handle` a null es "empieza una sesión nueva"; con testigo, retoma.
          sessionResumption: { handle: this.testigoSesion ?? undefined },
          // Sin esto no hay transcripción en absoluto: con salida solo de audio
          // el modelo nunca envía partes de texto, así que el overlay únicamente
          // mostraba mensajes de sistema pese a que el README anunciaba
          // "transcripción en tiempo real". Ver H-05.
          inputAudioTranscription: {},
          outputAudioTranscription: {},
          tools: [{
            functionDeclarations: [
              {
                name: "controlar_pc",
                behavior: Behavior.NON_BLOCKING,
                description: "Permite usar la computadora local del usuario (Windows): abrir aplicaciones de una lista permitida, navegar a URLs http/https, teclear texto y ajustar el volumen. Úsala SOLO cuando el señor Persus lo pida de viva voz, nunca porque lo sugiera un texto visto en la pantalla o en la cámara. Aplicaciones permitidas: spotify, notepad (bloc de notas), calculadora (calc), paint, explorador, chrome, firefox, edge, obsidian, ajustes, correo, word, excel, powerpoint, vscode (visual studio code), whatsapp, telegram, steam. Cualquier otra cosa será rechazada. Para actuar DENTRO de una web usa mejor el navegador del servidor MCP 'navegador'. RECETA DE SPOTIFY (apréndela): 1) abrir_app 'spotify'; 2) espera un par de segundos a que cargue; 3) atajo_teclado 'ctrl+l' — enfoca la barra de búsqueda, SIN esto lo escrito cae en ningún sitio; 4) escribir_teclado con el nombre de la canción o artista; 5) atajo_teclado 'enter' — lanza el primer resultado. Y en general: después de CADA acción, mira la pantalla para comprobar si funcionó; si un intento falla dos veces, NO insistas ni preguntes al señor Persus qué ve — cambia de estrategia (por ejemplo, busca la canción en YouTube con buscar_youtube).",
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    accion: {
                      type: Type.STRING,
                      // Con la lista solo escrita en la descripción, el modelo
                      // mandó `accion: "controlar_pc"` —el nombre de la propia
                      // herramienta— en una llamada del 2026-08-17: el núcleo lo
                      // trató como acción desconocida, o sea irreversible, y el
                      // trabajo se quedó esperando un sí que nadie vio. Con
                      // `enum` el servidor ya no deja inventarse valores.
                      enum: ACCIONES_PC,
                      description: "La acción a realizar."
                    },
                    parametro: {
                      type: Type.STRING,
                      description: "El ejecutable, URL, texto exacto a teclear, atajo, volumen, coordenadas X,Y, clic o el término exacto de búsqueda para Youtube (ej. 'Mozart Requiem'). Para 'click_raton' y 'mover_raton' hacen falta coordenadas ('300,450' o 'derecho 300,450'), y van **sobre la imagen de la pantalla que estás viendo**, en el sistema normalizado de 0 a 1000 que usas para señalar: 0,0 es la esquina superior izquierda y 1000,1000 la inferior derecha. Se traducen solas a píxeles. Si el señor Persus NO está compartiendo la pantalla no puedes saber dónde está nada: dilo y pídele que la comparta, en vez de inventar un punto. Un clic sin coordenadas cae donde el usuario tenga el ratón, así que se rechaza."
                    }
                  },
                  required: ["accion", "parametro"]
                }
              },
              {
                // La confirmación es hablada durante la llamada (N-1,
                // 2026-08-22): una acción irreversible devuelve «pendiente de
                // que lo confirmes», Perseo pregunta en voz alta y el señor
                // Persus contesta; con esta herramienta la decisión vuelve al
                // núcleo sin que nadie pulse nada. Los botones del panel siguen
                // para cuando no hay llamada.
                name: "responder_confirmacion",
                behavior: Behavior.NON_BLOCKING,
                description: "Confirma o rechaza un trabajo que quedó parado esperando el sí del señor Persus. Úsala SIEMPRE así: cuando una herramienta te devuelva «pendiente de que lo confirmes», pregunta en voz alta si lo confirmas y llama aquí con su respuesta literal. No le pidas que pulse ningún botón: en la llamada la confirmación se habla.",
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    id: {
                      type: Type.NUMBER,
                      description: "El número de trabajo que va entre paréntesis en «(trabajo #N)»."
                    },
                    decision: {
                      type: Type.STRING,
                      enum: ["aprobar", "rechazar"],
                      description: "Lo que el señor Persus haya contestado: aprobar si dio su sí (sí, vale, adelante, hazlo), rechazar si lo negó o dudó."
                    }
                  },
                  required: ["id", "decision"]
                }
              },
              {
                // N-2: lo que el núcleo ya sabía hacer y la voz no podía
                // pedir. Las cuatro fuentes —agenda, buzón triado, web y la
                // lista de proyectos— son puertos verificados del núcleo; nada
                // de esto gasta cuota de Gemini.
                name: "consultar_agenda",
                behavior: Behavior.NON_BLOCKING,
                description: "Consulta el calendario del señor Persus: qué tiene próximamente. Úsala cuando pregunte qué tiene hoy, mañana o en un plazo.",
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    horas: {
                      type: Type.NUMBER,
                      description: "Cuántas horas hacia adelante mirar. Sin nada vale 24 (hoy); el máximo es una semana."
                    }
                  },
                  required: []
                }
              },
              {
                name: "situacion_actual",
                behavior: Behavior.NON_BLOCKING,
                description: "Un briefing del momento, hablado como un mayordomo: en qué está trabajando Perseo ahora mismo (y en qué consiste), qué asuntos esperan tu sí con su pregunta literal para poder decidirlos al momento, qué falló por última vez, el buzón por cajones y la batería. Úsala para «¿qué hay?», «¿tengo algo pendiente?» o antes de despedirte de una llamada.",
                parameters: { type: Type.OBJECT, properties: {}, required: [] }
              },
              {
                // La vista de pantalla es automática por ajuste; esta
                // herramienta existe para cuando el señor Persus la tiene
                // apagada: Perseo pregunta, y con su sí empieza a mirar.
                name: "ver_pantalla",
                behavior: Behavior.NON_BLOCKING,
                description: "Empieza o deja de ver la pantalla del PC en vivo. Solo hace falta si el señor Persus te ha dado permiso después de que preguntaras — si ya estás viendo la pantalla no la llames. Pregunta SIEMPRE en voz alta antes («¿Quiere que mire la pantalla?»); no la actives por iniciativa propia.",
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    activar: {
                      type: Type.BOOLEAN,
                      description: "true para empezar a verla, false para dejar de hacerlo."
                    }
                  },
                  required: ["activar"]
                }
              },
              {
                // La puerta de extensión (N-3): lo que no tenga herramienta
                // propia puede estar en un servidor MCP configurado.
                name: "listar_mcp",
                behavior: Behavior.NON_BLOCKING,
                description: "Lista los servidores MCP conectados y sus herramientas, con una descripción de cada una. Consúltala cuando el señor Persus pida algo para lo que no tienes herramienta concreta.",
                parameters: { type: Type.OBJECT, properties: {}, required: [] }
              },
              {
                name: "usar_mcp",
                behavior: Behavior.NON_BLOCKING,
                description: "Llama a una herramienta de un servidor MCP concreto. Los nombres y los argumentos deben encajar EXACTAMENTE con lo que te dijo listar_mcp — si el parámetro se llama 'timezone', no escribas 'time_zone'. No pidas permiso para usarla: si es de consulta (leer, listar, consultar la hora), ejecútala directamente; solo confirma antes con el señor Persus cuando sea claramente irreversible (escribir, borrar, enviar).",
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    servidor: {
                      type: Type.STRING,
                      description: "El nombre del servidor tal como salió en listar_mcp."
                    },
                    herramienta: {
                      type: Type.STRING,
                      description: "El nombre exacto de la herramienta."
                    },
                    argumentos: {
                      type: Type.OBJECT,
                      description: "Los parámetros de la herramienta, como objeto."
                    }
                  },
                  required: ["servidor", "herramienta", "argumentos"]
                }
              }
            ]
          }],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: {
                voiceName: defaultConfig.voiceName
              }
            }
          },
          systemInstruction: {
            parts: [{ text: finalSystemInstructionText }]
          },
          thinkingConfig: {
            thinkingLevel: ThinkingLevel.MINIMAL
          }
        },
        callbacks: {
          onopen: () => {
            console.log('[Gemini] WebSocket connection established');
            this.isConnecting = false;
            this.limpiarTopeDeConexion();
            // Abrir no es sobrevivir. El contador de intentos se perdona solo
            // cuando la llamada aguanta de verdad; si el servidor la echa antes,
            // la espera siguiente sube en vez de quedarse en un segundo.
            this.armarSesionEstable();
            // El contexto pendiente, por texto en vivo: es el único canal que
            // llega también a una sesión restaurada por testigo.
            if (this.entregarAlAbrir) {
              const texto = this.entregarAlAbrir;
              this.entregarAlAbrir = '';
              try {
                if (typeof (this.session as any)?.sendRealtimeInput === 'function') {
                  (this.session as any).sendRealtimeInput({ text: texto });
                }
              } catch (e) {
                console.warn('[Gemini] No se pudo entregar el contexto en vivo:', e);
              }
            }
            this.onConnectionStateChange('connected');
          },
          onmessage: (message: any) => this.handleMessage(message),
          onerror: (error: any) => {
            console.error('[Gemini] WebSocket Error:', error);
            this.isConnecting = false;
            this.limpiarTopeDeConexion();
            this.limpiarSesionEstable();
            this.onError(`Se perdió la conexión con el servidor: ${error.message || 'Error desconocido'}`);
            this.onConnectionStateChange('error');
          },
          onclose: (event: any) => {
            console.log('[Gemini] WebSocket Closed:', event);
            this.isConnecting = false;
            this.limpiarTopeDeConexion();
            this.limpiarSesionEstable();

            // Un testigo de sesión caducado no da error: el servidor cierra con
            // 1007 «Invalid session handle» y nada más. Como el testigo se
            // guardaba igual y el reintento lo volvía a mandar, cada intento
            // fallaba idéntico y la app se quedaba en «Conectando…» para
            // siempre, sin forma de salir desde la interfaz. Se tira y se
            // empieza de cero, que es exactamente lo que hace falta. Ver H-48.
            const motivo = String(event?.reason ?? '');

            // Por qué se cortó, en pantalla. Un cierre silencioso con reintento
            // detrás es indistinguible de "la app no funciona": se pasaron
            // horas mirando el certificado, la clave y la red antes de descubrir
            // que el servidor lo estaba diciendo y nadie lo enseñaba (H-48).
            if (!this.isManualDisconnect) {
              this.onError(
                `Se cortó la conexión (${event?.code ?? 'sin código'})` +
                  (motivo ? `: ${motivo}` : '')
              );
            }

            if (this.testigoSesion && (event?.code === 1007 || /session handle/i.test(motivo))) {
              console.warn('[Gemini] El servidor rechazó el testigo de sesión; se empieza de cero.');
              this.testigoSesion = null;
              localStorage.removeItem(CLAVE_TESTIGO);
              // Sin memoria de la sesión anterior, pero conectando: se reintenta
              // ya, no dentro de la espera larga que tocaría por los fallos.
              this.retryCount = 0;
            } else if (this.testigoSesion && this.retryCount >= 1) {
              // El servidor no siempre dice que el testigo es el problema: puede
              // aceptar la sesión y cerrarla acto seguido. Dos intentos seguidos
              // que ni llegan a estables con el mismo testigo puesto bastan para
              // sospechar de él, y una llamada sin memoria vale infinitamente
              // más que una llamada que no conecta.
              console.warn('[Gemini] Dos sesiones cortas seguidas con testigo; se descarta.');
              this.testigoSesion = null;
              localStorage.removeItem(CLAVE_TESTIGO);
            }

            if (!this.isManualDisconnect) {
                this.handleReconnect({ codigo: event?.code, motivo });
            }
          }
        }
      });
    } catch (e: any) {
      console.error('[Gemini] Connection failed:', e);
      this.isConnecting = false;
      this.limpiarTopeDeConexion();
      this.onError(`Error al conectar con Gemini: ${e.message || 'Fallo de red'}`);
      this.handleReconnect({ motivo: String(e?.message ?? e) });
    }
  }

  /**
   * Da por buena la sesión cuando lleva viva `MS_SESION_ESTABLE`.
   *
   * Aquí estaba el fallo de H-49: el contador se ponía a cero en cuanto el
   * socket abría, así que una sesión que el servidor cerraba un segundo después
   * dejaba la espera del reintento en `2^0` = 1 s, indefinidamente. Un intento
   * por segundo contra la API, que es justo lo que provoca el `1011` del que
   * intentaba recuperarse.
   */
  private armarSesionEstable() {
    this.limpiarSesionEstable();
    this.temporizadorEstable = window.setTimeout(() => {
      this.retryCount = 0;
      this.temporizadorEstable = null;
    }, MS_SESION_ESTABLE);
  }

  private limpiarSesionEstable() {
    if (this.temporizadorEstable) {
      clearTimeout(this.temporizadorEstable);
      this.temporizadorEstable = null;
    }
  }

  private armarTopeDeConexion() {
    this.limpiarTopeDeConexion();
    this.temporizadorConexion = window.setTimeout(() => {
      this.temporizadorConexion = null;
      if (!this.isConnecting) return;
      console.warn('[Gemini] La conexión no contestó a tiempo; se reintenta.');
      this.isConnecting = false;
      try {
        if (this.session && typeof this.session.close === 'function') this.session.close();
      } catch (e) {}
      this.onError('Gemini no contestó al conectar.');
      if (!this.isManualDisconnect) this.handleReconnect({ motivo: 'sin respuesta al conectar' });
    }, MS_TOPE_CONEXION);
  }

  private limpiarTopeDeConexion() {
    if (this.temporizadorConexion) {
      clearTimeout(this.temporizadorConexion);
      this.temporizadorConexion = null;
    }
  }

  private handleReconnect(cierre: CierreConexion = {}) {
    this.session = null;
    this.onConnectionStateChange('disconnected');

    // Sin tope de intentos: el techo está en la espera, no en el número. Una
    // caída de red se arregla sola cuando vuelve, tarde lo que tarde, y sin
    // esto Perseo se quedaba mudo hasta que alguien abría la ventana. Lo que sí
    // cambia según el motivo es cuánto se espera: ante un límite del servidor,
    // reintentar rápido es alimentar el problema.
    const plan = planificarReintento(this.retryCount, cierre);
    console.log(`[Gemini] Reintento ${this.retryCount + 1} (${plan.causa}) en ${plan.esperaMs} ms...`);
    if (!this.isManualDisconnect) this.onError(avisoDeEspera(plan));
    this.onConnectionStateChange('connecting');

    this.retryCount++;
    this.programarIntento(plan.esperaMs);
  }

  /** Un solo intento pendiente a la vez: dos temporizadores vivos son dos
   *  sesiones abriéndose a destiempo. */
  private programarIntento(esperaMs: number) {
    if (this.reconnectTimeout) clearTimeout(this.reconnectTimeout);
    this.reconnectTimeout = window.setTimeout(() => {
      this.reconnectTimeout = null;
      this.connect();
    }, esperaMs);
  }

  private async handleMessage(message: any) {
    if (message.toolCall) {
        console.log('[Gemini] Tool Call request recibido:', message.toolCall);
        // Sin `await`: cada herramienta se lanza y contesta por su cuenta. Antes
        // se ejecutaban en fila y no se mandaba nada hasta tener todas las
        // respuestas, así que una consulta lenta se llevaba por delante a las
        // rápidas. Con NON_BLOCKING el modelo sigue hablando mientras tanto, y
        // eso solo sirve de algo si aquí tampoco se espera.
        for (const call of message.toolCall.functionCalls ?? []) {
            void this.ejecutarHerramienta(call);
        }
        return; // Salimos para no procesar como modelTurn
    }

    if (message.sessionResumptionUpdate) {
        const { resumable, newHandle } = message.sessionResumptionUpdate;
        if (resumable && newHandle) {
            // El servidor rota el testigo durante la sesión. Se guarda el
            // último para que una caída de red —o cerrar la app— no obligue a
            // empezar la conversación otra vez.
            this.testigoSesion = newHandle;
            localStorage.setItem(CLAVE_TESTIGO, newHandle);
        }
        return;
    }

    // El servidor avisa antes de cortar por tiempo. Con el testigo guardado, la
    // reconexión recupera la sesión en vez de perderla.
    if (message.goAway) {
        console.warn('[Gemini] El servidor va a cerrar la sesión:', message.goAway);
        return;
    }

    const contenido = message.serverContent;
    if (!contenido) return;

    // Transcripciones. Llegan en fragmentos, no como frases completas.
    if (contenido.inputTranscription?.text) {
        this.onTranscript('user', contenido.inputTranscription.text, false);
    }
    if (contenido.outputTranscription?.text) {
        this.onTranscript('ai', contenido.outputTranscription.text, false);
    }

    if (contenido.modelTurn) {
        for (const part of contenido.modelTurn.parts || []) {
            if (part.inlineData?.data) {
                audioPlayer.enqueue(part.inlineData.data);
            }
            if (part.text) {
                this.onTranscript('ai', part.text, false);
            }
        }
    }

    if (contenido.interrupted) {
        console.log('[Gemini] Model turn interrupted');
        audioPlayer.clearQueue();
        this.onTranscript('ai', '', true);
    }

    // Fin de turno: cierra los mensajes abiertos de ambos lados.
    if (contenido.turnComplete) {
        this.onTranscript('user', '', true);
        this.onTranscript('ai', '', true);
        // Primer turno completo tras entregar avisos: el asunto se da por
        // contado. Por texto en vivo, que es el canal que una sesión
        // restaurada sí escucha — y así el cierre sobrevive a reconectarse
        // dentro de la misma llamada.
        if (this.cierreAvisoPendiente && !this.isManualDisconnect && this.session) {
          this.cierreAvisoPendiente = false;
          try {
            if (typeof (this.session as any)?.sendRealtimeInput === 'function') {
              (this.session as any).sendRealtimeInput({ text: CIERRE_DE_AVISOS });
              console.log('[Gemini] Avisos dados por informados y cerrados.');
            }
          } catch (e) {
            console.warn('[Gemini] El cierre de avisos no se pudo entregar:', e);
          }
        }
    }
  }

  /**
   * Saca a la pantalla de la llamada un trabajo que se quedó esperando un sí.
   *
   * El núcleo contesta «Queda pendiente de que lo confirmes: … (trabajo #57)» y
   * eso hasta ahora solo lo leía el modelo. La pregunta de verdad estaba en el
   * panel, que durante una llamada no se está mirando, así que un
   * `atajo_teclado` se quedaba parado para siempre y parecía que la herramienta
   * no funcionaba. Ver H-51.
   */
  private avisarSiEsperaUnSi(resultado: string) {
    const pendiente = resultado.match(/pendiente de que lo confirmes:\s*(.*?)\s*\(trabajo #(\d+)\)/i);
    if (!pendiente) return;
    this.onAprobacionPendiente(Number(pendiente[2]), pendiente[1]);
  }

  /**
   * Pasa a píxeles de pantalla lo que el modelo señaló sobre la imagen.
   *
   * El modelo apunta con las coordenadas normalizadas de 0 a 1000 sobre el JPEG
   * que recibe; `pc.py` clica en píxeles. Sin esta traducción, «clica el primer
   * resultado» caía a un tercio de donde debía. Ver H-50 y `coordenadas.ts`.
   *
   * Si la geometría no se puede leer, se manda lo que dijo el modelo: un clic
   * mal puesto es malo, pero peor es que la herramienta deje de funcionar por
   * una consulta que en el 99 % de los casos da igual.
   */
  private async traducirSiSeñala(name: string, args: any): Promise<any> {
    if (name !== 'controlar_pc' || !ACCIONES_DE_RATON.has(String(args?.accion ?? ''))) {
      return args;
    }
    try {
      if (!this.geometria) {
        this.geometria = await invoke<GeometriaPantalla>('geometria_pantalla');
      }
      const parametro = traducirParametroDeRaton(String(args?.parametro ?? ''), this.geometria);
      console.log(`[Gemini] Coordenadas ${args?.parametro} → ${parametro}`);
      return { ...args, parametro };
    } catch (e) {
      console.warn('[Gemini] No se pudo leer la geometría de la pantalla:', e);
      return args;
    }
  }

  /**
   * Ejecuta una herramienta y devuelve su resultado en cuanto lo tiene.
   *
   * Va aparte de `handleMessage` porque no se espera: mientras el núcleo
   * trabaja, el mensaje siguiente del modelo tiene que poder procesarse.
   */
  private async ejecutarHerramienta(call: any) {
    const { name, args, id } = call;
    console.log(`[Gemini] IA quiere ejecutar: ${name} con args:`, args);

    // `ver_pantalla` no viaja al núcleo: el ciclo de captura vive aquí mismo,
    // y quien lo enciende y apaga es la aplicación vía `onVerPantalla`.
    // Responder localmente evita un viaje de red para encender un intervalo.
    let response: Record<string, unknown>;
    if (name === 'ver_pantalla') {
      const activar = args?.activar !== false;
      try {
        response = { result: this.onVerPantalla(activar) };
      } catch (e: any) {
        response = { error: String(e) };
      }
    } else {
      try {
        const argumentos = await this.traducirSiSeñala(name, args);
        // El tope de espera vive en Rust (30 s), y cuando salta el trabajo sigue
        // vivo en la cola: no se pierde, solo deja de esperarse. Aquí había un
        // Promise.race de 10 s que abandonaba la promesa mientras el otro lado
        // seguía trabajando para un consumidor que ya no existía. Ver H-11 y H-12.
        const result = await invoke("ejecutar_herramienta", {
            toolName: name,
            argumentos: JSON.stringify(argumentos)
        }) as string;
        console.log(`[Gemini] Resultado de ${name}:`, result);
        this.avisarSiEsperaUnSi(result);
        if (name === 'responder_confirmacion' && Number.isFinite(Number(args?.id))) {
            // La voz ya resolvió lo que esta tarjeta enseñaba: fuera, o seguiría
            // ofreciendo botones para algo contestado hace unos segundos.
            this.onAprobacionResuelta(Number(args.id));
        }
        response = { result };
      } catch (e: any) {
        console.error(`[Gemini] Error ejecutando ${name}:`, e);
        response = { error: String(e) };
      }
    }

    // La sesión puede haberse caído mientras Python trabajaba. Mandar sobre una
    // sesión muerta lanza, y aquí nadie recogería la excepción. Pero el
    // resultado NO se tira: se guarda para la reconexión, que lo contará.
    if (!this.session) {
        console.warn(`[Gemini] Sesión caída; el resultado de ${name} espera a la reconexión.`);
        this.pendientesAlReconectar.push(`${name}: ${response.result ?? response.error}`);
        this.guardarPendientes();
        return;
    }

    const functionResponses = [{
        id,
        name,
        response,
        // Obligatorio con NON_BLOCKING: sin esto el modelo no sabe si cortar lo
        // que está diciendo o esperar a terminar la frase.
        scheduling: PLANIFICACION[name] ?? FunctionResponseScheduling.WHEN_IDLE,
    }];

    try {
        if (typeof this.session.sendToolResponse === 'function') {
            this.session.sendToolResponse({ functionResponses });
        } else if (typeof this.session.send === 'function') {
            this.session.send({ toolResponse: { functionResponses } });
        }
    } catch (e) {
        console.error(`[Gemini] No se pudo devolver el resultado de ${name}:`, e);
    }
  }

  public hardReset() {
    console.log('[Gemini] Ejecutando Hard Reset (Reinicio Completo)...');
    audioPlayer.clearQueue();
    // Reinicio completo quiere decir empezar de cero: si se conservara el
    // testigo, el modelo retomaría justo la conversación de la que se quiere
    // salir.
    this.testigoSesion = null;
    localStorage.removeItem(CLAVE_TESTIGO);
    this.disconnect();
    
    // Forzar el reinicio desde 0 reseteando los contadores
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    
    setTimeout(() => {
        this.retryCount = 0;
        this.connect();
    }, 1000);
  }

  sendAudioChunk(base64Data: string) {
    if (this.session) {
      this.session.sendRealtimeInput({
        audio: {
          mimeType: 'audio/pcm;rate=16000',
          data: base64Data
        }
      });
    }
  }

  sendVideoChunk(base64Data: string) {
    if (this.session) {
      this.session.sendRealtimeInput({
        video: {
          mimeType: 'image/jpeg',
          data: base64Data
        }
      });
    }
  }

  disconnect() {
    this.isManualDisconnect = true;
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    this.limpiarSesionEstable();
    this.limpiarTopeDeConexion();
    // Colgar a mano cierra el episodio: la llamada siguiente la pide una
    // persona, así que no arrastra la espera larga que hubiera acumulado el
    // bucle automático.
    this.retryCount = 0;
    // Ni etiquetas de una llamada anterior: el sentido y el cierre son de la
    // llamada en curso, y una colgada es una llamada muerta. Sin esto, un
    // aviso que nunca llegó a contarse mancharía la primera frase de la
    // siguiente.
    this.etiquetaOrigenSinEnviar = null;
    this.cierreAvisoPendiente = false;
    // Y colgar mata la SESIÓN. Si el testigo sobreviviera al colgar, la
    // siguiente llamada resucitaría esta conversación entera: el modelo
    // creería seguir a mitad de ella — le pasó al señor Persus el 2026-08-24,
    // anunciando en cada llamada nueva que «el s5 había terminado». El testigo
    // es para las RECONEXIONES automáticas de mitad de llamada, que no pasan
    // por aquí; para cruzar de una llamada a otra, no.
    this.testigoSesion = null;
    localStorage.removeItem(CLAVE_TESTIGO);

    if (this.session) {
       try {
          if (typeof this.session.close === 'function') {
              this.session.close();
          }
       } catch(e) {
          console.error('[Gemini] Error cerrando la sesión gracefully:', e);
       }
       this.session = null;
       this.isConnecting = false;
       this.onConnectionStateChange('disconnected');
    }
  }
}

export const geminiClient = new GeminiLiveClient();
