/**
 * El cliente de Gemini Live: la boca y los oídos de la llamada.
 * Es el fichero más grande de `src/lib/` y hasta el 2026-08-26 fue el único sin
 * una línea que dijera qué era. Lo que hay dentro, en orden:
 * 1. **La sesión**: apertura contra `v1alpha` con audio nativo, testigo de
 *    reanudación guardado (`perseo.sesion.testigo`) y cierre limpio. El porqué
 *    de cada constante está anotado justo debajo de ella.
 * 2. **Las herramientas**: lo que el modelo puede pedir. Aquí NO se ejecuta
 *    ninguna. Cada `toolCall` se convierte en un `invoke(...)` hacia Rust, y
 *    Rust lo encola en el núcleo (`src-tauri/src/nucleo.rs`). Es la regla de la
 *    casa: las caras no piensan. `PLANIFICACION` decide, herramienta por
 *    herramienta, si la respuesta corta lo que se esté diciendo o espera turno.
 * 3. **La reconexión**, cuya aritmética vive aparte —`reconexion.ts`— para
 *    poder probarla sin abrir un WebSocket.
 * 4. **Lo que se le cuenta al modelo sin que lo pida**: quién está delante
 *    (`quien-hay.ts`), quién empezó la llamada y qué avisos quedan pendientes
 *    (`aviso-llamada.ts`), cómo va el seguimiento de hábitos (`habitos.ts`) y
 *    por dónde va el tablero de tareas (`tareas.ts`).
 * Quien toque esto: la interfaz va incrustada en el binario, así que editar
 * este fichero no cambia nada hasta `python commands/perseo.py actualizar`.
 */

import {
  Behavior,
  EndSensitivity,
  FunctionResponseScheduling,
  GoogleGenAI,
  Modality,
  StartSensitivity,
  ThinkingLevel,
  Type,
} from '@google/genai';
import { invoke } from '@tauri-apps/api/core';
import { defaultConfig } from './config';
import { audioPlayer } from './audio-player';
import { apuntar, hablaElUsuario, respondePerseo } from './diagnostico';
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
import { bloqueCenso, type PerfilConocido } from './quien-hay';
import { resumenGuardado } from './habitos';
import {
  COLUMNAS as COLUMNAS_TAREAS,
  crearDeFuera as crearTarea,
  moverDeFuera as moverTarea,
  resumenGuardado as resumenTareas,
  type Columna as ColumnaTarea,
} from './tareas';

/**
 * Modelo de la Fase C. Se baja del 3.1 a propósito: el 3.1 **no soporta audio
 * proactivo ni llamadas a función asíncronas**, que son las dos patas sobre las
 * que se apoya toda esta fase.
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
 * Ojo: esto es la PREFERENCIA, no la última palabra. Si Perseo está hablando
 * en el momento de devolver el resultado, se degrada a `WHEN_IDLE` — ver
 * `ejecutarHerramienta`. Cortarle a media palabra se oye como una avería.
 */
const PLANIFICACION: Record<string, FunctionResponseScheduling> = {
  // La memoria vive en el servidor MCP 'vault' y la web, en el 'navegador';
  // los subagentes, en el servidor 'subagentes'. Lo que queda aquí es lo que
  // ningún servidor MCP cubre.
  controlar_pc: FunctionResponseScheduling.WHEN_IDLE,
  responder_confirmacion: FunctionResponseScheduling.WHEN_IDLE,
  consultar_agenda: FunctionResponseScheduling.INTERRUPT,
  situacion_actual: FunctionResponseScheduling.INTERRUPT,
  // Quien preguntó qué pone en sus notas espera la respuesta ahora, no
  // detrás de lo que se esté diciendo.
  buscar_en_memoria: FunctionResponseScheduling.INTERRUPT,
  leer_nota: FunctionResponseScheduling.INTERRUPT,
  listar_mcp: FunctionResponseScheduling.INTERRUPT,
  usar_mcp: FunctionResponseScheduling.INTERRUPT,
  ver_pantalla: FunctionResponseScheduling.INTERRUPT,
  // Ponerle nombre a quien tiene delante se contesta en el acto: quien acaba
  // de decir cómo se llama espera oírlo de vuelta, no treinta segundos después.
  nombrar_persona: FunctionResponseScheduling.INTERRUPT,
  quien_conozco: FunctionResponseScheduling.INTERRUPT,
  // Los hábitos se leen del `localStorage` de esta misma ventana: no hay viaje
  // de red que esperar, así que la respuesta corta lo que se esté diciendo en
  // vez de hacer cola detrás de ello.
  consultar_habitos: FunctionResponseScheduling.INTERRUPT,
  // El tablero está en el mismo `localStorage` y por el mismo motivo corta:
  // preguntar «¿qué tengo pendiente?» y que la lista llegue treinta segundos
  // después es no haberla preguntado.
  consultar_tareas: FunctionResponseScheduling.INTERRUPT,
  // Clavar y mover son escrituras en el `localStorage` de esta ventana: pasan
  // en el acto y se ven en el corcho al momento. Que la confirmación llegue
  // detrás de otra frase haría dudar de si se hizo.
  crear_tarea: FunctionResponseScheduling.INTERRUPT,
  mover_tarea: FunctionResponseScheduling.INTERRUPT,
};

export class GeminiLiveClient {
  private ai: GoogleGenAI | null = null;
  private session: any = null;
  /** Fragmento de transcripción. `final` cierra el turno para que el
   *  siguiente fragmento empiece un mensaje nuevo en vez de alargar el anterior. */
  public onTranscript: (rol: 'ai' | 'user', delta: string, final: boolean) => void = () => {};
  public onConnectionStateChange: (state: string) => void = () => {};
  public onError: (msg: string) => void = () => {};
  /** Alguien dejó de ser «Desconocido N». La aplicación repinta la etiqueta. */
  public onPersonaNombrada: (etiqueta: string, nombre: string) => void = () => {};
  /** Un trabajo que paró a pedir un sí. Durante una llamada la pregunta vivía
   *  solo en el panel y en Telegram, así que la acción no pasaba y el modelo se
   * quedaba diciendo «no parece que haya funcionado». */
  /**
   * Quién está hablando ahora mismo, según el reconocimiento de voz, o null si
   * no se sabe. Lo rellena la aplicación (App.tsx) desde el vigilante de
   * identidad, y viaja con CADA herramienta que se ejecute: el núcleo necesita
   * saber de quién es la voz que pide teclear, no solo que entró «por voz».
   * Sin él, una orden de una visita y una del señor Persus pesaban lo mismo.
   */
  public quienHabla: () => string | null = () => null;

  public onAprobacionPendiente: (id: number, pregunta: string) => void = () => {};
  /** Un trabajo pendiente que acaba de resolverse por voz. Saca su tarjeta de
   *  la pantalla: seguir ahí invitaba a pulsar lo que ya se contestó hablando. */
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
  /** Si el socket de esta sesión está abierto. Hace falta aparte de `session`
   *  porque `onopen` y la asignación de `session` no llegan en orden fijo:
   *  ver `entregarContextoEnVivo()`. */
  private socketAbierto = false;
  /**
   * Intentos seguidos **sin una sesión estable**. No se reinicia al abrir el
   * socket —eso era el bucle de— sino cuando una llamada aguanta
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
   * Va por `sendClientContent` con `turnComplete: false`, que es el canal para
   * añadir contexto SIN pedir turno. Antes iba por realtime-input, igual que el
   * motivo de llamada, y ahí el aviso entra como si alguien acabara de hablar:
   * el modelo contesta a lo que "ha oído" y el 2026-08-26 abrió la llamada
   * leyendo la marca —«[IDENTIDAD] Persus Buenos días, señor Persus»—. Como
   * contexto a secas, el aviso está cuando le toque hablar y no le empuja a
   * hablar por sí solo. Si la sesión está cerrada se guarda para entregarse al
   * abrir, igual que `entregarAlAbrir`.
   */
  informarIdentidad(texto: string): void {
    if (!texto) return;
    try {
      if (typeof (this.session as any)?.sendClientContent === 'function') {
        (this.session as any).sendClientContent({
          turns: [{ role: 'user', parts: [{ text: texto }] }],
          turnComplete: false,
        });
        return;
      }
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
   * El censo de gente conocida, traído del núcleo para las instrucciones.
   * Nunca lanza: si la biometría está apagada, sin modelos o el núcleo no
   * contesta, la llamada tiene que abrirse igual. Lo que se pierde entonces es
   * una lista de nombres, no la conversación.
   */
  private async censoDePersonas(): Promise<string | null> {
    if (!defaultConfig.identidadActivada) return null;
    try {
      const estado = await invoke<{ perfiles?: PerfilConocido[] }>('biometria_estado');
      return bloqueCenso(estado?.perfiles ?? [], defaultConfig.perfilPersus);
    } catch (e) {
      console.warn('[Gemini] Sin censo de personas para esta llamada:', e);
      return null;
    }
  }

  /**
   * El cliente del SDK, construido con la clave que haya **ahora**.
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
    this.socketAbierto = false;
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

      // A quién reconoce hoy el ordenador, escrito en las instrucciones antes
      // de que hable nadie. Un aviso [IDENTIDAD] suelto a mitad de frase llega
      // tarde: el modelo ya ha decidido cómo trata a quien tiene delante. Con
      // el censo entra sabiendo que hay más de una persona en esta casa.
      const censo = await this.censoDePersonas();
      if (censo) finalSystemInstructionText += `

${censo}`;

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

      // Colgar antes de llegar aquí cuenta. Entre la pulsación de «llamar» y
      // esta línea hay esperas (el censo de personas, entre otras), y en esa
      // ventana `disconnect()` no encuentra socket que cerrar: si no se mira
      // el testigo, la llamada colgada se abre igual unos segundos después.
      if (this.isManualDisconnect) {
        this.isConnecting = false;
        this.limpiarTopeDeConexion();
        this.onConnectionStateChange('disconnected');
        return;
      }

      const sesion = await this.cliente().live.connect({
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
          // "transcripción en tiempo real".
          inputAudioTranscription: {},
          outputAudioTranscription: {},
          // Cuándo se da por terminada una frase. Sin esto manda el ajuste de
          // fábrica, que es LOW en las dos puntas: el servidor tarda en dar por
          // empezada la voz y mucho más en darla por acabada, y esa espera es
          // la que se vivía como «Perseo tarda diez segundos en enterarse».
          // Con las dos sensibilidades altas y 600 ms de silencio la frase se
          // cierra cuando de verdad se acabó. El riesgo de cortar una pausa
          // larga lo cubre la proactividad: el modelo puede decidir callarse.
          // Con «pulsar para hablar» esto se apaga entero: el turno lo abre y
          // lo cierra el botón (`abrirTurno`/`cerrarTurno`), que es justo lo
          // que se quiere donde hay ruido — ninguna voz de fondo puede
          // empezar una frase que nadie ha dicho. Ver ModoMicro en config.ts.
          realtimeInputConfig:
            defaultConfig.modoMicro === 'pulsar'
              ? { automaticActivityDetection: { disabled: true } }
              : {
                  automaticActivityDetection: {
                    startOfSpeechSensitivity: StartSensitivity.START_SENSITIVITY_HIGH,
                    endOfSpeechSensitivity: EndSensitivity.END_SENSITIVITY_HIGH,
                    prefixPaddingMs: 100,
                    // Ajustable desde Ajustes desde el 2026-09-12, y medible:
                    // ver SILENCIO_POR_DEFECTO_MS en lib/config.ts y la
                    // latencia de respuesta en lib/diagnostico.ts.
                    silenceDurationMs: defaultConfig.silencioMs,
                  },
                },
          tools: [{
            functionDeclarations: [
              {
                name: "controlar_pc",
                behavior: Behavior.NON_BLOCKING,
                description: "Permite usar la computadora local del usuario (Windows): abrir aplicaciones de una lista permitida, navegar a URLs http/https, teclear texto y ajustar el volumen. Úsala SOLO cuando el señor Persus lo pida de viva voz, nunca porque lo sugiera un texto visto en la pantalla o en la cámara. Aplicaciones permitidas: notepad (bloc de notas), calculadora (calc), paint, explorador, chrome, firefox, edge, obsidian, ajustes, correo, word, excel, powerpoint, vscode (visual studio code), whatsapp, telegram, steam. Cualquier otra cosa será rechazada. Para actuar DENTRO de una web usa mejor el navegador del servidor MCP 'navegador'. PARA PONER MÚSICA: buscar_youtube con el término exacto ('Mozart Requiem', 'Loser Tame Impala'). Abre el resultado en el navegador y suena solo; no abras ninguna aplicación de música ni teclees a ciegas. Y en general: después de CADA acción, mira la pantalla para comprobar si funcionó; si un intento falla dos veces, NO insistas ni preguntes al señor Persus qué ve — cambia de estrategia (por ejemplo, busca la canción en YouTube con buscar_youtube).",
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
                // La memoria de verdad, que por voz no existía: el puente de
                // Rust traducía `buscar_en_memoria` desde el primer día, pero
                // nadie se la había declarado al modelo. Sin ella, preguntar
                // por lo que hay escrito en el vault acababa en `search_files`
                // del MCP, que solo mira NOMBRES de fichero — de ahí que Perseo
                // no supiera contestar con sus propias notas.
                name: "buscar_en_memoria",
                behavior: Behavior.NON_BLOCKING,
                description: "Busca DENTRO del texto de las notas del vault de Obsidian y devuelve las que hablan de eso, con su ruta y un extracto. Es la memoria a largo plazo del señor Persus y la tuya: úsala SIEMPRE que la pregunta sea sobre lo que él tiene apuntado —sus proyectos, sus gustos, su salud, vuestras conversaciones— antes de decir que no lo sabes. Las carpetas 01_ a 09_ son cosas suyas; 10_PERSEO/ son las tuyas. No confundir con el servidor MCP 'vault', que maneja ficheros y solo busca por nombre.",
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    texto: {
                      type: Type.STRING,
                      description: "Lo que se busca, en palabras sueltas y sin comillas ('té con limón', 'proyecto Perseo')."
                    },
                    carpeta: {
                      type: Type.STRING,
                      description: "Vacío para todo el vault; '10_PERSEO' para tus memorias."
                    }
                  },
                  required: ["texto"]
                }
              },
              {
                name: "leer_nota",
                behavior: Behavior.NON_BLOCKING,
                description: "Abre entera una nota del vault. La ruta sale tal cual de buscar_en_memoria; no te la inventes.",
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    ruta: {
                      type: Type.STRING,
                      description: "La ruta relativa que devolvió buscar_en_memoria, por ejemplo '10_PERSEO/Sobre Perseo.md'."
                    }
                  },
                  required: ["ruta"]
                }
              },
              {
                name: "guardar_recuerdo",
                behavior: Behavior.NON_BLOCKING,
                description: "Apunta algo en el vault para acordarse mañana. Añade, nunca sobrescribe. Úsala cuando el señor Persus cuente algo que merezca quedar escrito.",
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    entidad: {
                      type: Type.STRING,
                      description: "De quién o de qué es el recuerdo: el título de la nota."
                    },
                    contexto: {
                      type: Type.STRING,
                      description: "Lo que hay que recordar, en prosa."
                    },
                    descripcion_visual: {
                      type: Type.STRING,
                      description: "Solo si viene de algo que estás VIENDO por la cámara o la pantalla. Si no, se deja vacío."
                    }
                  },
                  required: ["entidad"]
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
                // El eslabón que faltaba entre la cámara y la memoria: el
                // reconocimiento sabe DISTINGUIR a una persona desde el primer
                // fotograma, pero no puede saber cómo se llama — eso solo lo
                // dice ella en voz alta, y hasta hoy nadie recogía la respuesta.
                // El 2026-08-25 el padre del señor Persus se quedó en
                // «Desconocido» toda la llamada por esto.
                name: "nombrar_persona",
                behavior: Behavior.NON_BLOCKING,
                description: "Le pone el nombre real a alguien que el reconocimiento etiquetó como «Desconocido N». Úsala en cuanto esa persona te diga cómo se llama: el perfil se queda hecho con ese nombre y se apunta una nota suya en «Perseo/Personas» del vault, así que la próxima vez la reconocerás sola. La etiqueta va COPIADA LITERAL del aviso [IDENTIDAD] («Desconocido 1», no «el desconocido»). No la uses para renombrar al señor Persus ni para inventar un nombre que nadie te haya dicho.",
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    etiqueta: {
                      type: Type.STRING,
                      description: "La etiqueta provisional tal cual vino en el aviso, por ejemplo 'Desconocido 1'."
                    },
                    nombre: {
                      type: Type.STRING,
                      description: "El nombre real, tal como la persona lo ha dicho. Por ejemplo 'Antonio'."
                    }
                  },
                  required: ["etiqueta", "nombre"]
                }
              },
              {
                name: "quien_conozco",
                behavior: Behavior.NON_BLOCKING,
                description: "A quién reconoce este ordenador por voz o por cara, con los que aún esperan nombre. Úsala cuando te pregunten a quién conoces, o antes de 'nombrar_persona' para no repetir un nombre que ya existe.",
                parameters: { type: Type.OBJECT, properties: {}, required: [] }
              },
              {
                // Los hábitos estaban en la pantalla de hábitos y en ningún
                // sitio más: el señor Persus podía verlos y Perseo no. Con esto
                // el seguimiento deja de ser una hoja bonita y pasa a ser algo
                // que se puede preguntar de viva voz a mitad de una llamada.
                name: "consultar_habitos",
                behavior: Behavior.NON_BLOCKING,
                description: "El seguimiento de hábitos del señor Persus tal como está ahora mismo: cuántas casillas lleva del mes y su porcentaje, cuáles le faltan HOY, las rachas vivas, los que peor van, el detalle hábito por hábito y las medias de ánimo y motivación. Úsala siempre que pregunte cómo va, qué le falta hoy, por su racha de algo, o cuando te pida que le animes o le eches en cara un hábito concreto: sin ella te lo estarías inventando. Es de solo lectura y no gasta cuota; no pidas permiso para llamarla. NO sirve para marcar ni desmarcar nada — eso lo hace él en la pantalla de hábitos.",
                parameters: { type: Type.OBJECT, properties: {}, required: [] }
              },
              {
                // El corcho estaba en la pantalla de tareas y en ningún sitio
                // más, igual que los hábitos antes de tener herramienta: él
                // veía sus notas y Perseo no. Con esto el tablero deja de ser
                // un tablón y pasa a ser algo sobre lo que se puede preguntar.
                name: "consultar_tareas",
                behavior: Behavior.NON_BLOCKING,
                description: "El tablero de tareas del señor Persus tal como está ahora mismo: cuántas notas lleva sin hacer, en proceso y completadas, qué tiene entre manos con el detalle de cada nota, lo que lleva días parado sin moverse, los pendientes y lo cerrado esta semana. Úsala siempre que pregunte qué tiene que hacer, por dónde va, qué se le está atascando, o cuando te pida ayuda para organizarse o elegir por dónde seguir: sin ella te lo estarías inventando. Es de solo lectura y no gasta cuota; no pidas permiso para llamarla. NO sirve para crear, mover ni tirar notas — eso lo hace él en la pantalla de tareas.",
                parameters: { type: Type.OBJECT, properties: {}, required: [] }
              },
              {
                // Escribir en el tablero, no solo leerlo. Pedido por el señor
                // Persus el 2026-09-03: apuntar una tarea mientras habla es lo
                // que hace que no se le olvide, y parar la conversación para ir
                // a la pantalla es exactamente lo que no quiere hacer.
                name: "crear_tarea",
                behavior: Behavior.NON_BLOCKING,
                description: "Clava una nota nueva en el tablero de tareas del señor Persus. Úsala cuando te pida apuntar algo, o cuando en la conversación aparezca algo que él dice que tiene que hacer. El título es corto y en infinitivo o imperativo, como lo escribiría él («Llamar al fontanero»), y el detalle es para lo que no cabe en el título — no repitas ahí el título. Se clava en «sin hacer» salvo que él diga otra cosa. Es inmediato y reversible: de la papelera se recupera, así que no pidas permiso para apuntar. NO la uses para recordarte cosas a ti: el tablero es suyo.",
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    titulo: {
                      type: Type.STRING,
                      description: "El título de la nota, corto. Es lo que se lee en el corcho."
                    },
                    detalle: {
                      type: Type.STRING,
                      description: "Lo que no cabe en el título: con quién, para cuándo, qué hace falta. Vacío si no hay nada que añadir."
                    },
                    columna: {
                      type: Type.STRING,
                      enum: ["sin_hacer", "en_proceso", "completadas"],
                      description: "Dónde se clava. Sin nada, «sin_hacer». Usa «en_proceso» solo si él dice que ya está con ello."
                    }
                  },
                  required: ["titulo"]
                }
              },
              {
                name: "mover_tarea",
                behavior: Behavior.NON_BLOCKING,
                description: "Mueve una nota del tablero a otra columna, buscándola por su título. Úsala cuando el señor Persus diga que ya ha terminado algo (a «completadas»), que se pone con ello («en_proceso»), o que lo tira («papelera»). El título no tiene que ser exacto: se busca sin tildes ni mayúsculas y basta con que empiece igual — pero si encajan dos notas no se mueve ninguna y te lo dirá, y entonces pregúntale a cuál se refiere. La papelera no borra: de ahí se recupera. Para borrar de verdad tiene que ir él a la pantalla.",
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    titulo: {
                      type: Type.STRING,
                      description: "El título de la nota, tal como él la ha llamado."
                    },
                    columna: {
                      type: Type.STRING,
                      enum: ["sin_hacer", "en_proceso", "completadas", "papelera"],
                      description: "La columna de destino."
                    }
                  },
                  required: ["titulo", "columna"]
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
            // Si ya se colgó mientras esto se abría, abrir no significa
            // «en llamada»: el socket que llega tarde lo cierra el guardián
            // de abajo, y aquí no se anuncia nada.
            if (this.isManualDisconnect) {
              this.limpiarTopeDeConexion();
              this.onConnectionStateChange('disconnected');
              return;
            }
            this.limpiarTopeDeConexion();
            // Abrir no es sobrevivir. El contador de intentos se perdona solo
            // cuando la llamada aguanta de verdad; si el servidor la echa antes,
            // la espera siguiente sube en vez de quedarse en un segundo.
            this.armarSesionEstable();
            // El contexto pendiente, por texto en vivo: es el único canal que
            // llega también a una sesión restaurada por testigo.
            this.socketAbierto = true;
            this.entregarContextoEnVivo();
            this.onConnectionStateChange('connected');
          },
          onmessage: (message: any) => this.handleMessage(message),
          onerror: (error: any) => {
            console.error('[Gemini] WebSocket Error:', error);
            this.isConnecting = false;
            this.socketAbierto = false;
            this.limpiarTopeDeConexion();
            this.limpiarSesionEstable();
            // Un socket que revienta porque acabamos de colgar no es un error
            // que enseñar: colgado ya está, y pintar «Error de enlace» encima
            // solo confunde.
            if (this.isManualDisconnect) {
              this.onConnectionStateChange('disconnected');
              return;
            }
            this.onError(`Se perdió la conexión con el servidor: ${error.message || 'Error desconocido'}`);
            this.onConnectionStateChange('error');
          },
          onclose: (event: any) => {
            console.log('[Gemini] WebSocket Closed:', event);
            apuntar(
              `enlace: cerrado (${event?.code ?? 'sin código'})` +
                (event?.reason ? `: ${String(event.reason)}` : '')
            );
            this.isConnecting = false;
            this.socketAbierto = false;
            this.limpiarTopeDeConexion();
            this.limpiarSesionEstable();

            // Un testigo de sesión caducado no da error: el servidor cierra con
            // 1007 «Invalid session handle» y nada más. Como el testigo se
            // guardaba igual y el reintento lo volvía a mandar, cada intento
            // fallaba idéntico y la app se quedaba en «Conectando…» para
            // siempre, sin forma de salir desde la interfaz. Se tira y se
            // empieza de cero, que es exactamente lo que hace falta.
            const motivo = String(event?.reason ?? '');

            // Por qué se cortó, en pantalla. Un cierre silencioso con reintento
            // detrás es indistinguible de "la app no funciona": se pasaron
            // horas mirando el certificado, la clave y la red antes de descubrir
            // que el servidor lo estaba diciendo y nadie lo enseñaba.
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

      // Y colgar mientras el socket se abría también cuenta: la sesión llega
      // ya huérfana, así que se cierra en vez de guardarla. Guardarla era la
      // llamada que volvía sola después de colgar.
      if (this.isManualDisconnect) {
        try {
          if (typeof sesion?.close === 'function') sesion.close();
        } catch (e) {}
        this.isConnecting = false;
        this.limpiarTopeDeConexion();
        this.limpiarSesionEstable();
        this.onConnectionStateChange('disconnected');
        return;
      }

      this.session = sesion;
      // Y el contexto en vivo, ahora que hay por dónde mandarlo. `onopen`
      // suele llegar ANTES que esta asignación, y allí `this.session` todavía
      // era null: el motivo de la llamada se descartaba en silencio. El que
      // llegue segundo es el que lo manda.
      this.entregarContextoEnVivo();
    } catch (e: any) {
      console.error('[Gemini] Connection failed:', e);
      this.isConnecting = false;
      this.limpiarTopeDeConexion();
      this.onError(`Error al conectar con Gemini: ${e.message || 'Fallo de red'}`);
      if (this.isManualDisconnect) {
        this.onConnectionStateChange('disconnected');
        return;
      }
      this.handleReconnect({ motivo: String(e?.message ?? e) });
    }
  }

  /**
   * Manda el contexto pendiente por texto en tiempo real, si ya hay sesión y
   * socket abierto.
   * Se llama desde los dos sitios que pueden completar esa pareja —`onopen` y
   * la asignación de `this.session`— porque el SDK no garantiza cuál va
   * primero. Es idempotente: quien llega segundo encuentra el texto y lo
   * manda; quien llega primero, no. Si el envío falla, el texto se queda
   * puesto para el siguiente intento en vez de perderse.
   */
  private entregarContextoEnVivo() {
    if (!this.entregarAlAbrir || !this.socketAbierto) return;
    const canal = this.session as any;
    if (typeof canal?.sendRealtimeInput !== 'function') return;
    const texto = this.entregarAlAbrir;
    this.entregarAlAbrir = '';
    try {
      canal.sendRealtimeInput({ text: texto });
    } catch (e) {
      console.warn('[Gemini] No se pudo entregar el contexto en vivo:', e);
      this.entregarAlAbrir = texto;
    }
  }

  /**
   * Da por buena la sesión cuando lleva viva `MS_SESION_ESTABLE`.
   * Aquí estuvo el fallo: el contador se ponía a cero en cuanto el
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
        const nombres = (message.toolCall.functionCalls ?? []).map((c: any) => c?.name).join(', ');
        apuntar(`herramienta: ${nombres || 'sin nombre'}`);
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
        // Al cuaderno también: un `1011` precedido de este aviso es un cierre
        // anunciado por tiempo, y uno sin aviso es otra cosa. Sin la línea, los
        // dos se ven igual desde fuera.
        apuntar(`enlace: el servidor avisa de que va a cerrar (${JSON.stringify(message.goAway)})`);
        return;
    }

    const contenido = message.serverContent;
    if (!contenido) return;

    // Transcripciones. Llegan en fragmentos, no como frases completas.
    if (contenido.inputTranscription?.text) {
        // Cada trozo de transcripción de entrada mueve el «dejó de hablar»
        // hacia delante; el último antes de que conteste Perseo es el bueno.
        // Ver latenciaDeRespuesta en lib/diagnostico.ts.
        hablaElUsuario();
        this.onTranscript('user', contenido.inputTranscription.text, false);
    }
    if (contenido.outputTranscription?.text) {
        this.onTranscript('ai', contenido.outputTranscription.text, false);
    }

    if (contenido.modelTurn) {
        for (const part of contenido.modelTurn.parts || []) {
            if (part.inlineData?.data) {
                // La primera sílaba de la respuesta cierra la medición.
                respondePerseo();
                audioPlayer.enqueue(part.inlineData.data);
            }
            if (part.text) {
                this.onTranscript('ai', part.text, false);
            }
        }
    }

    if (contenido.interrupted) {
        // Con cuánta respuesta ya recibida se corta. Es LA cifra para separar
        // los dos motivos de que a Perseo se le oiga entrecortado: si aquí se
        // tiran cientos de milisegundos una y otra vez, no es la red — es la
        // detección automática de voz tomando por orden el ruido de la sala (o
        // la propia voz de Perseo por los altavoces), y la cura es «pulsar
        // para hablar». Si no salen interrupciones, el corte es de la cola de
        // reproducción y lo cuenta `[AudioPlayer] Cola seca`.
        const reservaMs = audioPlayer.diagnostico().reservaMs;
        apuntar(`enlace: interrumpido — se tiran ${reservaMs} ms de respuesta ya recibida`);
        audioPlayer.clearQueue();
        this.onTranscript('ai', '', true);
    }

    // Fin de turno: cierra los mensajes abiertos de ambos lados.
    if (contenido.turnComplete) {
        // Y la frase siguiente vuelve a nacer con colchón. No corta nada de lo
        // que aún está sonando. Ver lib/audio-player.ts.
        audioPlayer.finDeTurno();
        apuntar(`enlace: turno completo — quedaban ${audioPlayer.diagnostico().reservaMs} ms por sonar`);
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
   * El núcleo contesta «Queda pendiente de que lo confirmes: … (trabajo #57)» y
   * eso hasta ahora solo lo leía el modelo. La pregunta de verdad estaba en el
   * panel, que durante una llamada no se está mirando, así que un
   * `atajo_teclado` se quedaba parado para siempre y parecía que la herramienta
   * no funcionaba.
   */
  private avisarSiEsperaUnSi(resultado: string) {
    const pendiente = resultado.match(/pendiente de que lo confirmes:\s*(.*?)\s*\(trabajo #(\d+)\)/i);
    if (!pendiente) return;
    this.onAprobacionPendiente(Number(pendiente[2]), pendiente[1]);
  }

  /**
   * Pasa a píxeles de pantalla lo que el modelo señaló sobre la imagen.
   * El modelo apunta con las coordenadas normalizadas de 0 a 1000 sobre el JPEG
   * que recibe; `pc.py` clica en píxeles. Sin esta traducción, «clica el primer
   * resultado» caía a un tercio de donde debía. Ver y `coordenadas.ts`.
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
    } else if (name === 'nombrar_persona') {
      // Tampoco viaja por `ejecutar_herramienta`: la biometría tiene su propio
      // puente en Rust y meterla en el router de herramientas del núcleo sería
      // un rodeo para llegar al mismo sitio.
      const etiqueta = String(args?.etiqueta ?? '').trim();
      const nombre = String(args?.nombre ?? '').trim();
      if (!etiqueta || !nombre) {
        response = { error: 'Hacen falta la etiqueta provisional y el nombre real.' };
      } else {
        try {
          const r = await invoke<{ ok?: boolean; error?: string; nombre?: string }>(
            'biometria_renombrar',
            { nombre: etiqueta, nuevoNombre: nombre },
          );
          if (r?.error) {
            response = { error: r.error };
          } else {
            const puesto = r?.nombre ?? nombre;
            this.onPersonaNombrada(etiqueta, puesto);
            response = {
              result: `Hecho: quien figuraba como «${etiqueta}» es ${puesto}. El perfil queda guardado y hay nota suya en Perseo/Personas.`,
            };
          }
        } catch (e: any) {
          response = { error: String(e) };
        }
      }
    } else if (name === 'quien_conozco') {
      const censo = await this.censoDePersonas();
      response = {
        result:
          censo ??
          'El reconocimiento de personas está apagado o todavía no reconoce a nadie en este ordenador.',
      };
    } else if (name === 'consultar_habitos') {
      // Tampoco viaja al núcleo: el seguimiento vive en el `localStorage` de
      // esta ventana y esta ventana es donde corre este código. Un viaje a
      // Python para leer algo que está en la memoria del proceso sería un rodeo
      // con dos formas nuevas de fallar —el núcleo apagado y el espejo viejo—
      // para llegar al mismo texto. El núcleo tiene su copia (ver `espejar()`)
      // porque el chat escrito no puede leer este almacén, no al revés.
      try {
        response = { result: resumenGuardado() };
      } catch (e: any) {
        response = { error: `No se pudo leer el seguimiento de hábitos: ${e}` };
      }
    } else if (name === 'consultar_tareas') {
      // Mismo caso que los hábitos: el tablero vive en el `localStorage` de
      // esta ventana, y esta ventana es donde corre este código. El núcleo
      // tiene su copia porque el chat escrito no puede leer este almacén, no
      // porque haga falta pasar por él para leerlo desde aquí.
      try {
        response = { result: resumenTareas() };
      } catch (e: any) {
        response = { error: `No se pudo leer el tablero de tareas: ${e}` };
      }
    } else if (name === 'crear_tarea' || name === 'mover_tarea') {
      // Escrituras, y también aquí dentro: el tablero está en el
      // `localStorage` de esta ventana y esta ventana es donde corre esto. La
      // pantalla, si está abierta, se entera sola —`aplicarDeFuera` avisa— y el
      // espejo del núcleo sale con el cambio siguiente.
      try {
        const columna = COLUMNAS_TAREAS.includes(args?.columna)
          ? (args.columna as ColumnaTarea)
          : undefined;
        response = {
          result:
            name === 'crear_tarea'
              ? crearTarea({
                  titulo: String(args?.titulo ?? ''),
                  detalle: String(args?.detalle ?? ''),
                  columna,
                })
              : columna
                ? moverTarea(String(args?.titulo ?? ''), columna)
                : 'No se ha movido nada: hace falta decir a qué columna.',
        };
      } catch (e: any) {
        response = { error: `No se pudo tocar el tablero de tareas: ${e}` };
      }
    } else {
      try {
        const argumentos = await this.traducirSiSeñala(name, args);
        // El tope de espera vive en Rust (30 s), y cuando salta el trabajo sigue
        // vivo en la cola: no se pierde, solo deja de esperarse. Aquí había un
        // Promise.race de 10 s que abandonaba la promesa mientras el otro lado
        // seguía trabajando para un consumidor que ya no existía. Ver y.
        const result = await invoke("ejecutar_herramienta", {
            toolName: name,
            argumentos: JSON.stringify(argumentos),
            // La etiqueta caduca a los 4,5 s de callarse (lib/identidad.ts), así
            // que esto es «quién acaba de hablar», que es justo a quien hay que
            // atribuir la orden. Si nadie ha hablado o el reconocimiento está
            // apagado, va null y el núcleo decide como siempre.
            quien: this.quienHabla() ?? null
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

    // Obligatorio con NON_BLOCKING: sin esto el modelo no sabe si cortar lo
    // que está diciendo o esperar a terminar la frase.
    // Y el corte SOLO se pide si ahora mismo no está hablando. El mapa de
    // arriba dice qué merece llegar cuanto antes; esto dice cuándo se puede
    // sin partir una palabra por la mitad. El 2026-09-09 el cuaderno de la
    // llamada lo dejó medido: 1.357 ms de voz ya recibida a la basura, 100 ms
    // después de `consultar_tareas`. Eso es lo que se oía entrecortado, y no
    // era ni la red ni el colchón de reproducción. Lo que se retrasa a cambio
    // es un «un momento, por favor» de dos segundos.
    const preferida = PLANIFICACION[name] ?? FunctionResponseScheduling.WHEN_IDLE;
    const hablando = audioPlayer.estaSonando();
    const scheduling = hablando ? FunctionResponseScheduling.WHEN_IDLE : preferida;
    if (hablando && preferida === FunctionResponseScheduling.INTERRUPT) {
        apuntar(`herramienta: ${name} espera a que acabe la frase en vez de cortarla`);
    }

    const functionResponses = [{
        id,
        name,
        response,
        scheduling,
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

  /**
   * Abre un turno a mano: «empiezo a hablar».
   * Solo tiene sentido con la detección automática apagada (modo `pulsar`).
   * Sin este aviso el servidor no da por empezada ninguna frase y el audio que
   * se le mande no se contesta nunca.
   */
  abrirTurno() {
    if (!this.session) return;
    this.session.sendRealtimeInput({ activityStart: {} });
  }

  /** Cierra el turno abierto: «he terminado, conteste». */
  cerrarTurno() {
    if (!this.session) return;
    this.session.sendRealtimeInput({ activityEnd: {} });
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
    this.socketAbierto = false;
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
    }

    // Colgar cuelga SIEMPRE, haya socket o no. Antes esto vivía dentro del
    // `if (this.session)`, y colgar mientras la conexión se estaba abriendo
    // —o durante la espera de una reconexión automática, donde la sesión ya
    // es null— no cambiaba el estado: la interfaz se quedaba en «Conectando…»
    // y el intento en vuelo terminaba de abrirse encima. Ver el guardián de
    // `isManualDisconnect` en `connect()`.
    this.isConnecting = false;
    this.onConnectionStateChange('disconnected');
  }
}

export const geminiClient = new GeminiLiveClient();
