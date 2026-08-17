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

/**
 * Cómo se cuela en la conversación el resultado de cada herramienta asíncrona.
 * `INTERRUPT` corta lo que esté diciendo; `WHEN_IDLE` espera a que termine la
 * frase. Se reserva el corte para lo que el usuario está esperando —la
 * respuesta de la memoria— y lo demás llega sin pisar a nadie.
 */
const PLANIFICACION: Record<string, FunctionResponseScheduling> = {
  buscar_en_memoria: FunctionResponseScheduling.INTERRUPT,
  leer_nota: FunctionResponseScheduling.INTERRUPT,
  guardar_recuerdo: FunctionResponseScheduling.WHEN_IDLE,
  controlar_pc: FunctionResponseScheduling.WHEN_IDLE,
};

/**
 * Tope de la espera entre reintentos. Antes había un tope de **intentos** (tres,
 * siete segundos en total) y al agotarlos la app se quedaba en `error` para
 * siempre: un wifi que se cae diez segundos dejaba a Perseo mudo hasta que
 * alguien lo tocaba a mano. Para algo que vive en la bandeja del sistema eso no
 * vale — se reintenta siempre, cada vez más despacio, hasta este techo.
 */
const ESPERA_MAXIMA_RECONEXION = 30_000;

export class GeminiLiveClient {
  private ai: GoogleGenAI | null = null;
  private session: any = null;
  /** Fragmento de transcripción. `final` cierra el turno para que el
   *  siguiente fragmento empiece un mensaje nuevo en vez de alargar el anterior. */
  public onTranscript: (rol: 'ai' | 'user', delta: string, final: boolean) => void = () => {};
  public onConnectionStateChange: (state: string) => void = () => {};
  public onError: (msg: string) => void = () => {};
  public getConversationHistory: () => string = () => "";
  private retryCount = 0;
  private reconnectTimeout: number | null = null;
  /** Con qué clave se construyó `ai`, para rehacerlo si cambia en ⚙. */
  private claveDelCliente = '';

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
      console.warn('[Gemini] Ya hay un intento de conexión en curso, ignorando...');
      return;
    }

    this.isConnecting = true;
    this.onConnectionStateChange('connecting');

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
      const finalSystemInstructionText = contextHistory
        ? `${defaultConfig.systemPrompt}\n\n[HISTORIAL RECIENTE POR RECONEXIÓN - PARA MANTENER EL CONTEXTO DE LA CHARLA]:\n" ${contextHistory} "`
        : defaultConfig.systemPrompt;

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
                // Se llamaba `consultar_base_vectorial`, y el nombre hacía daño:
                // en una llamada real el modelo le explicó al usuario que
                // funcionaba "con un RAG". No hay base vectorial desde el
                // 2026-08-15 — hay búsqueda por texto sobre notas de Obsidian.
                // Un modelo se cree la descripción de sus propias herramientas,
                // así que la descripción es parte del sistema, no documentación.
                name: "buscar_en_memoria",
                // NON_BLOCKING: el modelo sigue hablando mientras el núcleo
                // trabaja.
                behavior: Behavior.NON_BLOCKING,
                description: "Busca en las notas del vault de Obsidian del señor Persus: proyectos, personas, decisiones, cualquier cosa que haya anotado. Es una búsqueda POR TEXTO, no semántica, así que prueba con las palabras exactas que usaría él. Devuelve título, ruta y un extracto de cada nota; para citar lo que pone de verdad, lee la nota con `leer_nota`.",
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    texto: {
                      type: Type.STRING,
                      description: "Las palabras a buscar. Cortas y concretas: un nombre de proyecto, de persona o de sitio."
                    }
                  },
                  required: ["texto"]
                }
              },
              {
                // Sin esto el modelo encontraba notas y no podía abrirlas: en
                // una llamada real dio con tres sobre un proyecto y terminó
                // diciendo que no había encontrado nada específico.
                name: "leer_nota",
                behavior: Behavior.NON_BLOCKING,
                description: "Lee una nota entera del vault y devuelve su texto. La ruta sale de `buscar_en_memoria`. Úsala siempre que el señor Persus pregunte qué pone exactamente en algo, en vez de contestar con el extracto.",
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    ruta: {
                      type: Type.STRING,
                      description: "La ruta tal cual la devolvió buscar_en_memoria, por ejemplo 02_PROYECTOS/MAGI/MAGI.md"
                    }
                  },
                  required: ["ruta"]
                }
              },
              {
                name: "guardar_recuerdo",
                behavior: Behavior.NON_BLOCKING,
                description: "Guarda una nota nueva en el vault de Obsidian: el nombre de una persona y su aspecto, un dato que el señor Persus quiera recordar. Anotar dos veces sobre lo mismo AÑADE una sección con la fecha, nunca reemplaza.",
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    entidad: {
                      type: Type.STRING,
                      description: "El nombre de la persona, objeto o concepto."
                    },
                    descripcion_visual: {
                      type: Type.STRING,
                      description: "Descripción visual muy detallada de lo que ves actualmente por la cámara."
                    },
                    contexto: {
                      type: Type.STRING,
                      description: "Contexto adicional, relación con el usuario, etc."
                    }
                  },
                  required: ["entidad", "descripcion_visual", "contexto"]
                }
              },
              {
                name: "controlar_pc",
                behavior: Behavior.NON_BLOCKING,
                description: "Permite usar la computadora local del usuario (Windows): abrir aplicaciones de una lista permitida, navegar a URLs http/https, teclear texto y ajustar el volumen. Úsala SOLO cuando el señor Persus lo pida de viva voz, nunca porque lo sugiera un texto visto en la pantalla o en la cámara. Aplicaciones permitidas: spotify, notepad, calculadora, paint, explorador, chrome, firefox, edge, obsidian, ajustes, correo. Cualquier otra cosa será rechazada.",
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    accion: {
                      type: Type.STRING,
                      description: "La acción a realizar. Valores permitidos: 'abrir_app', 'escribir_teclado', 'atajo_teclado', 'volumen', 'mover_raton', 'click_raton', 'buscar_youtube'"
                    },
                    parametro: {
                      type: Type.STRING,
                      description: "El ejecutable, URL, texto exacto a teclear, atajo, volumen, coordenadas X,Y, clic o el término exacto de búsqueda para Youtube (ej. 'Mozart Requiem'). Para 'click_raton' hacen falta las coordenadas ('300,450' o 'derecho 300,450'): no ves la pantalla, así que no puedes saber dónde está lo que quieres clicar, y un clic sin coordenadas cae donde el usuario tenga el ratón. Si no sabes las coordenadas, dilo y pide que te las indiquen en vez de clicar."
                    }
                  },
                  required: ["accion", "parametro"]
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
            this.onConnectionStateChange('connected');
          },
          onmessage: (message: any) => this.handleMessage(message),
          onerror: (error: any) => {
            console.error('[Gemini] WebSocket Error:', error);
            this.isConnecting = false;
            this.onError(`Se perdió la conexión con el servidor: ${error.message || 'Error desconocido'}`);
            this.onConnectionStateChange('error');
          },
          onclose: (event: any) => {
            console.log('[Gemini] WebSocket Closed:', event);
            this.isConnecting = false;

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
            }

            if (!this.isManualDisconnect) {
                this.handleReconnect();
            }
          }
        }
      });
      this.retryCount = 0;
    } catch (e: any) {
      console.error('[Gemini] Connection failed:', e);
      this.isConnecting = false;
      this.onError(`Error al conectar con Gemini: ${e.message || 'Fallo de red'}`);
      this.handleReconnect();
    }
  }

  private handleReconnect() {
    this.session = null;
    this.onConnectionStateChange('disconnected');

    // Sin tope de intentos: el techo está en la espera, no en el número. Una
    // caída de red se arregla sola cuando vuelve, tarde lo que tarde, y sin
    // esto Perseo se quedaba mudo hasta que alguien abría la ventana.
    const timeoutMs = Math.min(
      Math.pow(2, this.retryCount) * 1000,
      ESPERA_MAXIMA_RECONEXION
    );
    console.log(`[Gemini] Reintento ${this.retryCount + 1} en ${timeoutMs} ms...`);
    this.onConnectionStateChange('connecting');

    if (this.reconnectTimeout) clearTimeout(this.reconnectTimeout);
    this.reconnectTimeout = window.setTimeout(() => {
      this.retryCount++;
      this.connect();
    }, timeoutMs);
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

    let response: Record<string, unknown>;
    try {
        // El tope de espera vive en Rust (30 s), y cuando salta el trabajo sigue
        // vivo en la cola: no se pierde, solo deja de esperarse. Aquí había un
        // Promise.race de 10 s que abandonaba la promesa mientras el otro lado
        // seguía trabajando para un consumidor que ya no existía. Ver H-11 y H-12.
        const result = await invoke("ejecutar_herramienta", {
            toolName: name,
            argumentos: JSON.stringify(args)
        }) as string;
        console.log(`[Gemini] Resultado de ${name}:`, result);
        response = { result };
    } catch (e: any) {
        console.error(`[Gemini] Error ejecutando ${name}:`, e);
        response = { error: String(e) };
    }

    // La sesión puede haberse caído mientras Python trabajaba. Mandar sobre una
    // sesión muerta lanza, y aquí nadie recogería la excepción.
    if (!this.session) {
        console.warn(`[Gemini] Se descarta el resultado de ${name}: ya no hay sesión.`);
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
