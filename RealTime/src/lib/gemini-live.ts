import { GoogleGenAI, Modality, ThinkingLevel, Type } from '@google/genai';
import { invoke } from '@tauri-apps/api/core';
import { defaultConfig } from './config';
import { audioPlayer } from './audio-player';

export class GeminiLiveClient {
  private ai: GoogleGenAI;
  private session: any = null;
  /** Fragmento de transcripción. `final` cierra el turno para que el
   *  siguiente fragmento empiece un mensaje nuevo en vez de alargar el anterior. */
  public onTranscript: (rol: 'ai' | 'user', delta: string, final: boolean) => void = () => {};
  public onConnectionStateChange: (state: string) => void = () => {};
  public onError: (msg: string) => void = () => {};
  public getConversationHistory: () => string = () => "";
  private retryCount = 0;
  private reconnectTimeout: number | null = null;

  private isConnecting = false;
  private isManualDisconnect = false;

  constructor() {
    console.log('[Gemini] Initializing client... API Key present:', !!defaultConfig.geminiApiKey);
    this.ai = new GoogleGenAI({ apiKey: defaultConfig.geminiApiKey });
  }

  async connect() {
    this.isManualDisconnect = false;
    if (!defaultConfig.geminiApiKey) {
      this.onError('No se ha detectado la API Key. Por favor, asegúrate de haber reiniciado el servidor npm.');
      return;
    }
    
    if (this.isConnecting) {
      console.warn('[Gemini] Ya hay un intento de conexión en curso, ignorando...');
      return;
    }

    this.isConnecting = true;
    this.onConnectionStateChange('connecting');

    // Arrancar el proceso de herramientas ya, en paralelo a la conexión: así
    // precarga el índice vectorial mientras el usuario todavía está saludando,
    // y la primera consulta a la memoria responde en milisegundos.
    invoke('precalentar_herramientas').catch(e =>
      console.warn('[Gemini] No se pudo precalentar el puente de herramientas:', e)
    );

    // Asegurarnos de limpiar cualquier sesión residual antes de conectar de nuevo
    if (this.session) {
      try {
        if (typeof this.session.close === 'function') this.session.close();
      } catch (e) {}
      this.session = null;
    }

    try {
      const contextHistory = this.getConversationHistory();
      const finalSystemInstructionText = contextHistory 
        ? `${defaultConfig.systemPrompt}\n\n[HISTORIAL RECIENTE POR RECONEXIÓN - PARA MANTENER EL CONTEXTO DE LA CHARLA]:\n" ${contextHistory} "` 
        : defaultConfig.systemPrompt;

      this.session = await this.ai.live.connect({
        model: 'gemini-3.1-flash-live-preview',
        config: {
          responseModalities: [Modality.AUDIO],
          // Sin esto no hay transcripción en absoluto: con salida solo de audio
          // el modelo nunca envía partes de texto, así que el overlay únicamente
          // mostraba mensajes de sistema pese a que el README anunciaba
          // "transcripción en tiempo real". Ver H-05.
          inputAudioTranscription: {},
          outputAudioTranscription: {},
          tools: [{
            functionDeclarations: [
              {
                name: "consultar_base_vectorial",
                description: "Busca información en la memoria a largo plazo (base vectorial) sobre conocimientos pasados, personas que Perseo ya debió haber conocido, objetos o conceptos.",
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    query: {
                      type: Type.STRING,
                      description: "La pregunta o búsqueda detallada basada en las características visuales que ves o lo que el usuario pide."
                    }
                  },
                  required: ["query"]
                }
              },
              {
                name: "guardar_recuerdo",
                description: "Guarda un recuerdo, como el nombre de una persona y su rostro/apariencia, en la memoria a largo plazo.",
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
                      description: "El ejecutable, URL, texto exacto a teclear, atajo, volumen, coordenadas X,Y, clic('izquierdo', 'derecho') o el término exacto de búsqueda para Youtube (ej. 'Mozart Requiem')."
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

    const maxRetries = 3;
    if (this.retryCount < maxRetries) {
       const timeoutMs = Math.pow(2, this.retryCount) * 1000;
       console.log(`[Gemini] Reconnecting in ${timeoutMs}ms...`);
       this.onConnectionStateChange('connecting');
       
       if (this.reconnectTimeout) clearTimeout(this.reconnectTimeout);
       this.reconnectTimeout = window.setTimeout(() => {
          this.retryCount++;
          this.connect();
       }, timeoutMs);
    } else {
       console.error('[Gemini] Max reconnection attempts reached.');
       this.onConnectionStateChange('error');
    }
  }

  private async handleMessage(message: any) {
    if (message.toolCall) {
        console.log('[Gemini] Tool Call request recibido:', message.toolCall);
        const functionCalls = message.toolCall.functionCalls;
        
        if (functionCalls && functionCalls.length > 0) {
            const functionResponses = [];
            
            for (const call of functionCalls) {
                const { name, args, id } = call;
                console.log(`[Gemini] IA quiere ejecutar: ${name} con args:`, args);
                
                try {
                    // El timeout vive ahora en Rust, que además mata el proceso.
                    // Aquí había un Promise.race de 10 s que abandonaba la promesa
                    // pero dejaba a Python trabajando para un consumidor que ya no
                    // existía, y que además saltaba siempre en la primera consulta
                    // al RAG (7,5 s de arranque en frío). Ver H-11 y H-12.
                    const result = await invoke("ejecutar_herramienta_python", {
                        toolName: name,
                        argumentos: JSON.stringify(args)
                    }) as string;

                    console.log(`[Gemini] Resultado de ${name}:`, result);
                    functionResponses.push({
                        id,
                        name,
                        response: { result: result }
                    });
                } catch (e: any) {
                    console.error(`[Gemini] Error ejecutando ${name}:`, e);
                    functionResponses.push({
                        id,
                        name,
                        response: { error: String(e) }
                    });
                }
            }
            
            // Devolver las respuestas a Gemini para que continúe hablando
            if (this.session && typeof this.session.sendToolResponse === 'function') {
                this.session.sendToolResponse({ functionResponses });
            } else if (this.session && typeof this.session.send === 'function') {
                this.session.send({ toolResponse: { functionResponses } });
            }
        }
        return; // Salimos para no procesar como modelTurn
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

  public hardReset() {
    console.log('[Gemini] Ejecutando Hard Reset (Reinicio Completo)...');
    audioPlayer.clearQueue();
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
