import { GoogleGenAI, Modality, ThinkingLevel, Type } from '@google/genai';
import { invoke } from '@tauri-apps/api/core';
import { defaultConfig } from './config';
import { audioPlayer } from './audio-player';

export class GeminiLiveClient {
  private ai: GoogleGenAI;
  private session: any = null;
  public onTranscriptChange: (text: string) => void = () => {};
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
                description: "Permite usar la computadora local del usuario (Windows). Sirve para abrir aplicaciones registradas en el sistema, navegar a URLs específicas, escribir texto interactivo (teclear) y ajustar volumen general.",
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
                    // Timeout preventor: No queremos que Perseo se quede colgado eternamente 
                    // si Python se bloquea (por ejemplo, fallo en local Ollama o lectura del vault)
                    const timeoutPromise = new Promise((_, reject) => 
                        setTimeout(() => reject(new Error('Timeout: Python tardó más de 10 segundos en responder')), 10000)
                    );

                    // Llamamos al backend de Rust (Tauri) para que ejecute el Python
                    const result = await Promise.race([
                        invoke("ejecutar_herramienta_python", { 
                            toolName: name, 
                            argumentos: JSON.stringify(args)
                        }),
                        timeoutPromise
                    ]) as string;
                    
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

    if (message.serverContent && message.serverContent.modelTurn) {
        const parts = message.serverContent.modelTurn.parts || [];
        for (const part of parts) {
            if (part.inlineData && part.inlineData.data) {
                console.log('[Gemini] Received audio data');
                audioPlayer.enqueue(part.inlineData.data);
            }
            if (part.text) {
                console.log('[Gemini] Received transcript part:', part.text);
                this.onTranscriptChange(part.text);
            }
        }
    } else if (message.serverContent && message.serverContent.interrupted) {
        console.log('[Gemini] Model turn interrupted');
        audioPlayer.clearQueue();
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
