import { geminiClient } from './gemini-live';

export class AudioManager {
  private captureContext: AudioContext | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private sourceNode: MediaStreamAudioSourceNode | null = null;
  private stream: MediaStream | null = null;
  
  public onError: (msg: string) => void = () => {};
  public onStreamReady: () => void = () => {};
  /**
   * Gancho para el reconocimiento de personas (lib/identidad.ts): recibe el
   * MISMO trozo PCM que va a Gemini, sin copiar el stream ni abrir otro
   * worklet. Si nadie lo registra, no se hace nada — coste cero.
   */
  public onTrozoPCM: ((trozo: ArrayBuffer) => void) | null = null;

  async start() {
    if (this.captureContext) return;

    try {
      console.log('[AudioManager] Attempting to start...');
      this.captureContext = new AudioContext({ sampleRate: 16000 });
      console.log('[AudioManager] AudioContext (16kHz) created. State:', this.captureContext.state);
      
      await this.captureContext.audioWorklet.addModule('/audio-processor.js');
      console.log('[AudioManager] AudioWorklet module loaded');

      this.stream = await navigator.mediaDevices.getUserMedia({ 
        audio: { 
          sampleRate: 16000, 
          channelCount: 1, 
          echoCancellation: true, 
          noiseSuppression: true 
        } 
      });
      console.log('[AudioManager] Microphone stream obtained successfully');

      this.sourceNode = this.captureContext.createMediaStreamSource(this.stream);
      this.workletNode = new AudioWorkletNode(this.captureContext, 'audio-capture');
      this.sourceNode.connect(this.workletNode);

      this.workletNode.port.onmessage = (event) => {
        const pcmBuffer = event.data; // ArrayBuffer of Int16
        this.onTrozoPCM?.(pcmBuffer);
        const base64 = this.arrayBufferToBase64(pcmBuffer);
        geminiClient.sendAudioChunk(base64);
      };

      this.onStreamReady();

    } catch (e: any) {
      console.error('[AudioManager] Start failed:', e);
      let errorMsg = 'Error desconocido al acceder al micrófono.';
      if (e.name === 'NotAllowedError') {
        errorMsg = 'Permiso de micrófono denegado. Por favor, reinicia la app y permite el acceso.';
      } else if (e.name === 'NotFoundError') {
        errorMsg = 'No se encontró ningún micrófono conectado.';
      }
      this.onError(errorMsg);
      this.stop();
    }
  }

  stop() {
    if (this.workletNode) {
      this.workletNode.disconnect();
      this.workletNode = null;
    }
    if (this.sourceNode) {
      this.sourceNode.disconnect();
      this.sourceNode = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
    if (this.captureContext) {
      this.captureContext.close();
      this.captureContext = null;
    }
  }

  private arrayBufferToBase64(buffer: ArrayBuffer): string {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    const len = bytes.byteLength;
    const chunkSize = 8192;
    for (let i = 0; i < len; i += chunkSize) {
        binary += String.fromCharCode.apply(null, Array.from(bytes.subarray(i, Math.min(i + chunkSize, len))));
    }
    return window.btoa(binary);
  }
}

export const audioManager = new AudioManager();
