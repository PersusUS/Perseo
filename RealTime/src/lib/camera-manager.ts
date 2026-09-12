import { geminiClient } from './gemini-live';
import { defaultConfig } from './config';

class CameraManager {
  private stream: MediaStream | null = null;
  private videoElement: HTMLVideoElement | null = null;
  private canvasElement: HTMLCanvasElement | null = null;
  private intervalId: number | null = null;
  public onStreamReady: (stream: MediaStream) => void = () => {};
  /**
   * Gancho para el reconocimiento de personas (lib/identidad.ts): el mismo
   * JPEG que va a Gemini, sin segunda captura. El vigilante decide si mandarlo
   * al núcleo o descartarlo según su ritmo propio.
   */
  public onFotograma: ((base64: string) => void) | null = null;

  async start() {
    if (this.stream) return;

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' }
      });

      this.videoElement = document.createElement('video');
      this.videoElement.srcObject = this.stream;
      this.videoElement.muted = true;
      this.videoElement.playsInline = true;
      await this.videoElement.play();

      this.canvasElement = document.createElement('canvas');
      this.canvasElement.width = 640;
      this.canvasElement.height = 480;

      const fps = defaultConfig.cameraFps;
      const intervalMs = 1000 / fps;

      this.intervalId = window.setInterval(() => this.captureFrame(), intervalMs);
      this.onStreamReady(this.stream);
    } catch (e) {
      console.error('Camera capture failed:', e);
    }
  }

  private captureFrame() {
    if (!this.videoElement || !this.canvasElement || this.videoElement.videoWidth === 0) return;

    const ctx = this.canvasElement.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(this.videoElement, 0, 0, 640, 480);
    const dataUrl = this.canvasElement.toDataURL('image/jpeg', 0.7);
    const base64 = dataUrl.split(',')[1];
    
    this.onFotograma?.(base64);
    geminiClient.sendVideoChunk(base64);
  }

  stop() {
    if (this.intervalId) {
      window.clearInterval(this.intervalId);
      this.intervalId = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
    this.videoElement = null;
    this.canvasElement = null;
  }
}

export const cameraManager = new CameraManager();
