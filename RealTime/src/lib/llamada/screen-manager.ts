/**
 * La pantalla, vista por el modelo: una captura cada dos segundos.
 * La captura no la hace el navegador sino Rust (`capture_screen_base64`, con
 * `xcap`), porque el WebView embebido no puede grabar el escritorio que lo
 * contiene. Aquí solo está el reloj: pedir el JPEG, mandarlo a la sesión de
 * Gemini y pararse cuando toca.
 * Las dos banderas (`isCapturing`, `isRunning`) no son adorno: sin la primera,
 * una captura lenta se solapa con la siguiente y se envían fotogramas fuera de
 * orden; sin la segunda, la captura que estaba en vuelo cuando se llamó a
 * `stop()` llega después y reabre el envío con la sesión ya cerrada.
 */

import { invoke } from '@tauri-apps/api/core';
import { geminiClient } from './gemini-live';
import { defaultConfig } from '../datos/config';
import { apuntar } from './diagnostico';

class ScreenManager {
  private intervalId: number | null = null;
  private isCapturing = false;
  private isRunning = false;
  public onFrameReady: ((base64: string) => void) | null = null;

  start() {
    if (this.intervalId) return;
    
    this.isRunning = true;

    const fps = defaultConfig.screenFps;
    const intervalMs = 1000 / fps;

    this.intervalId = window.setInterval(async () => {
      if (this.isRunning) {
        await this.captureAndSend();
      }
    }, intervalMs);
  }

  private async captureAndSend() {
    if (this.isCapturing || !this.isRunning) return; // prevent overlap or sending after stop
    this.isCapturing = true;
    
    try {
      // Call Rust command
      const pedida = performance.now();
      const base64Jpeg: string = await invoke('capture_screen_base64', { 
        quality: defaultConfig.screenQuality 
      });
      const traida = performance.now();

      // Checking again in case stop() was called while we were waiting for invoke
      if (!this.isRunning) return;

      geminiClient.sendVideoChunk(base64Jpeg);
      // Lo que cuesta MANDARLA corre en el hilo principal, y ahí es donde vive
      // también la reproducción: si esto tarda, el audio se queda sin quien lo
      // alimente y se oye entrecortado. Se apunta solo cuando pasa de un
      // umbral, que si no el cuaderno sería una lista de líneas iguales.
      const enviada = performance.now();
      if (enviada - pedida > 150) {
        apuntar(
          `pantalla: captura ${Math.round(traida - pedida)} ms + envío ` +
          `${Math.round(enviada - traida)} ms (${Math.round(base64Jpeg.length / 1024)} kB)`
        );
      }
      if (this.onFrameReady) {
        this.onFrameReady(base64Jpeg);
      }
    } catch (e) {
      console.error('Screen capture failed:', e);
    } finally {
      this.isCapturing = false;
    }
  }

  stop() {
    this.isRunning = false;
    if (this.intervalId) {
      window.clearInterval(this.intervalId);
      this.intervalId = null;
    }
  }
}

export const screenManager = new ScreenManager();
