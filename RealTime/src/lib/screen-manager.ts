import { invoke } from '@tauri-apps/api/core';
import { geminiClient } from './gemini-live';
import { defaultConfig } from './config';

export class ScreenManager {
  private intervalId: number | null = null;
  private isCapturing = false;
  private isRunning = false;
  public onFrameReady: ((base64: string) => void) | null = null;

  start() {
    if (this.intervalId) return;
    
    console.log('[ScreenManager] Starting capture loop...');
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
      const base64Jpeg: string = await invoke('capture_screen_base64', { 
        quality: defaultConfig.screenQuality 
      });

      // Checking again in case stop() was called while we were waiting for invoke
      if (!this.isRunning) return;

      geminiClient.sendVideoChunk(base64Jpeg);
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
    console.log('[ScreenManager] Stopped capture loop.');
  }
}

export const screenManager = new ScreenManager();
