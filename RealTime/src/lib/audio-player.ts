/**
 * AudioPlayer — Gapless PCM playback at 24kHz for Gemini Live output.
 *
 * Strategy:
 *  1. Pre-buffer the first N chunks before scheduling anything (avoids initial stutter).
 *  2. Once streaming, schedule every chunk back-to-back on the AudioContext
 *     high-precision timeline — zero gap between buffers.
 *  3. Only reset the timeline if there's a TRUE gap (>300ms behind), which
 *     signals a new speech turn rather than normal network jitter.
 */

import { decodePCM, mergeChunks } from './audio-pcm';
export class AudioPlayer {
  private ctx: AudioContext | null = null;
  private nextTime = 0;
  private activeNodes: AudioBufferSourceNode[] = [];
  public onVolumeChange?: (volume: number) => void;
  private volumeInterval: number | null = null;
  private analyser: AnalyserNode | null = null;

  // Pre-buffer: accumulate chunks before scheduling to absorb network jitter
  private preBuffer: Float32Array[] = [];
  private isStreaming = false;
  private readonly PRE_BUFFER_COUNT = 3;       // Wait for 3 chunks (~180ms) before playing
  private readonly GAP_THRESHOLD_S = 0.3;      // 300ms gap → treat as new utterance
  private readonly INITIAL_DELAY_S = 0.08;     // 80ms initial delay for buffer headroom

  initialize() {
    if (!this.ctx) {
      this.ctx = new window.AudioContext({ sampleRate: 24000 });
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = 256;
      this.analyser.connect(this.ctx.destination);
      console.log('[AudioPlayer] AudioContext created (24kHz). State:', this.ctx.state);
      this.startVolumePolling();
    }
    if (this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
  }

  enqueue(base64Pcm: string) {
    if (!this.ctx) this.initialize();
    if (this.ctx!.state === 'suspended') this.ctx!.resume();

    const pcm = decodePCM(base64Pcm);

    if (!this.isStreaming) {
      // Accumulate in pre-buffer
      this.preBuffer.push(pcm);
      if (this.preBuffer.length >= this.PRE_BUFFER_COUNT) {
        this.flushPreBuffer();
      }
    } else {
      this.scheduleChunk(pcm);
    }
  }

  // ─── Internal ──────────────────────────────────────────────

  private flushPreBuffer() {
    this.isStreaming = true;

    // Merge all pre-buffered chunks into one large buffer to minimise node count
    const merged = mergeChunks(this.preBuffer);
    this.preBuffer = [];

    // Start slightly in the future so the very first sample isn't clipped
    this.nextTime = this.ctx!.currentTime + this.INITIAL_DELAY_S;
    this.scheduleChunk(merged);
  }

  private scheduleChunk(pcm: Float32Array) {
    if (!this.ctx) return;

    const now = this.ctx.currentTime;

    // If we've fallen far behind (true gap), reset timeline.
    // This happens at the start of a new AI utterance after silence.
    if (this.nextTime < now - this.GAP_THRESHOLD_S) {
      this.nextTime = now + this.INITIAL_DELAY_S;
    }

    // If we're only slightly behind (network jitter), DON'T add a gap —
    // just clamp to "now" so the chunk plays immediately.
    if (this.nextTime < now) {
      this.nextTime = now;
    }

    const buf = this.ctx.createBuffer(1, pcm.length, 24000);
    buf.getChannelData(0).set(pcm);

    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.analyser!);
    src.start(this.nextTime);

    this.activeNodes.push(src);
    src.onended = () => {
      const idx = this.activeNodes.indexOf(src);
      if (idx > -1) this.activeNodes.splice(idx, 1);
    };

    this.nextTime += buf.duration;
  }

  // El decodificado y el pegado de trozos viven en `audio-pcm.ts`: son pura
  // aritmética y así se pueden probar sin levantar un AudioContext. Ver H-28.

  // ─── Barge-in ──────────────────────────────────────────────

  private startVolumePolling() {
      const dataArray = new Uint8Array(this.analyser!.frequencyBinCount);
      const poll = () => {
          if (!this.analyser) return;
          this.analyser.getByteTimeDomainData(dataArray);
          let sum = 0;
          for (let i = 0; i < dataArray.length; i++) {
              const v = (dataArray[i] - 128) / 128;
              sum += v * v;
          }
          const rms = Math.sqrt(sum / dataArray.length);
          const normalized = Math.min(1, rms * 10); // Boost for visibility
          if (this.onVolumeChange) this.onVolumeChange(normalized);
          this.volumeInterval = requestAnimationFrame(poll);
      };
      this.volumeInterval = requestAnimationFrame(poll);
  }

  clearQueue() {
    this.isStreaming = false;
    this.preBuffer = [];
    this.nextTime = 0;

    if (this.volumeInterval) cancelAnimationFrame(this.volumeInterval);
    this.startVolumePolling(); // Keep polling but it will be 0

    // Hard-stop every scheduled node
    for (const node of this.activeNodes) {
      try { node.stop(); node.disconnect(); } catch (_) {}
    }
    this.activeNodes = [];
  }
}

export const audioPlayer = new AudioPlayer();
