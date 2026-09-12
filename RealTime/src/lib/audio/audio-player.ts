/**
 * AudioPlayer — Gapless PCM playback at 24kHz for Gemini Live output.
 * Strategy:
 * 1. Pre-buffer ~200 ms of audio before scheduling anything (avoids initial stutter).
 *  2. Once streaming, schedule every chunk back-to-back on the AudioContext
 *     high-precision timeline — zero gap between buffers.
 * 3. If the scheduled queue runs dry — a new utterance after silence, a
 * network hiccup, or the main thread blocked by a screen capture — the
 * cushion is rebuilt instead of playing the late chunk on the spot.
 * El punto 3 es lo que se arregló el 2026-09-09, y es el que faltaba: hasta
 * entonces `isStreaming` se ponía a `true` con el primer colchón y ya no
 * volvía a bajar en toda la llamada. A partir de ahí cada trozo que llegaba
 * tarde se programaba «para ya mismo», así que el silencio entre lo que se
 * acabó de oír y lo que llega se oía tal cual: la respuesta entrecortada de
 * la que se quejaba el señor Persus. Rehacer el colchón cuesta 200 ms una vez
 * y evita el corte en cada trozo posterior.
 * Esto NO es la simplificación que el roadmap prohíbe: la estrategia de
 * colchón + línea de tiempo sigue igual, lo que se añade es volver a llenarlo
 * cuando se seca.
 */

import { decodePCM, mergeChunks } from './audio-pcm';
import { apuntar } from '../llamada/diagnostico';
class AudioPlayer {
  private ctx: AudioContext | null = null;
  private nextTime = 0;
  private activeNodes: AudioBufferSourceNode[] = [];
  public onVolumeChange?: (volume: number) => void;
  private volumeInterval: number | null = null;
  private analyser: AnalyserNode | null = null;

  // Pre-buffer: accumulate chunks before scheduling to absorb network jitter
  private preBuffer: Float32Array[] = [];
  private muestrasEnColchon = 0;
  private isStreaming = false;
  private readonly PRE_BUFFER_S = 0.2; // 200ms de colchón antes de sonar
  /**
   * El colchón de verdad, que crece con los disgustos.
   * Empieza en los 200 ms de arriba y sube 100 ms cada vez que la cola se
   * queda seca, hasta 600 ms. Un colchón fijo no puede estar bien: en una red
   * buena 200 ms sobran y en una mala se quedan cortos, y quien sabe cuál de
   * las dos hay delante es la propia llamada. Lo que se paga al subirlo es
   * retraso al empezar cada frase; lo que se evita es el corte a mitad, que
   * molesta muchísimo más.
   */
  private colchonS = this.PRE_BUFFER_S;
  private readonly COLCHON_MAX_S = 0.6;
  private readonly COLCHON_PASO_S = 0.1;
  private readonly PRE_BUFFER_MAX = 12; // Tope duro por si llegan trozos diminutos
  private readonly INITIAL_DELAY_S = 0.08;     // 80ms initial delay for buffer headroom
  /**
   * Cuánta reserva programada tiene que quedar para no dar la cola por seca.
   * No es cero a propósito: si se espera a que la reserva llegue exactamente a
   * cero, el corte YA se ha oído. Con 40 ms se rehace el colchón justo antes.
   */
  private readonly MARGEN_SECO_S = 0.04;

  /** Veces que la cola se ha quedado seca en esta llamada. Es la cifra que
   * dice si el audio entrecortado viene de la red o de otra cosa: se lee con
   * `audioPlayer.diagnostico()` desde la consola. */
  private vecesSeca = 0;
  private trozosRecibidos = 0;
  /** Cuándo llegó el trozo anterior. Solo para el cuaderno: un hueco largo
   * aquí dice que el silencio venía de fuera y no de la reproducción. */
  private ultimoTrozoMs = 0;

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
    this.trozosRecibidos++;
    const ahora = performance.now();
    const desdeElAnterior = this.ultimoTrozoMs ? Math.round(ahora - this.ultimoTrozoMs) : 0;
    this.ultimoTrozoMs = ahora;

    // ¿Queda reserva programada? Si no, este trozo llega tarde: sonarlo ahora
    // mismo es exactamente lo que se oye como un corte. Se vuelve a llenar el
    // colchón antes de seguir.
    if (this.isStreaming && this.ctx!.currentTime > this.nextTime - this.MARGEN_SECO_S) {
      this.isStreaming = false;
      this.vecesSeca++;
      const reservaMs = Math.round((this.nextTime - this.ctx!.currentTime) * 1000);
      this.colchonS = Math.min(this.COLCHON_MAX_S, this.colchonS + this.COLCHON_PASO_S);
      apuntar(
        `audio: cola seca (${this.vecesSeca}ª) — reserva ${reservaMs} ms, ` +
        `${desdeElAnterior} ms desde el trozo anterior, trozo de ${Math.round((pcm.length / 24000) * 1000)} ms; ` +
        `colchón a ${Math.round(this.colchonS * 1000)} ms`
      );
    }

    if (!this.isStreaming) {
      // Accumulate in pre-buffer
      this.preBuffer.push(pcm);
      this.muestrasEnColchon += pcm.length;
      if (
        this.muestrasEnColchon / 24000 >= this.colchonS ||
        this.preBuffer.length >= this.PRE_BUFFER_MAX
      ) {
        this.flushPreBuffer();
      }
    } else {
      this.scheduleChunk(pcm);
    }
  }

  /**
   * El modelo ha terminado de hablar (`turnComplete`).
   * No corta nada de lo que ya está programado: solo baja la bandera para que
   * la SIGUIENTE frase vuelva a nacer con colchón. Sin esto, la primera frase
   * de la llamada era la única que lo tenía.
   */
  finDeTurno() {
    // Y lo que quede a medio colchón suena YA: una respuesta de dos palabras
    // puede no llegar a los 200 ms, y esperar a llenar el colchón la dejaría
    // muda para siempre.
    if (this.preBuffer.length) this.flushPreBuffer();
    this.isStreaming = false;
  }

  /**
   * ¿Le queda voz por decir ahora mismo?
   * Lo pregunta `gemini-live.ts` antes de devolver el resultado de una
   * herramienta: cortarle a media frase se oye como un fallo, y lo que hay
   * programado aquí es la única fuente honesta de «está hablando» — el estado
   * de React llega tarde y va por otro camino.
   */
  estaSonando(): boolean {
    if (!this.ctx) return false;
    return this.nextTime > this.ctx.currentTime + 0.05;
  }

  /** Lo que se le pregunta a la consola cuando el audio se oye entrecortado. */
  diagnostico() {
    return {
      trozosRecibidos: this.trozosRecibidos,
      vecesSeca: this.vecesSeca,
      colchonMs: Math.round(this.colchonS * 1000),
      // Nunca en negativo: sin nada programado la reserva es cero, y un
      // «quedan -175 s por sonar» en el cuaderno solo confunde al que lo lea.
      reservaMs: this.ctx
        ? Math.max(0, Math.round((this.nextTime - this.ctx.currentTime) * 1000))
        : 0,
      nodosVivos: this.activeNodes.length,
      estadoContexto: this.ctx?.state ?? 'sin contexto',
    };
  }

  // ─── Internal ──────────────────────────────────────────────

  private flushPreBuffer() {
    this.isStreaming = true;

    // Merge all pre-buffered chunks into one large buffer to minimise node count
    const merged = mergeChunks(this.preBuffer);
    this.preBuffer = [];
    this.muestrasEnColchon = 0;

    // Start slightly in the future so the very first sample isn't clipped.
    // El `max` es lo que impide pisar lo que aún está sonando: al rehacer el
    // colchón a mitad de frase, arrancar en «ahora» solaparía las dos voces.
    this.nextTime = Math.max(
      this.nextTime,
      this.ctx!.currentTime + this.INITIAL_DELAY_S
    );
    this.scheduleChunk(merged);
  }

  private scheduleChunk(pcm: Float32Array) {
    if (!this.ctx) return;

    const now = this.ctx.currentTime;

    // Red de seguridad. Lo normal es que aquí no se entre nunca: `enqueue`
    // detecta la cola seca antes y rehace el colchón. Si aun así se llega
    // tarde —el reloj del contexto se ha ido de golpe—, se arranca con el
    // mismo margen que usa el colchón en vez de a pelo.
    if (this.nextTime < now) {
      this.nextTime = now + this.INITIAL_DELAY_S;
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
  // aritmética y así se pueden probar sin levantar un AudioContext.

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
    this.muestrasEnColchon = 0;
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

// A mano en la consola de la app: `audioPlayer.diagnostico()` durante una
// llamada dice si la cola se está quedando seca y cuánta reserva queda. Es lo
// que hay que mirar cuando a Perseo se le oye entrecortado.
if (typeof window !== 'undefined') {
  (window as unknown as { audioPlayer: AudioPlayer }).audioPlayer = audioPlayer;
}
