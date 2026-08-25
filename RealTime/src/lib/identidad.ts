/**
 * Quién habla y quién sale por la cámara: la parte de la llamada.
 *
 * Aquí NO se decide nada: esto es un cartero. Toma los mismos trozos que ya
 * viajan a Gemini —PCM del micrófono y JPEG de la cámara—, los manda al núcleo
 * por los comandos `biometria_*` de Rust, y reparte las etiquetas que vuelven
 * («Javi», «Desconocido 1»…) a quien quiera pintarlas. Los vectores, los
 * umbrales y los perfiles viven en perseo_core/biometria.py.
 *
 * Dos detalles que importan:
 *
 *  1. **El VAD es de energía y basta.** No hace falta separar voz de ruido con
 *     precisión: solo decidir qué trozos merecen gastar una consulta. El suelo
 *     de ruido es adaptativo (media lenta del RMS), así que un micrófono ruidoso
 *     no lo deja todo en «habla» ni uno limpio se queda sordo.
 *
 *  2. **Una sola petición en vuelo por canal.** El núcleo tarda decenas de ms
 *     por trozo; si llega otro mientras, se acumula para la siguiente ronda.
 *     Sin esto, un hablante nervioso llenaría la cola más rápido de lo que se
 *     vacía y la etiqueta llegaría siempre tarde.
 */
import { invoke } from '@tauri-apps/api/core';

/** Una cara vista en el último fotograma analizado. Caja en píxeles sobre 640x480. */
export interface CaraDetectada {
  nombre: string | null;
  caja: [number, number, number, number];
  confianza: number;
  aprendiendo?: { etiqueta: string; peso: number; objetivo: number };
  aprendido?: boolean;
}

export interface Progreso {
  etiqueta: string;
  peso: number;
  objetivo: number;
}

interface RespuestaVoz {
  nombre?: string | null;
  confianza?: number;
  aprendiendo?: Progreso | null;
  aprendido?: boolean;
  error?: string;
}

interface EstadoBiometriaInterno {
  perfiles: { nombre: string; voz: boolean; caras: number; muestras: number; creado: string }[];
  aprendiendo: { voz: Progreso | null; cara: Progreso | null };
  disponibilidad: { voz: boolean; motivo_voz?: string; cara: boolean; motivo_cara?: string };
}

const MUESTRAS_SEGUNDO = 16000;
/** Trozo cómodo para ECAPA: dos segundos de voz seguida. */
const OBJETIVO_MUESTRAS = MUESTRAS_SEGUNDO * 2;
/** Menos de esto no vale mandarlo: vector inestable y CPU tirada. */
const MINIMO_MUESTRAS = MUESTRAS_SEGUNDO / 4;
/** Tope duro del búfer: si algo va mal, se descarta antes que crecer sin fin. */
const TOPE_MUESTRAS = OBJETIVO_MUESTRAS * 3;
/** Cuánto sigue valiendo la última etiqueta tras callarse (ms). */
const CADUCIDAD_HABLANTE_MS = 4500;
/** Cada cuánto se analiza un fotograma de cámara (ms). */
const CADA_CARA_MS = 4000;
/**
 * Lo mismo, mientras hay alguien a medio aprender.
 *
 * Fijar una cara desconocida cuesta ocho detecciones buenas, y a una cada
 * cuatro segundos eso son más de treinta segundos con esa persona delante:
 * media conversación tratándola de «Desconocido». Con el ritmo corto baja a la
 * mitad. Solo se acelera cuando hay algo que aprender, así que en la llamada
 * normal —el señor Persus solo delante de la cámara— no cambia nada.
 */
const CADA_CARA_APRENDIENDO_MS = 2000;
/** Cuánto sigue valiendo la última lectura de caras (ms). */
const CADUCIDAD_CARAS_MS = 7000;

class VigilanteIdentidad {
  activa = false;
  /** Última etiqueta de voz vigente, o null si se calló o caducó. */
  hablanteActual: string | null = null;
  /** Última lectura de caras vigente. */
  carasActuales: CaraDetectada[] = [];

  onHablante: (nombre: string | null) => void = () => {};
  onCaras: (caras: CaraDetectada[]) => void = () => {};

  private colaVoz: Int16Array[] = [];
  private muestrasAcumuladas = 0;
  private sueloRuido = 400;
  private enVueloVoz = false;
  private enVueloCara = false;
  /** Si la última lectura traía alguien a medio aprender, se mira más a menudo. */
  private aprendiendoCara = false;
  private ultimoResultadoVozMs = 0;
  private ultimoEnvioCaraMs = 0;
  private fotogramaPendiente: string | null = null;
  private avisoErrorDado = false;
  private temporizador: number | null = null;

  activar(): void {
    if (this.activa) return;
    this.activa = true;
    this.avisoErrorDado = false;
    this.temporizador = window.setInterval(() => this.latido(), 500);
  }

  desactivar(): void {
    if (!this.activa && this.temporizador === null) return;
    this.activa = false;
    if (this.temporizador !== null) {
      window.clearInterval(this.temporizador);
      this.temporizador = null;
    }
    this.colaVoz = [];
    this.muestrasAcumuladas = 0;
    this.fotogramaPendiente = null;
    this.carasActuales = [];
    this.aprendiendoCara = false;
    if (this.hablanteActual !== null) {
      this.hablanteActual = null;
      this.onHablante(null);
    }
    this.onCaras([]);
  }

  /** AudioManager lo llama con cada trozo del worklet (~8 ms de PCM Int16). */
  consumirAudio(trozo: ArrayBuffer): void {
    if (!this.activa) return;
    const muestras = new Int16Array(trozo);
    const rms = energiaRms(muestras);
    // Suelo lento: sube despacio con el ruido de fondo y baja igual. El umbral
    // de habla flota tres veces por encima, con un mínimo absoluto por si el
    // silencio fuera total.
    this.sueloRuido = this.sueloRuido * 0.95 + rms * 0.05;
    const umbral = Math.max(500, this.sueloRuido * 3);

    if (rms < umbral) {
      // Fin de frase probable: si lo acumulado ya sirve, vuela. Si era un clic
      // de boca, se descarta solo al caer el mínimo.
      if (this.muestrasAcumuladas >= MINIMO_MUESTRAS) void this.enviarVoz();
      return;
    }

    this.colaVoz.push(muestras);
    this.muestrasAcumuladas += muestras.length;
    if (this.muestrasAcumuladas >= TOPE_MUESTRAS) {
      void this.enviarVoz();
    } else if (this.muestrasAcumuladas >= OBJETIVO_MUESTRAS) {
      void this.enviarVoz();
    }
  }

  /** CameraManager lo llama con cada JPEG capturado; aquí solo se guarda el último. */
  consumirFotograma(base64: string): void {
    if (!this.activa) return;
    this.fotogramaPendiente = base64;
  }

  private latido(): void {
    const ahora = Date.now();
    if (
      this.hablanteActual !== null &&
      ahora - this.ultimoResultadoVozMs > CADUCIDAD_HABLANTE_MS
    ) {
      this.hablanteActual = null;
      this.onHablante(null);
    }
    if (
      this.carasActuales.length > 0 &&
      ahora - this.ultimoEnvioCaraMs > CADUCIDAD_CARAS_MS
    ) {
      this.carasActuales = [];
      this.onCaras([]);
    }
    // Cara: una foto cada tanto, si la cámara dejó alguna pendiente.
    const cadaCuanto = this.aprendiendoCara ? CADA_CARA_APRENDIENDO_MS : CADA_CARA_MS;
    if (this.fotogramaPendiente && ahora - this.ultimoEnvioCaraMs >= cadaCuanto && !this.enVueloCara) {
      void this.enviarCara(this.fotogramaPendiente);
      this.fotogramaPendiente = null;
    }
  }

  private async enviarVoz(): Promise<void> {
    if (this.enVueloVoz || !this.activa) return;
    const trozos = this.colaVoz;
    const total = this.muestrasAcumuladas;
    this.colaVoz = [];
    this.muestrasAcumuladas = 0;
    if (total < MINIMO_MUESTRAS || trozos.length === 0) return;

    this.enVueloVoz = true;
    try {
      const fusionado = new Int16Array(total);
      let cursor = 0;
      for (const trozo of trozos) {
        fusionado.set(trozo, cursor);
        cursor += trozo.length;
      }
      const respuesta = await invoke<RespuestaVoz>('biometria_voz', {
        audio: base64DeInt16(fusionado),
      });
      this.avisoErrorDado = false;
      this.ultimoResultadoVozMs = Date.now();
      if (respuesta.nombre && respuesta.nombre !== this.hablanteActual) {
        this.hablanteActual = respuesta.nombre;
        this.onHablante(respuesta.nombre);
      } else if (respuesta.nombre) {
        // Mismo hablante: refresca la caducidad sin repintar.
        this.hablanteActual = respuesta.nombre;
      }
    } catch (e) {
      if (!this.avisoErrorDado) {
        console.warn('[Identidad] El núcleo no respondió:', e);
        this.avisoErrorDado = true;
      }
    } finally {
      this.enVueloVoz = false;
    }
  }

  private async enviarCara(base64: string): Promise<void> {
    this.enVueloCara = true;
    this.ultimoEnvioCaraMs = Date.now();
    try {
      const respuesta = await invoke<{ caras?: CaraDetectada[]; error?: string }>(
        'biometria_cara',
        { imagen: base64 },
      );
      if (respuesta.error) {
        console.warn('[Identidad] Caras:', respuesta.error);
        return;
      }
      this.carasActuales = respuesta.caras ?? [];
      this.aprendiendoCara = this.carasActuales.some(c => c.aprendiendo);
      this.onCaras(this.carasActuales);
    } catch (e) {
      console.warn('[Identidad] No se pudo analizar la imagen:', e);
    } finally {
      this.enVueloCara = false;
    }
  }
}

function energiaRms(muestras: Int16Array): number {
  let suma = 0;
  for (let i = 0; i < muestras.length; i++) {
    suma += muestras[i] * muestras[i];
  }
  return Math.sqrt(suma / Math.max(1, muestras.length));
}

/** Int16 → PCM little-endian → base64. Mismo formato que manda AudioManager. */
function base64DeInt16(muestras: Int16Array): string {
  const bytes = new Uint8Array(muestras.buffer, muestras.byteOffset, muestras.byteLength);
  let binaria = '';
  const PASO = 8192;
  for (let i = 0; i < bytes.length; i += PASO) {
    binaria += String.fromCharCode(...bytes.subarray(i, i + PASO));
  }
  return btoa(binaria);
}

// --------------------------------------------------------------------------- //
// Gestión manual (Ajustes)
// --------------------------------------------------------------------------- //

export type EstadoBiometria = EstadoBiometriaInterno;

export async function estadoBiometria(): Promise<EstadoBiometria> {
  return invoke<EstadoBiometria>('biometria_estado');
}

/**
 * Graba `segundos` del micrófono y devuelve la muestra como PCM 16k mono en
 * base64, lista para `crearPerfil`. Va aparte del AudioManager a propósito:
 * desde Ajustes puede querer grabarse SIN estar en llamada, y ahí el AudioManager
 * está parado. Se usa ScriptProcessorNode y no un AudioWorklet porque el módulo
 * del worklet vive en un fichero servido por Vite y esto tiene que funcionar
 * también sin build.
 */
export async function grabarMuestra(segundos = 6): Promise<string> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });
  const contexto = new AudioContext({ sampleRate: 16000 });
  const fuente = contexto.createMediaStreamSource(stream);
  const procesador = contexto.createScriptProcessor(4096, 1, 1);
  const trozos: Float32Array[] = [];

  return await new Promise<string>((resolver, rechazar) => {
    procesador.onaudioprocess = (evento) => {
      trozos.push(new Float32Array(evento.inputBuffer.getChannelData(0)));
    };
    fuente.connect(procesador);
    // El silencioso destino es obligatorio: sin él Chrome/WebView2 no arranca
    // el grafo y onaudioprocess nunca suena.
    procesador.connect(contexto.destination);

    // Si en el doble del plazo no hay muestras suficientes, algo se rompió
    // (permiso retirado a mitad, dispositivo desconectado): mejor un fallo
    // claro que seis segundos de vacío convertidos en perfil basura.
    window.setTimeout(() => {
      limpiar();
      const esperadas = contexto.sampleRate * segundos;
      if (trozos.reduce((n, t) => n + t.length, 0) < esperadas / 2) {
        rechazar(new Error('El micrófono no entregó audio suficiente'));
        return;
      }
      resolver(base64DeInt16(aPcm16(trozos)));
    }, segundos * 1000);
  });

  function limpiar(): void {
    try {
      fuente.disconnect();
      procesador.disconnect();
      stream.getTracks().forEach((t) => t.stop());
      void contexto.close();
    } catch {
      /* ya estaba muerto: da igual */
    }
  }
}

/**
 * Abre la cámara y devuelve un fotograma JPEG en base64, listo para
 * `crearPerfil`. Va aparte de CameraManager por el mismo motivo que
 * `grabarMuestra` va aparte del AudioManager: desde Ajustes se quiere capturar
 * SIN estar en llamada, con el manager parado. Se dejan dos segundos de
 * margen antes del disparo para que la cámara ajuste exposición y enfoque;
 * sin esa espera sale una foto oscura o desenfocada.
 */
export async function capturarCara(): Promise<string> {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480, facingMode: 'user' },
  });
  try {
    const video = document.createElement('video');
    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;
    await video.play();
    await new Promise((resolver) => window.setTimeout(resolver, 2000));

    const lienzo = document.createElement('canvas');
    lienzo.width = video.videoWidth || 640;
    lienzo.height = video.videoHeight || 480;
    const ctx = lienzo.getContext('2d');
    if (!ctx) throw new Error('Sin contexto 2D para el fotograma');
    ctx.drawImage(video, 0, 0, lienzo.width, lienzo.height);
    const dataUrl = lienzo.toDataURL('image/jpeg', 0.7);
    return dataUrl.split(',')[1];
  } finally {
    stream.getTracks().forEach((t) => t.stop());
  }
}

/** Float32 [-1..1] → Int16 PCM. */
function aPcm16(trozos: Float32Array[]): Int16Array {
  const total = trozos.reduce((n, t) => n + t.length, 0);
  const pcm = new Int16Array(total);
  let cursor = 0;
  for (const trozo of trozos) {
    for (let i = 0; i < trozo.length; i++) {
      const v = Math.max(-1, Math.min(1, trozo[i]));
      pcm[cursor++] = v < 0 ? v * 32768 : v * 32767;
    }
  }
  return pcm;
}

export async function crearPerfil(
  nombre: string,
  audio?: string,
  imagen?: string,
): Promise<{ ok?: boolean; error?: string; añadido?: string[] }> {
  return invoke('biometria_enrolar', { nombre, audio, imagen });
}

export async function renombrarPerfil(
  nombre: string,
  nuevoNombre: string,
): Promise<{ ok?: boolean; error?: string }> {
  return invoke('biometria_renombrar', { nombre, nuevoNombre });
}

export async function borrarPerfil(nombre: string): Promise<{ ok?: boolean; error?: string }> {
  return invoke('biometria_borrar', { nombre });
}

export const vigilante = new VigilanteIdentidad();
