/**
 * Quién habla y quién sale por la cámara: la parte de la llamada.
 *
 * Aquí NO se decide nada: esto es un cartero. Toma los mismos trozos que ya
 * viajan a Gemini —PCM del micrófono y JPEG de la cámara—, los manda al núcleo
 * por los comandos `biometria_*` de Rust, y reparte las etiquetas que vuelven
 * («Javi», «Desconocido 1»…) a quien quiera pintarlas. Los vectores, los
 * umbrales y los perfiles viven en perseo_core/servicios/biometria.py.
 *
 * Tres detalles que importan:
 *
 *  1. **Aquí no se decide qué es voz.** Se mandan ventanas seguidas de
 *     micrófono (`ventana-voz.ts`) y el núcleo les quita los silencios. Hasta
 *     el 2026-09-23 el recorte se hacía aquí, y mal: solo llegaban picos de
 *     sílaba pegados, demasiado cortos para reconocer a nadie.
 *
 *  2. **Mientras habla Perseo no se escucha.** Lo que sale por el altavoz entra
 *     por el micrófono aunque el navegador cancele el eco, y mandarlo era
 *     enseñarle al núcleo la voz de Perseo como la de un «Desconocido». Se
 *     calla mientras suena y 300 ms después, que es lo que tarda la cola del
 *     eco en apagarse.
 *
 *  3. **Una sola petición en vuelo por canal.** El núcleo tarda decenas de ms
 *     por ventana; si se cierra otra mientras, espera su turno (solo la última:
 *     una etiqueta vieja no le sirve a nadie).
 */
import { invoke } from '@tauri-apps/api/core';

import { audioPlayer } from '../audio/audio-player';
import { VentanaVoz } from './ventana-voz';

/** Una cara vista en el último fotograma analizado. Caja en píxeles sobre 640x480. */
export interface CaraDetectada {
  nombre: string | null;
  caja: [number, number, number, number];
  confianza: number;
  aprendiendo?: { etiqueta: string; peso: number; objetivo: number };
  aprendido?: boolean;
}

interface Progreso {
  etiqueta: string;
  peso: number;
  objetivo: number;
}

interface RespuestaVoz {
  nombre?: string | null;
  confianza?: number;
  aprendiendo?: Progreso | null;
  aprendido?: boolean;
  /** Se parece a alguien, pero no lo bastante: no se nombra a nadie. */
  dudoso?: string;
  /** La voz dudaba y la cámara lo confirmó. */
  por?: 'voz+cara';
  error?: string;
}

interface EstadoBiometriaInterno {
  perfiles: {
    nombre: string;
    voz: boolean;
    voces?: number;
    /** Perfil de voz de antes del 2026-09-23, hecho con el refuerzo roto. */
    voz_antigua?: boolean;
    caras: number;
    muestras: number;
    creado: string;
  }[];
  aprendiendo: { voz: Progreso | null; cara: Progreso | null };
  disponibilidad: { voz: boolean; motivo_voz?: string; cara: boolean; motivo_cara?: string };
}

/** Cuánto sigue valiendo la última etiqueta tras callarse (ms). Una ventana
 *  dura hasta tres segundos, así que con menos la etiqueta se apagaría entre
 *  una y la siguiente aunque siguiera hablando el mismo. */
const CADUCIDAD_HABLANTE_MS = 6000;
/** Lo que se sigue sin escuchar después de que Perseo se calle (ms). */
const COLA_ECO_MS = 300;
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
  /** Habla alguien y el núcleo no sabe quién: se parece a alguien, pero no lo bastante. */
  onDuda: () => void = () => {};
  onCaras: (caras: CaraDetectada[]) => void = () => {};
  /** ¿Está sonando Perseo? Se pregunta al reproductor, que es quien lo sabe. */
  hablaPerseo: () => boolean = () => audioPlayer.estaSonando();

  private ventana = new VentanaVoz();
  /** La ventana que se cerró con otra en vuelo: la siguiente en salir. */
  private ventanaEnEspera: Int16Array | null = null;
  private ecoHastaMs = 0;
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
    this.ventana.descartar();
    this.ventanaEnEspera = null;
    this.fotogramaPendiente = null;
    this.carasActuales = [];
    this.aprendiendoCara = false;
    if (this.hablanteActual !== null) {
      this.hablanteActual = null;
      this.onHablante(null);
    }
    this.onCaras([]);
  }

  /** AudioManager lo llama con cada trozo del worklet (64 ms de PCM Int16). */
  consumirAudio(trozo: ArrayBuffer): void {
    if (!this.activa) return;
    const ahora = Date.now();
    if (this.hablaPerseo()) this.ecoHastaMs = ahora + COLA_ECO_MS;
    if (ahora < this.ecoHastaMs) {
      // Lo que entra ahora es Perseo, o su eco: ni se manda ni se pega a lo
      // que diga el siguiente.
      this.ventana.descartar();
      return;
    }
    const lista = this.ventana.empujar(new Int16Array(trozo));
    if (lista) void this.enviarVoz(lista);
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

  private async enviarVoz(ventana: Int16Array): Promise<void> {
    if (!this.activa) return;
    if (this.enVueloVoz) {
      this.ventanaEnEspera = ventana;
      return;
    }

    this.enVueloVoz = true;
    try {
      const respuesta = await invoke<RespuestaVoz>('biometria_voz', {
        audio: base64DeInt16(ventana),
      });
      this.avisoErrorDado = false;
      this.ultimoResultadoVozMs = Date.now();
      if (respuesta.nombre && respuesta.nombre !== this.hablanteActual) {
        this.hablanteActual = respuesta.nombre;
        this.onHablante(respuesta.nombre);
      } else if (respuesta.nombre) {
        // Mismo hablante: refresca la caducidad sin repintar.
        this.hablanteActual = respuesta.nombre;
      } else if (respuesta.dudoso) {
        // Hablaba alguien y no se sabe quién: la etiqueta de antes ya no vale.
        // Seguir pintando «Persus» era dejar que el modelo lo siguiera creyendo.
        if (this.hablanteActual !== null) {
          this.hablanteActual = null;
          this.onHablante(null);
        }
        this.onDuda();
      }
    } catch (e) {
      if (!this.avisoErrorDado) {
        console.warn('[Identidad] El núcleo no respondió:', e);
        this.avisoErrorDado = true;
      }
    } finally {
      this.enVueloVoz = false;
      const siguiente = this.ventanaEnEspera;
      this.ventanaEnEspera = null;
      if (siguiente) void this.enviarVoz(siguiente);
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

/** Lo que contesta el núcleo a una toma de alta. */
interface ResultadoAlta {
  ok?: boolean;
  error?: string;
  añadido?: string[];
  /** Segundos de voz útil que encontró en la toma. */
  segundos_voz?: number;
  /** Cuánto se parece esta toma a las anteriores del mismo perfil. */
  parecido_voz?: number;
  voces?: number;
  caras?: number;
}

/**
 * Da de alta o refuerza un perfil con una toma.
 *
 * Una toma mala —poca voz, cara movida o de lado— el núcleo la rechaza con un
 * 400 **y el motivo**, que es lo único que permite repetirla bien. El puente de
 * Rust convierte ese 400 en una excepción con el texto dentro; sin capturarla
 * aquí, Ajustes no enseñaba nada y la toma parecía no haber pasado.
 */
export async function crearPerfil(
  nombre: string,
  audio?: string,
  imagen?: string,
): Promise<ResultadoAlta> {
  try {
    return await invoke<ResultadoAlta>('biometria_enrolar', { nombre, audio, imagen });
  } catch (e) {
    return { error: motivoDelNucleo(e) };
  }
}

/**
 * Lo que se le dice a quien acaba de grabar una toma: cuánto sirvió y si
 * conviene otra. Sin esto el alta era un botón que decía «guardado» igual con
 * seis segundos de voz que con uno, y nadie sabía si el perfil había quedado
 * bien. Con tres tomas —y una foto un poco girada— el reconocimiento va mejor.
 */
export function resumenDeToma(r: ResultadoAlta): string {
  if (r.añadido?.includes('voz')) {
    const partes = [`Toma guardada: ${(r.segundos_voz ?? 0).toFixed(1).replace('.', ',')} s de voz útil`];
    if (r.parecido_voz !== undefined) {
      const parecido = Math.round(r.parecido_voz * 100);
      partes.push(
        r.parecido_voz < 0.4
          ? `se parece poco a las anteriores (${parecido} %): ¿misma persona y mismo micrófono?`
          : `se parece a las anteriores un ${parecido} %`,
      );
    }
    const voces = r.voces ?? 1;
    partes.push(`${voces} muestra(s) de voz`);
    return partes.join(' · ') + (voces < 3 ? '. Graba otra toma para afinar: con tres va mejor.' : '.');
  }
  const caras = r.caras ?? 1;
  return `Foto guardada · ${caras} ángulo(s)` + (caras < 3 ? '. Otra con la cara un poco girada ayuda.' : '.');
}

/** «El nucleo respondio 400 Bad Request: Se oye muy poca voz…» → el motivo. */
export function motivoDelNucleo(e: unknown): string {
  return String(e).replace(/^El nucleo respondio \d{3}[^:]*:\s*/, '');
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
