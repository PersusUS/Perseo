/**
 * El cuaderno de la llamada: por qué se oyó entrecortado.
 *
 * La ventana de release no tiene consola a mano, así que un `console.warn` es
 * un aviso que nadie lee nunca. Aquí las líneas se juntan un segundo y se
 * escriben en `perseo_core/datos/llamada.log` (comando `anotar_diagnostico`,
 * en `src-tauri/src/commands.rs`), que se puede leer desde fuera mientras la
 * llamada sigue abierta.
 *
 * Lo que se apunta es lo que separa las tres causas posibles de un corte:
 *
 * - `audio: cola seca` — la reproducción se quedó sin reserva. Red lenta, o el
 *   hilo principal ocupado.
 * - `enlace: interrumpido` — el servidor cortó a Perseo a mitad de frase
 *   porque la detección automática de voz oyó algo. Con ruido en la sala, esto
 *   se oye exactamente igual que un fallo de red y no lo es.
 * - `bloqueo` — el hilo principal se ha ido más de 300 ms. Ahí no hay red que
 *   valga: la ventana estaba haciendo otra cosa (una captura de pantalla, por
 *   ejemplo) y el audio programado se quedó sin quien lo alimentara.
 *
 * Escribir NO puede costar lo que se está midiendo: por eso se acumula en una
 * lista y se vuelca una vez por segundo, y nunca desde el camino del audio.
 */

import { invoke } from '@tauri-apps/api/core';

const cola: string[] = [];
let temporizador: number | null = null;
let vigilante: number | null = null;
let ultimaVuelta = 0;

/** Cada cuánto se vacía la cola al fichero. */
const VOLCADO_MS = 1000;
/** Ritmo del vigilante del hilo principal. */
const VIGILANCIA_MS = 250;
/** A partir de cuánto retraso se considera que el hilo se fue a otra cosa. */
const BLOQUEO_MS = 300;

/**
 * `setTimeout` y compañía sin pasar por `window`.
 *
 * El cuaderno lo importa media aplicación, y media aplicación se prueba en
 * Node, donde no hay `window`: con `window.setTimeout` escrito a pelo, apuntar
 * una línea dentro de una prueba reventaba con «window is not defined» — un
 * módulo de diagnóstico tirando la prueba de lo que diagnostica.
 */
const relojes = globalThis as unknown as {
  setTimeout: (fn: () => void, ms: number) => number;
  setInterval: (fn: () => void, ms: number) => number;
  clearInterval: (id: number) => void;
};

function marcaDeTiempo(): string {
  return new Date().toISOString().slice(11, 23);
}

async function volcar(): Promise<void> {
  temporizador = null;
  if (!cola.length) return;
  const lineas = cola.splice(0, cola.length);
  try {
    await invoke('anotar_diagnostico', { lineas });
  } catch {
    // Sin Tauri detrás —el navegador de las maquetas— no hay fichero que
    // escribir. La consola ya lo ha dicho; no hay nada más que hacer.
  }
}

/** Apunta una línea en el cuaderno. Barata a propósito: no toca el disco. */
export function apuntar(linea: string): void {
  const texto = `${marcaDeTiempo()} ${linea}`;
  console.log(`[diag] ${texto}`);
  cola.push(texto);
  if (temporizador === null) {
    temporizador = relojes.setTimeout(volcar, VOLCADO_MS);
  }
}

/**
 * Empieza a vigilar el hilo principal.
 *
 * El vigilante es un reloj que debería sonar cada 250 ms: si suena mucho más
 * tarde, es que el hilo estuvo ocupado justo ese rato — y ese rato es el que
 * dejó al audio sin alimentar.
 */
export function iniciarDiagnostico(motivo: string): void {
  apuntar(`--- llamada abierta (${motivo}) ---`);
  if (vigilante !== null) return;
  ultimaVuelta = performance.now();
  vigilante = relojes.setInterval(() => {
    const ahora = performance.now();
    const retraso = ahora - ultimaVuelta - VIGILANCIA_MS;
    ultimaVuelta = ahora;
    if (retraso > BLOQUEO_MS) {
      apuntar(`bloqueo: el hilo principal se fue ${Math.round(retraso)} ms`);
    }
  }, VIGILANCIA_MS);
}

/** Cierra el cuaderno y vuelca lo que quede. */
export function pararDiagnostico(motivo: string): void {
  if (vigilante !== null) {
    relojes.clearInterval(vigilante);
    vigilante = null;
  }
  apuntar(`--- llamada cerrada (${motivo}) ---`);
  void volcar();
}

// --------------------------------------------------------------------------- #
// Cuánto tarda en contestar
// --------------------------------------------------------------------------- #

/**
 * La cifra que faltaba: desde que el señor Persus deja de hablar hasta que se
 * oye la primera sílaba de Perseo.
 *
 * Sin ella, tocar la detección de voz (`silenceDurationMs` en gemini-live.ts)
 * era a ojo: se bajaba, parecía más rápido, y nadie podía decir si el precio
 * era cortar frases. Se mide con lo que ya llega por el socket —el último
 * trozo de transcripción de entrada como final del habla, el primer trozo de
 * audio del modelo como principio de la respuesta—, así que no cuesta nada.
 *
 * No es la latencia de red: dentro van el silencio que espera el detector, el
 * modelo pensando y el colchón del reproductor. Es justo la espera que se vive.
 */
const latencias: number[] = [];
/** Tope de muestras guardadas: una llamada larga no debe crecer sin fin. */
const TOPE_LATENCIAS = 200;

let finDelHablaMs = 0;
let esperandoRespuesta = false;

/** El usuario sigue hablando: la última vez que se le oyó es esta. */
export function hablaElUsuario(): void {
  finDelHablaMs = performance.now();
  esperandoRespuesta = true;
}

/** Primer audio de Perseo tras ese silencio: ahí se cierra la medición. */
export function respondePerseo(): void {
  if (!esperandoRespuesta || finDelHablaMs === 0) return;
  esperandoRespuesta = false;
  const tardanza = Math.round(performance.now() - finDelHablaMs);
  latencias.push(tardanza);
  if (latencias.length > TOPE_LATENCIAS) latencias.shift();
  apuntar(`respuesta: ${tardanza} ms desde que dejó de hablar`);
}

/** Lo que se ha medido en esta llamada. Se lee desde la consola o al colgar. */
export function latenciaDeRespuesta(): {
  veces: number;
  ultima: number;
  mediana: number;
  peor: number;
} {
  if (!latencias.length) return { veces: 0, ultima: 0, mediana: 0, peor: 0 };
  const ordenadas = [...latencias].sort((a, b) => a - b);
  return {
    veces: latencias.length,
    ultima: latencias[latencias.length - 1],
    mediana: ordenadas[Math.floor(ordenadas.length / 2)],
    peor: ordenadas[ordenadas.length - 1],
  };
}

/** Empezar de cero. Lo llama el arranque de cada llamada. */
export function olvidarLatencias(): void {
  latencias.length = 0;
  finDelHablaMs = 0;
  esperandoRespuesta = false;
}
