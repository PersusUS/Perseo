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
    temporizador = window.setTimeout(volcar, VOLCADO_MS);
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
  vigilante = window.setInterval(() => {
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
    window.clearInterval(vigilante);
    vigilante = null;
  }
  apuntar(`--- llamada cerrada (${motivo}) ---`);
  void volcar();
}
