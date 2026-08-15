/**
 * Las dos piezas puras de la reproducción de audio.
 *
 * Estaban dentro de `AudioPlayer` como métodos privados, y por eso no había
 * forma de probarlas sin levantar un `AudioContext`. Aquí no dependen de nada
 * del navegador: entran bytes y salen muestras.
 *
 * Sacarlas **no toca la lógica de reproducción**, que está en la lista de
 * intocables del roadmap: los dos `AudioContext` (16 y 24 kHz) siguen donde
 * estaban y siguen siendo dos. Lo único que se ha movido es la aritmética.
 */

/**
 * Convierte el audio que manda Gemini —PCM de 16 bits con signo, en base64— en
 * las muestras entre -1 y 1 que espera la Web Audio API.
 *
 * El divisor es 32768 y no 32767 a propósito: es el valor absoluto del mínimo de
 * un entero de 16 bits con signo, así que `-32768` cae exactamente en `-1` y
 * ningún pico se sale del rango. Con 32767, el negativo más grave se pasaría de
 * -1 y algunos motores lo recortan.
 */
export function decodePCM(base64: string): Float32Array {
  const raw = atob(base64);
  const len = raw.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) bytes[i] = raw.charCodeAt(i);

  const int16 = new Int16Array(bytes.buffer);
  const float = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) float[i] = int16[i] / 32768.0;
  return float;
}

/** Junta varios trozos en uno solo, en orden y sin copiar de más. */
export function mergeChunks(chunks: Float32Array[]): Float32Array {
  const totalLen = chunks.reduce((s, c) => s + c.length, 0);
  const merged = new Float32Array(totalLen);
  let offset = 0;
  for (const c of chunks) {
    merged.set(c, offset);
    offset += c.length;
  }
  return merged;
}
