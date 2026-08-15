/**
 * Las dos piezas puras del audio de salida.
 *
 * Son aritmética sobre bytes, así que un fallo aquí no revienta: suena mal, o
 * suena a medias, y cuesta relacionarlo con el código. Por eso están probadas —
 * es lo último que le faltaba a H-28.
 */

import { describe, expect, it } from 'vitest';

import { decodePCM, mergeChunks } from '../src/lib/audio-pcm';

/** Arma el base64 de un PCM de 16 bits con signo, que es lo que manda Gemini. */
function pcmBase64(muestras: number[]): string {
  const int16 = Int16Array.from(muestras);
  const bytes = new Uint8Array(int16.buffer);
  let binario = '';
  for (const b of bytes) binario += String.fromCharCode(b);
  return btoa(binario);
}

describe('decodePCM', () => {
  it('devuelve una muestra por cada entero de 16 bits', () => {
    const pcm = decodePCM(pcmBase64([0, 1, 2, 3]));
    expect(pcm).toBeInstanceOf(Float32Array);
    expect(pcm.length).toBe(4);
  });

  it('deja el silencio en cero', () => {
    expect(Array.from(decodePCM(pcmBase64([0, 0])))).toEqual([0, 0]);
  });

  it('lleva el mínimo exactamente a -1', () => {
    // Por esto el divisor es 32768 y no 32767: con 32767 el negativo más grave
    // se pasaría de -1 y algunos motores lo recortan.
    expect(decodePCM(pcmBase64([-32768]))[0]).toBe(-1);
  });

  it('deja el máximo justo por debajo de 1', () => {
    const valor = decodePCM(pcmBase64([32767]))[0];
    expect(valor).toBeLessThan(1);
    expect(valor).toBeGreaterThan(0.9999);
  });

  it('conserva el signo y la proporción', () => {
    const pcm = decodePCM(pcmBase64([16384, -16384]));
    expect(pcm[0]).toBeCloseTo(0.5, 6);
    expect(pcm[1]).toBeCloseTo(-0.5, 6);
  });

  it('no se inventa muestras con una entrada vacía', () => {
    expect(decodePCM('').length).toBe(0);
  });

  it('lee los bytes en little endian, como los manda el modelo', () => {
    // 0x0100 en little endian son los bytes 00 01, que valen 256.
    const bytes = new Uint8Array([0x00, 0x01]);
    let binario = '';
    for (const b of bytes) binario += String.fromCharCode(b);
    expect(decodePCM(btoa(binario))[0]).toBeCloseTo(256 / 32768, 8);
  });
});

describe('mergeChunks', () => {
  it('pega los trozos en orden', () => {
    const juntos = mergeChunks([
      Float32Array.from([1, 2]),
      Float32Array.from([3]),
      Float32Array.from([4, 5]),
    ]);
    expect(Array.from(juntos)).toEqual([1, 2, 3, 4, 5]);
  });

  it('suma las longitudes', () => {
    const juntos = mergeChunks([new Float32Array(10), new Float32Array(7)]);
    expect(juntos.length).toBe(17);
  });

  it('con un solo trozo devuelve lo mismo', () => {
    expect(Array.from(mergeChunks([Float32Array.from([0.5])]))).toEqual([0.5]);
  });

  it('sin trozos devuelve algo vacío, no nulo', () => {
    const juntos = mergeChunks([]);
    expect(juntos).toBeInstanceOf(Float32Array);
    expect(juntos.length).toBe(0);
  });

  it('los trozos vacíos no desplazan a los demás', () => {
    const juntos = mergeChunks([
      new Float32Array(0),
      Float32Array.from([1]),
      new Float32Array(0),
      Float32Array.from([2]),
    ]);
    expect(Array.from(juntos)).toEqual([1, 2]);
  });

  it('no toca los trozos originales', () => {
    const primero = Float32Array.from([1, 2]);
    mergeChunks([primero, Float32Array.from([3])]);
    expect(Array.from(primero)).toEqual([1, 2]);
  });
});

describe('los dos juntos', () => {
  it('un flujo troceado suena igual que entero', () => {
    // Es lo que hace el reproductor de verdad: decodifica cada trozo que llega y
    // los pega antes de programarlos.
    const entero = decodePCM(pcmBase64([1000, -1000, 2000, -2000]));
    const troceado = mergeChunks([
      decodePCM(pcmBase64([1000, -1000])),
      decodePCM(pcmBase64([2000, -2000])),
    ]);
    expect(Array.from(troceado)).toEqual(Array.from(entero));
  });
});
