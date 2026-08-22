/**
 * El reloj de la sesión (T-11).
 *
 * Es la única pieza pura de la escenografía del modo live, y sale en las tres
 * pantallas a la vez —la lectura de la esquina, la cinta y la cartela—, así que
 * un fallo aquí se ve tres veces. Lo demás de ese fichero es dibujo.
 */

import { describe, expect, it } from 'vitest';

import { comoReloj } from '../src/components/Escenografia';

describe('comoReloj', () => {
  it('rellena con ceros hasta las horas', () => {
    expect(comoReloj(0)).toBe('00:00:00');
    expect(comoReloj(9)).toBe('00:00:09');
    expect(comoReloj(75)).toBe('00:01:15');
    expect(comoReloj(3661)).toBe('01:01:01');
  });

  it('sigue contando pasadas las 24 h en vez de dar la vuelta', () => {
    // Una llamada tan larga no va a pasar, pero un contador que vuelve a cero
    // es peor que uno feo: parece que la sesión se ha reiniciado.
    expect(comoReloj(90000)).toBe('25:00:00');
  });

  it('no se rompe con segundos rotos ni negativos', () => {
    // El cronómetro sale de una resta de relojes: si la máquina se duerme y
    // despierta, el número puede llegar con decimales o en negativo.
    expect(comoReloj(12.7)).toBe('00:00:12');
    expect(comoReloj(-5)).toBe('00:00:00');
  });
});
