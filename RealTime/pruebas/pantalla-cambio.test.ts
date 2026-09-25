/**
 * La pantalla solo viaja cuando cambia. Ver `lib/llamada/pantalla-cambio.ts`.
 *
 * Antes: una captura cada dos segundos aunque fuera la misma — 1.062 en 25
 * llamadas, casi todas iguales a la anterior.
 */
import { describe, expect, it } from 'vitest';

import {
  ALTO,
  ANCHO,
  CambioDePantalla,
  REENVIO_MAXIMO_MS,
  UMBRAL_CELDA,
  celdasCambiadas,
} from '../src/lib/llamada/pantalla-cambio';

const lisa = (gris = 128) => new Uint8Array(ANCHO * ALTO).fill(gris);

function conCeldas(base: Uint8Array, celdas: number[], delta: number): Uint8Array {
  const copia = new Uint8Array(base);
  for (const i of celdas) copia[i] = Math.min(255, copia[i] + delta);
  return copia;
}

describe('celdasCambiadas', () => {
  it('una línea de texto nueva cambia varias celdas de golpe', () => {
    const antes = lisa();
    const despues = conCeldas(antes, [100, 101, 102, 103, 104, 105], 40);
    expect(celdasCambiadas(antes, despues)).toBe(6);
  });

  it('el cursor que parpadea o el reloj no llegan al umbral', () => {
    const antes = lisa();
    expect(celdasCambiadas(antes, conCeldas(antes, [575], UMBRAL_CELDA))).toBe(0);
  });

  it('miniaturas de tamaños distintos cuentan como cambio', () => {
    expect(celdasCambiadas(lisa(), new Uint8Array(10))).toBeGreaterThan(0);
  });
});

describe('CambioDePantalla', () => {
  it('la primera captura se manda siempre', () => {
    expect(new CambioDePantalla().merece(lisa(), 0)).toBe(true);
  });

  it('la misma pantalla no se vuelve a mandar', () => {
    const cambio = new CambioDePantalla();
    cambio.merece(lisa(), 0);
    expect(cambio.merece(lisa(), 2000)).toBe(false);
    expect(cambio.ahorradas).toBe(1);
  });

  it('un cambio de verdad se manda', () => {
    const cambio = new CambioDePantalla();
    cambio.merece(lisa(), 0);
    expect(cambio.merece(conCeldas(lisa(), [5, 6], 60), 2000)).toBe(true);
  });

  it('se compara con la última MANDADA: un cambio lento acaba notándose', () => {
    const cambio = new CambioDePantalla();
    cambio.merece(lisa(128), 0);
    // Seis pasos de 4 niveles: ninguno llega al umbral respecto al anterior,
    // pero el acumulado sí respecto a la que se mandó.
    const resultados = [1, 2, 3, 4, 5, 6].map(n => cambio.merece(lisa(128 + 4 * n), n * 2000));
    expect(resultados.slice(0, 2)).toEqual([false, false]);
    expect(resultados).toContain(true);
  });

  it('aunque nada cambie, se reenvía cada tanto', () => {
    const cambio = new CambioDePantalla();
    cambio.merece(lisa(), 0);
    expect(cambio.merece(lisa(), REENVIO_MAXIMO_MS - 1)).toBe(false);
    expect(cambio.merece(lisa(), REENVIO_MAXIMO_MS)).toBe(true);
  });

  it('sin miniatura se manda, como antes', () => {
    const cambio = new CambioDePantalla();
    cambio.merece(lisa(), 0);
    expect(cambio.merece(null, 1)).toBe(true);
  });

  it('olvidar hace que la siguiente llamada empiece mandando', () => {
    const cambio = new CambioDePantalla();
    cambio.merece(lisa(), 0);
    cambio.olvidar();
    expect(cambio.merece(lisa(), 1)).toBe(true);
    expect(cambio.mandadas).toBe(1);
  });
});
