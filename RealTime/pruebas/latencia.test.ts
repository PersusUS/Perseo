/**
 * La cifra que dice si Perseo tarda: desde que dejas de hablar hasta que se le
 * oye. Es lo que convierte el ajuste del silencio del micrófono en una decisión
 * con un número delante en vez de una impresión.
 *
 * Se prueba la contabilidad, no el reloj: que una respuesta se mida una sola
 * vez, que el audio que llega sin que nadie haya hablado no cuente, y que la
 * mediana no se la lleve un pico suelto.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';

import {
  hablaElUsuario,
  latenciaDeRespuesta,
  olvidarLatencias,
  respondePerseo,
} from '../src/lib/llamada/diagnostico';

/** Una respuesta entera: habla, pasan `ms`, contesta. */
function unaRespuesta(ms: number, reloj: { ahora: number }): void {
  hablaElUsuario();
  reloj.ahora += ms;
  respondePerseo();
}

describe('latencia de respuesta', () => {
  const reloj = { ahora: 0 };

  beforeEach(() => {
    reloj.ahora = 1000;
    vi.spyOn(performance, 'now').mockImplementation(() => reloj.ahora);
    olvidarLatencias();
  });

  it('sin medidas no inventa ninguna', () => {
    expect(latenciaDeRespuesta()).toEqual({ veces: 0, ultima: 0, mediana: 0, peor: 0 });
  });

  it('mide lo que se tarda en contestar', () => {
    unaRespuesta(820, reloj);
    expect(latenciaDeRespuesta()).toMatchObject({ veces: 1, ultima: 820, peor: 820 });
  });

  it('el audio que sigue llegando no cuenta como otra respuesta', () => {
    // Una respuesta son muchos trozos de audio seguidos; solo el primero
    // cierra la medición. Sin esto, cada trozo apuntaría una latencia mayor y
    // la mediana diría que Perseo tarda segundos.
    unaRespuesta(500, reloj);
    reloj.ahora += 3000;
    respondePerseo();
    respondePerseo();
    expect(latenciaDeRespuesta().veces).toBe(1);
  });

  it('un pico suelto no se lleva la mediana', () => {
    unaRespuesta(600, reloj);
    unaRespuesta(650, reloj);
    unaRespuesta(9000, reloj);
    const medido = latenciaDeRespuesta();
    expect(medido.peor).toBe(9000);
    expect(medido.mediana).toBe(650);
  });

  it('olvidar deja el cuaderno limpio para la llamada siguiente', () => {
    unaRespuesta(700, reloj);
    olvidarLatencias();
    expect(latenciaDeRespuesta().veces).toBe(0);
  });
});
