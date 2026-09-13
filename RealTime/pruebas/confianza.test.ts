/**
 * La única regla del modo confianza: cuándo toca volver a pedirla.
 *
 * Vivía dentro de `App.tsx` y no la probaba nadie, que es lo que pasa con la
 * lógica que comparte fichero con una pantalla. Ver `lib/llamada/confianza.ts`.
 */
import { describe, expect, it } from 'vitest';

import { MINUTOS_CON_IDENTIDAD, MINUTOS_SIN_IDENTIDAD, tocaRenovar } from '../src/lib/llamada/confianza';

const MIN = 60_000;

describe('tocaRenovar', () => {
  it('la pide cuando no hay ninguna', () => {
    expect(tocaRenovar(0, 10, 1_000_000)).toBe(true);
  });

  it('no la pide recién pedida', () => {
    const ahora = 1_000_000;
    const hasta = ahora + 10 * MIN;
    expect(tocaRenovar(hasta, 10, ahora)).toBe(false);
  });

  it('no la pide mientras quede más de la mitad', () => {
    const ahora = 1_000_000;
    const hasta = ahora + 10 * MIN;
    // Cuatro minutos después: quedan seis de diez.
    expect(tocaRenovar(hasta, 10, ahora + 4 * MIN)).toBe(false);
  });

  it('la pide justo a la mitad', () => {
    const ahora = 1_000_000;
    const hasta = ahora + 10 * MIN;
    expect(tocaRenovar(hasta, 10, ahora + 5 * MIN)).toBe(true);
  });

  it('la pide si ya caducó', () => {
    const ahora = 1_000_000;
    expect(tocaRenovar(ahora - MIN, 10, ahora)).toBe(true);
  });

  it('la ventana corta es la de cuando se sabe quién habla', () => {
    expect(MINUTOS_CON_IDENTIDAD).toBeLessThan(MINUTOS_SIN_IDENTIDAD);
  });
});
