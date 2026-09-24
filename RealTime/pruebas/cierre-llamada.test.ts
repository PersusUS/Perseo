/**
 * Qué se hace tras un cierre del enlace de voz: con el testigo de sesión y con
 * la ventana deslizante. Ver `trasElCierre` en `lib/llamada/reconexion.ts`.
 *
 * El caso que faltaba, medido en `llamada.log` el 2026-09-23: al despertar el
 * portátil el servidor cerraba con `1008 BidiGenerateContent session not found`
 * y el testigo viejo se volvía a mandar; se perdían dos intentos seguidos antes
 * de tirarlo.
 */
import { describe, expect, it } from 'vitest';

import { CONTEXTO_DESLIZANTE, trasElCierre } from '../src/lib/llamada/reconexion';

const cierre = (codigo: number | undefined, motivo = '', hayTestigo = true, intentos = 0) =>
  trasElCierre({ codigo, motivo, hayTestigo, intentos });

describe('trasElCierre', () => {
  it('1007 con testigo: se tira y se reintenta ya', () => {
    const d = cierre(1007, 'Invalid session handle');
    expect(d.tirarTestigo).toBe(true);
    expect(d.reintentarYa).toBe(true);
  });

  it('1008 «session not found» al despertar: igual que el 1007', () => {
    const d = cierre(1008, 'BidiGenerateContent session not found');
    expect(d.tirarTestigo).toBe(true);
    expect(d.reintentarYa).toBe(true);
  });

  it('un 1008 que no habla de la sesión no toca el testigo en el primer intento', () => {
    const d = cierre(1008, 'Policy violation');
    expect(d.tirarTestigo).toBe(false);
  });

  it('dos sesiones cortas seguidas con testigo: se tira, pero con la espera normal', () => {
    const d = cierre(1011, 'Internal error encountered.', true, 1);
    expect(d.tirarTestigo).toBe(true);
    expect(d.reintentarYa).toBe(false);
  });

  it('sin testigo no hay nada que tirar', () => {
    const d = cierre(1007, 'Invalid session handle', false);
    expect(d.tirarTestigo).toBe(false);
    expect(d.porque).toBe('');
  });

  it('si el servidor rechaza la ventana deslizante, se sigue sin ella y ya', () => {
    const d = cierre(1007, 'Unknown name "contextWindowCompression" at \'setup\'', false);
    expect(d.sinCompresion).toBe(true);
    expect(d.reintentarYa).toBe(true);
  });

  it('un cierre normal no apaga la compresión', () => {
    expect(cierre(1000, '', false).sinCompresion).toBe(false);
  });

  it('la ventana deslizante va con sus valores por defecto', () => {
    expect(CONTEXTO_DESLIZANTE).toEqual({ slidingWindow: {} });
  });
});
