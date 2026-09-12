/**
 * La regla del micrófono: qué modos dejan el paso abierto sin apretar nada.
 *
 * Parece una tontería de una línea, y es la que decidía si la pantalla ponía
 * «Escuchando» mientras el micrófono estaba cerrado — pasó el 2026-09-09. Ver
 * `lib/llamada/microfono.ts`.
 */
import { describe, expect, it } from 'vitest';

import { pasoSiempreAbierto } from '../src/lib/llamada/microfono';

describe('pasoSiempreAbierto', () => {
  it('en manos libres el paso está abierto', () => {
    expect(pasoSiempreAbierto('manos-libres')).toBe(true);
  });

  it('en pulsar para hablar, no', () => {
    expect(pasoSiempreAbierto('pulsar')).toBe(false);
  });
});
