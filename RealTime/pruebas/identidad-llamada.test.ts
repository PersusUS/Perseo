/**
 * Las dos reglas que decidían solas dentro de `App.tsx`: cuándo un aviso de
 * identidad es nuevo, y cuándo ha cambiado quién está delante de la cámara.
 *
 * Vivían mezcladas con la pantalla y no las probaba nadie. Ver
 * `lib/llamada/identidad-llamada.ts`.
 */
import { describe, expect, it } from 'vitest';

import {
  NO_REPETIR_MS,
  avisoNuevo,
  huellaDeCaras,
} from '../src/lib/llamada/identidad-llamada';

const caja: [number, number, number, number] = [0, 0, 10, 10];
const cara = (nombre: string | null) => ({ nombre, caja, confianza: 0.9 });

describe('avisoNuevo', () => {
  const ahora = 1_000_000;

  it('el primero siempre pasa', () => {
    expect(avisoNuevo({ texto: '', cuando: 0 }, '[IDENTIDAD] Habla Persus.', ahora)).toBe(true);
  });

  it('el mismo texto dentro del minuto no se repite', () => {
    const ultimo = { texto: 'igual', cuando: ahora - 30_000 };
    expect(avisoNuevo(ultimo, 'igual', ahora)).toBe(false);
  });

  it('pasado el minuto vuelve a pasar', () => {
    const ultimo = { texto: 'igual', cuando: ahora - NO_REPETIR_MS - 1 };
    expect(avisoNuevo(ultimo, 'igual', ahora)).toBe(true);
  });

  it('un texto distinto pasa siempre, aunque sea seguido', () => {
    // Voz y cara son dos avisos distintos, y los dos importan la primera vez.
    const ultimo = { texto: 'habla Persus', cuando: ahora };
    expect(avisoNuevo(ultimo, 'delante de la camara: Persus', ahora)).toBe(true);
  });
});

describe('huellaDeCaras', () => {
  it('sin caras, sin huella', () => {
    expect(huellaDeCaras([])).toBe('');
  });

  it('las caras sin nombre no cuentan', () => {
    // Sin el filtro saldria un literal «null» en el aviso al modelo.
    expect(huellaDeCaras([cara(null), cara(null)])).toBe('');
    expect(huellaDeCaras([cara('Persus'), cara(null)])).toBe('Persus');
  });

  it('el orden de deteccion no cambia la huella', () => {
    const a = huellaDeCaras([cara('Persus'), cara('Javi')]);
    const b = huellaDeCaras([cara('Javi'), cara('Persus')]);
    expect(a).toBe(b);
    expect(a).toBe('Javi, Persus');
  });

  it('quien entra cambia la huella', () => {
    expect(huellaDeCaras([cara('Persus')])).not.toBe(
      huellaDeCaras([cara('Persus'), cara('Javi')]),
    );
  });
});
