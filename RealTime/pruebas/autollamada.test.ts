/**
 * Qué se hace con un aviso de llamada. Ver `lib/llamada/autollamada.ts`.
 *
 * El fallo que esto arregla: un encargo que terminaba en plena llamada se
 * tiraba, porque la regla «ya está en llamada, no hagas nada» era para la
 * palabra clave y se aplicaba también a los avisos con motivo.
 */
import { describe, expect, it } from 'vitest';

import { avisoEnLlamada, decidirAviso } from '../src/lib/llamada/autollamada';

describe('decidirAviso', () => {
  it('un encargo o un recordatorio en plena llamada se cuenta en vivo', () => {
    expect(decidirAviso('Recordatorio: el té', 'connected', false)).toBe('contar-en-vivo');
  });

  it('la palabra clave en plena llamada no corta nada', () => {
    expect(decidirAviso('', 'connected', false)).toBe('nada');
  });

  it('mientras conecta, se guarda para contarlo al abrir', () => {
    expect(decidirAviso('El encargo #3 ha terminado.', 'connecting', false)).toBe('guardar');
  });

  it('sin llamada, un motivo suena el timbre y decide él', () => {
    expect(decidirAviso('Recordatorio: el té', 'disconnected', false)).toBe('timbre');
    expect(decidirAviso('Recordatorio: el té', 'error', true)).toBe('timbre');
  });

  it('sin motivo es la palabra clave: entra en llamada, salvo recién abierta', () => {
    expect(decidirAviso('', 'disconnected', false)).toBe('llamar');
    expect(decidirAviso('', 'disconnected', true)).toBe('nada');
  });
});

describe('avisoEnLlamada', () => {
  it('cada motivo es una línea, y va marcado como aviso del sistema', () => {
    const texto = avisoEnLlamada('Recordatorio: el té\nEl encargo #3 ha terminado.\n');
    expect(texto.startsWith('[AVISO DEL SISTEMA')).toBe(true);
    expect(texto).toContain('- Recordatorio: el té');
    expect(texto).toContain('- El encargo #3 ha terminado.');
  });
});
