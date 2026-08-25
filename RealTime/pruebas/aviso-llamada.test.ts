/**
 * Las etiquetas de sentido de llamada y el cierre de avisos.
 *
 * Es el arreglo de la llamada del 2026-08-25: el señor Persus llamaba a
 * Perseo y este abría con «el sistema me ha notificado que un encargo ha
 * finalizado» — y lo repetía tras cada reconexión. Un fallo aquí no revienta
 * nada: se ve como Perseo contando un motivo que nadie le pidió, que es lo
 * más caro de diagnosticar porque suena a memoria rota.
 */

import { describe, expect, it } from 'vitest';

import {
  CIERRE_DE_AVISOS,
  entregaEnVivo,
  etiquetaEncargosResueltos,
  etiquetaOrigen,
} from '../src/lib/aviso-llamada';

describe('etiquetaOrigen', () => {
  it('la llamada entrante dice QUIÉN llama y quita el motivo inventado', () => {
    const etiqueta = etiquetaOrigen('entrante');
    expect(etiqueta).toContain('LLAMADA ENTRANTE');
    expect(etiqueta).toContain('EL SEÑOR PERSUS');
    expect(etiqueta).toContain('NO has llamado');
    // La frase que repetía el modelo en la llamada del 2026-08-25 no puede
    // salir de una etiqueta que niega la autollamada.
    expect(etiqueta).toMatch(/ningún encargo terminado/i);
  });

  it('la saliente lleva al motivo pegado y dice que llamó el modelo', () => {
    const etiqueta = etiquetaOrigen('saliente', 'el subagente s2 terminó');
    expect(etiqueta).toContain('LLAMADA SALIENTE');
    expect(etiqueta).toContain('TÚ');
    expect(etiqueta).toContain('el subagente s2 terminó');
  });

  it('una saliente sin motivo no etiqueta nada: un tag vacío confunde más que ninguno', () => {
    expect(etiquetaOrigen('saliente', null)).toBeNull();
    expect(etiquetaOrigen('saliente', '')).toBeNull();
  });
});

describe('etiquetaEncargosResueltos', () => {
  it('sin encargos no hay texto ni en instrucciones ni en vivo', () => {
    expect(etiquetaEncargosResueltos([])).toBeNull();
  });

  it('los lista como contexto y NUNCA como motivo de la llamada', () => {
    const etiqueta = etiquetaEncargosResueltos(['usar_mcp: hecho']);
    expect(etiqueta).toContain('usar_mcp: hecho');
    expect(etiqueta).toMatch(/NO el motivo/i);
  });
});

describe('CIERRE_DE_AVISOS', () => {
  it('cierra el asunto: informado y terminado, sin volver a mencionarlo', () => {
    expect(CIERRE_DE_AVISOS).toMatch(/INFORMADOS Y CERRADOS/);
    expect(CIERRE_DE_AVISOS).toMatch(/ha TERMINADO/i);
  });
});

describe('entregaEnVivo', () => {
  it('con solo motivo va una línea', () => {
    const texto = entregaEnVivo('terminó s3', []);
    expect(texto).toBe('Motivo de esta llamada: terminó s3');
  });

  it('con solo resultados van listados', () => {
    const lineas = entregaEnVivo(null, ['uno', 'dos'])!.split('\n');
    expect(lineas[0]).toBe('Resultados que debías contar:');
    expect(lineas).toContain('- uno');
    expect(lineas).toContain('- dos');
  });

  it('motivo y resultados conviven, y sin nada no se entrega texto en vivo', () => {
    const ambos = entregaEnVivo('s1', ['r'])!;
    expect(ambos).toContain('Motivo de esta llamada: s1');
    expect(ambos).toContain('- r');
    expect(entregaEnVivo(null, [])).toBeNull();
  });
});
