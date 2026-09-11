/**
 * La aritmética de los reintentos.
 *
 * Es la pieza que dejó la app horas en «Conectando…» el 2026-08-17 abriendo una
 * sesión por segundo. Un fallo aquí no revienta nada: se ve como «no
 * conecta», que es lo más caro de diagnosticar.
 */

import { describe, expect, it } from 'vitest';

import {
  ESPERA_MAXIMA_RECONEXION,
  ESPERA_MAXIMA_TRAS_LIMITE,
  ESPERA_TRAS_LIMITE,
  avisoDeEspera,
  esLimite,
  planificarReintento,
} from '../src/lib/reconexion';

describe('esLimite', () => {
  it('reconoce el 1011 que manda Gemini al pasarse de cuota', () => {
    expect(esLimite({ codigo: 1011, motivo: 'You exceeded your current quota' })).toBe(true);
  });

  it('reconoce el texto aunque el código sea otro', () => {
    expect(esLimite({ codigo: 1000, motivo: 'RESOURCE_EXHAUSTED' })).toBe(true);
    expect(esLimite({ codigo: 1000, motivo: 'Too many requests' })).toBe(true);
  });

  it('no confunde una caída de red con un límite', () => {
    expect(esLimite({ codigo: 1006, motivo: '' })).toBe(false);
    expect(esLimite({})).toBe(false);
  });

  it('no toma por límite el testigo de sesión caducado', () => {
    expect(esLimite({ codigo: 1007, motivo: 'Invalid session handle' })).toBe(false);
  });
});

describe('planificarReintento', () => {
  it('empieza en un segundo cuando es un corte normal', () => {
    expect(planificarReintento(0, { codigo: 1006 })).toEqual({ causa: 'normal', esperaMs: 1000 });
  });

  it('dobla la espera con cada intento sin sesión estable', () => {
    expect(planificarReintento(1, {}).esperaMs).toBe(2000);
    expect(planificarReintento(3, {}).esperaMs).toBe(8000);
  });

  it('tiene techo en la espera normal', () => {
    expect(planificarReintento(30, {}).esperaMs).toBe(ESPERA_MAXIMA_RECONEXION);
  });

  it('ante un límite espera un minuto largo desde el primer intento', () => {
    // Lo contrario —reintentar en un segundo— es lo que alimentaba el bucle:
    // cada intento cuenta para el mismo límite que acaba de saltar.
    const plan = planificarReintento(0, { codigo: 1011, motivo: 'You exceeded your current quota' });
    expect(plan.causa).toBe('limite');
    expect(plan.esperaMs).toBe(ESPERA_TRAS_LIMITE);
  });

  it('sube la espera del límite y la corta a cinco minutos', () => {
    const cierre = { codigo: 1011, motivo: 'quota' };
    expect(planificarReintento(1, cierre).esperaMs).toBe(ESPERA_TRAS_LIMITE * 2);
    expect(planificarReintento(10, cierre).esperaMs).toBe(ESPERA_MAXIMA_TRAS_LIMITE);
  });

  it('trata un contador imposible como cero', () => {
    expect(planificarReintento(-5, {}).esperaMs).toBe(1000);
  });
});

describe('avisoDeEspera', () => {
  it('dice en segundos las esperas cortas', () => {
    expect(avisoDeEspera({ causa: 'normal', esperaMs: 4000 })).toContain('4 s');
  });

  it('dice en minutos las largas, y que la culpa es del límite', () => {
    const texto = avisoDeEspera({ causa: 'limite', esperaMs: 120_000 });
    expect(texto).toContain('2 min');
    expect(texto).toMatch(/limitando/i);
  });
});
