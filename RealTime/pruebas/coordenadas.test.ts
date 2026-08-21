/**
 * La traducción de «donde señala el modelo» a «donde clica el ratón».
 *
 * Es aritmética, así que cuando falla no revienta: el clic cae en otro sitio y
 * parece que el modelo se equivoca. Pasó en una llamada del 2026-08-17 (H-50).
 */

import { describe, expect, it } from 'vitest';

import {
  ACCIONES_DE_RATON,
  GeometriaPantalla,
  aPixeles,
  traducirParametroDeRaton,
} from '../src/lib/coordenadas';

const PANTALLA: GeometriaPantalla = {
  ancho_imagen: 1280,
  alto_imagen: 720,
  ancho_pantalla: 1920,
  alto_pantalla: 1080,
};

describe('aPixeles', () => {
  it('lleva el centro al centro', () => {
    expect(aPixeles(500, 500, PANTALLA)).toEqual([960, 540]);
  });

  it('deja el origen en el origen', () => {
    expect(aPixeles(0, 0, PANTALLA)).toEqual([0, 0]);
  });

  it('no se sale por la esquina de abajo', () => {
    // El modelo redondea a 1000, que sería un píxel fuera de la pantalla.
    expect(aPixeles(1000, 1000, PANTALLA)).toEqual([1919, 1079]);
  });

  it('escala cada eje por su lado', () => {
    expect(aPixeles(250, 750, PANTALLA)).toEqual([480, 810]);
  });

  it('vale para una pantalla que no es 16:9', () => {
    const cuadrada = { ...PANTALLA, ancho_pantalla: 1000, alto_pantalla: 1000 };
    expect(aPixeles(300, 300, cuadrada)).toEqual([300, 300]);
  });
});

describe('traducirParametroDeRaton', () => {
  it('traduce unas coordenadas sueltas', () => {
    expect(traducirParametroDeRaton('300,350', PANTALLA)).toBe('576,378');
  });

  it('conserva el tipo de clic delante', () => {
    expect(traducirParametroDeRaton('derecho 500,500', PANTALLA)).toBe('derecho 960,540');
  });

  it('deja intacto lo que no lleva coordenadas', () => {
    // Un clic sin coordenadas lo rechaza el núcleo; aquí no se decide eso.
    expect(traducirParametroDeRaton('izquierdo', PANTALLA)).toBe('izquierdo');
    expect(traducirParametroDeRaton('', PANTALLA)).toBe('');
  });

  it('respeta los espacios tal cual venían', () => {
    expect(traducirParametroDeRaton('doble  500,500', PANTALLA)).toBe('doble  960,540');
  });

  it('acepta decimales, que es como a veces señala el modelo', () => {
    expect(traducirParametroDeRaton('500.5,499.5', PANTALLA)).toBe('961,539');
  });

  it('no toca un texto que no es un punto', () => {
    expect(traducirParametroDeRaton('spotify', PANTALLA)).toBe('spotify');
  });
});

describe('ACCIONES_DE_RATON', () => {
  it('son las dos que llevan coordenadas, y solo esas', () => {
    expect([...ACCIONES_DE_RATON].sort()).toEqual(['click_raton', 'mover_raton']);
    expect(ACCIONES_DE_RATON.has('escribir_teclado')).toBe(false);
  });
});
