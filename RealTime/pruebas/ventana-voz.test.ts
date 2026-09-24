/**
 * Cómo se corta el micrófono en ventanas para saber quién habla.
 *
 * El fallo de antes, que estas pruebas no dejan volver: el suelo de ruido se
 * recalculaba mientras hablabas, a los ~300 ms alcanzaba a la voz y solo
 * pasaban picos de sílaba pegados. Ahora la voz seguida viaja entera, en
 * ventanas de hasta tres segundos, y el núcleo decide qué parte es voz.
 * Ver `lib/identidad/ventana-voz.ts` y `perseo_core/servicios/biometria_senal.py`.
 */
import { describe, expect, it } from 'vitest';

import {
  MUESTRAS_SEGUNDO,
  SONIDO_MINIMO,
  VENTANA_MAXIMA,
  VentanaVoz,
} from '../src/lib/identidad/ventana-voz';
import { motivoDelNucleo, resumenDeToma } from '../src/lib/identidad/identidad';

/** Un trozo como los del worklet: 1024 muestras, 64 ms. */
const TROZO = 1024;

/** Voz: tono con envolvente de sílabas, la misma que usan las pruebas del núcleo. */
function voz(desde: number): Int16Array {
  const trozo = new Int16Array(TROZO);
  for (let i = 0; i < TROZO; i++) {
    const n = desde + i;
    const envolvente = 0.2 + 0.8 * Math.abs(Math.sin((2 * Math.PI * 3 * n) / MUESTRAS_SEGUNDO));
    trozo[i] = Math.round(8000 * envolvente * Math.sin(n * 0.06));
  }
  return trozo;
}

const silencio = () => new Int16Array(TROZO);

/** Mete `segundos` de lo que diga `fuente` y devuelve las ventanas que salgan. */
function meter(ventana: VentanaVoz, segundos: number, fuente: (desde: number) => Int16Array) {
  const salidas: Int16Array[] = [];
  const trozos = Math.round((segundos * MUESTRAS_SEGUNDO) / TROZO);
  for (let i = 0; i < trozos; i++) {
    const lista = ventana.empujar(fuente(i * TROZO));
    if (lista) salidas.push(lista);
  }
  return salidas;
}

describe('VentanaVoz', () => {
  it('la voz seguida sale entera, en ventanas de tres segundos', () => {
    const ventana = new VentanaVoz();
    const salidas = meter(ventana, 7, voz);
    expect(salidas.length).toBe(2);
    for (const lista of salidas) {
      expect(lista.length).toBeGreaterThanOrEqual(VENTANA_MAXIMA);
    }
  });

  it('una frase se manda al acabar, con sus pausas dentro', () => {
    const ventana = new VentanaVoz();
    expect(meter(ventana, 1.5, voz)).toEqual([]);
    const salidas = meter(ventana, 0.6, silencio);
    expect(salidas.length).toBe(1);
    // Voz y el silencio de cierre: el núcleo quita lo que sobre.
    expect(salidas[0].length).toBeGreaterThanOrEqual(1.5 * MUESTRAS_SEGUNDO);
  });

  it('el silencio no manda nada', () => {
    expect(meter(new VentanaVoz(), 10, silencio)).toEqual([]);
  });

  it('una tos suelta no merece el viaje', () => {
    const ventana = new VentanaVoz();
    meter(ventana, 0.3, voz);
    expect(meter(ventana, 1, silencio)).toEqual([]);
    expect(0.3 * MUESTRAS_SEGUNDO).toBeLessThan(SONIDO_MINIMO);
  });

  it('descartar tira lo acumulado', () => {
    const ventana = new VentanaVoz();
    meter(ventana, 2, voz);
    ventana.descartar();
    expect(meter(ventana, 0.6, silencio)).toEqual([]);
  });

  it('el suelo no alcanza a la voz a mitad de frase', () => {
    // Lo que rompía el detector de antes: tras un rato hablando, todo pasaba
    // a contar como silencio. Aquí, diez segundos seguidos siguen saliendo.
    const salidas = meter(new VentanaVoz(), 10, voz);
    expect(salidas.length).toBe(3);
  });

  it('un ventilador que se queda deja de contar como voz', () => {
    const zumbido = () => {
      const trozo = new Int16Array(TROZO);
      for (let i = 0; i < TROZO; i++) trozo[i] = Math.round(1130 * Math.sin(i * 0.06));
      return trozo;
    };
    // Como mucho una ventana al principio, que el núcleo tira por plana; luego nada.
    expect(meter(new VentanaVoz(), 20, zumbido).length).toBeLessThanOrEqual(1);
  });
});

describe('resumenDeToma', () => {
  it('la primera toma dice cuánta voz útil tenía y pide otra', () => {
    const texto = resumenDeToma({ ok: true, añadido: ['voz'], segundos_voz: 5.84, voces: 1 });
    expect(texto).toContain('5,8 s de voz útil');
    expect(texto).toContain('Graba otra toma');
  });

  it('una toma que no se parece a las anteriores lo avisa', () => {
    const texto = resumenDeToma({ ok: true, añadido: ['voz'], segundos_voz: 5, parecido_voz: 0.21, voces: 2 });
    expect(texto).toContain('se parece poco');
  });

  it('con tres tomas ya no pide más', () => {
    const texto = resumenDeToma({ ok: true, añadido: ['voz'], segundos_voz: 5, parecido_voz: 0.8, voces: 3 });
    expect(texto).not.toContain('Graba otra');
    expect(texto).toContain('80 %');
  });

  it('la foto cuenta ángulos', () => {
    expect(resumenDeToma({ ok: true, añadido: ['cara'], caras: 1 })).toContain('un poco girada');
  });
});

describe('motivoDelNucleo', () => {
  it('quita la envoltura del puente de Rust y deja el motivo', () => {
    const e = 'El nucleo respondio 400 Bad Request: Se oye muy poca voz (0.8 s).';
    expect(motivoDelNucleo(e)).toBe('Se oye muy poca voz (0.8 s).');
  });

  it('lo que no viene envuelto se deja como está', () => {
    expect(motivoDelNucleo(new Error('sin red'))).toBe('Error: sin red');
  });
});
