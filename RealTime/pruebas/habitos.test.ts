/**
 * Las cuentas del seguimiento de hábitos.
 *
 * Aquí vive la única contabilidad de los hábitos de toda la casa: la pantalla
 * la dibuja, el Perseo de la llamada la lee y el núcleo guarda una copia de lo
 * que salga de aquí. Si estas funciones cuentan mal, las tres cosas mienten a
 * la vez y ninguna se contradice, que es la peor forma de estar equivocado.
 *
 * Dos cosas se prueban con más saña que las demás:
 *
 *  - **La racha cruza el mes.** Contada dentro del mes valía como mucho 1 cada
 *    día 1, y ver una cadena de cuarenta días caer a uno por girar la hoja del
 *    calendario es de las cosas que hacen abandonar un tracker.
 *  - **El resumen dice lo que hay y no lo que se supone.** Es el texto que
 *    Perseo lee en voz alta. Un modelo al que se le dan cifras adornadas las
 *    repite adornadas.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';

import {
  ALMACEN, MES_VACIO,
  clave, diasDelMes, foto, guardar, leer, porcentaje, racha, resumen, semanaDe,
  type Datos,
} from '../src/lib/datos/habitos';

/** Un almacén a medida. Los días marcados se dan por hábito y por mes. */
function almacen(
  habitos: { id: string; nombre: string }[],
  meses: Record<string, Record<string, number[]>>,
): Datos {
  const salida: Datos = { habitos, meses: {} };
  for (const [k, porHabito] of Object.entries(meses)) {
    const marcas: Record<string, boolean> = {};
    for (const [id, dias] of Object.entries(porHabito)) {
      for (const d of dias) marcas[`${id}|${d}`] = true;
    }
    salida.meses[k] = { ...MES_VACIO, marcas };
  }
  return salida;
}

const GIMNASIO = [{ id: 'g', nombre: 'Gimnasio' }];

describe('las cuentas de andar por casa', () => {
  it('un mes sin nada da cero y no NaN', () => {
    expect(porcentaje(0, 0)).toBe(0);
    expect(porcentaje(3, 4)).toBe(75);
  });

  it('las semanas son bloques de siete desde el 1, no semanas naturales', () => {
    // Es como están agrupadas las columnas de la rejilla, y la gráfica de la
    // derecha tiene que contar lo mismo que se ve debajo.
    expect(semanaDe(1)).toBe(1);
    expect(semanaDe(7)).toBe(1);
    expect(semanaDe(8)).toBe(2);
    expect(semanaDe(31)).toBe(5);
  });

  it('sabe cuántos días tiene un mes, febrero bisiesto incluido', () => {
    expect(diasDelMes(2026, 1)).toBe(28);
    expect(diasDelMes(2028, 1)).toBe(29);
    expect(diasDelMes(2026, 7)).toBe(31);
    expect(clave(2026, 7)).toBe('2026-08');
  });
});

describe('la racha', () => {
  it('cuenta hacia atrás y se rompe en el primer hueco', () => {
    const d = almacen(GIMNASIO, { '2026-08': { g: [10, 11, 12, 14, 15] } });
    expect(racha(d, 'g', 2026, 7, 15)).toBe(2);
    expect(racha(d, 'g', 2026, 7, 12)).toBe(3);
  });

  it('vale cero si el día de referencia no está marcado', () => {
    const d = almacen(GIMNASIO, { '2026-08': { g: [10, 11, 12] } });
    expect(racha(d, 'g', 2026, 7, 13)).toBe(0);
  });

  it('CRUZA el cambio de mes en vez de reiniciarse cada día 1', () => {
    // Julio entero desde el 29 y agosto desde el 1: son cinco días seguidos,
    // no uno. Contada dentro del mes, esta cifra valía 1 y era el motivo por el
    // que la racha no servía para nada el primer día de cada mes.
    const d = almacen(GIMNASIO, {
      '2026-07': { g: [29, 30, 31] },
      '2026-08': { g: [1, 2] },
    });
    expect(racha(d, 'g', 2026, 7, 2)).toBe(5);
  });

  it('cruza también el cambio de año', () => {
    const d = almacen(GIMNASIO, {
      '2025-12': { g: [30, 31] },
      '2026-01': { g: [1] },
    });
    expect(racha(d, 'g', 2026, 0, 1)).toBe(3);
  });

  it('se para en el hueco aunque esté en el mes anterior', () => {
    // Julio termina el 30, no el 31: la cadena se corta ahí.
    const d = almacen(GIMNASIO, {
      '2026-07': { g: [29, 30] },
      '2026-08': { g: [1, 2] },
    });
    expect(racha(d, 'g', 2026, 7, 2)).toBe(2);
  });

  it('no se va por un bucle infinito con un almacén lleno del todo', () => {
    // Veinte meses seguidos completos. Sin el tope de saltos, esto retrocedería
    // hasta el principio de los tiempos buscando el hueco que no hay.
    const meses: Record<string, Record<string, number[]>> = {};
    for (let m = 0; m < 24; m++) {
      const f = new Date(2025, m, 1);
      const k = clave(f.getFullYear(), f.getMonth());
      meses[k] = { g: Array.from({ length: diasDelMes(f.getFullYear(), f.getMonth()) }, (_, i) => i + 1) };
    }
    const n = racha(almacen(GIMNASIO, meses), 'g', 2026, 7, 15);
    expect(n).toBeGreaterThan(300);
    expect(Number.isFinite(n)).toBe(true);
  });
});

describe('la foto', () => {
  const AHORA = new Date(2026, 7, 10, 12, 0, 0); // 10 de agosto de 2026
  const DOS = [{ id: 'g', nombre: 'Gimnasio' }, { id: 'l', nombre: 'Leer' }];

  it('cuenta hasta hoy y no hasta fin de mes', () => {
    // Con el mes entero por objetivo, el anillo no llegaría al 100 % ni siendo
    // perfecto hasta hoy, y «faltan» contaría días que aún no existen.
    const d = almacen(DOS, { '2026-08': { g: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], l: [1, 2] } });
    const f = foto(d, AHORA);
    expect(f.objetivo).toBe(20);
    expect(f.hechos).toBe(12);
    expect(f.porcentaje).toBe(60);
    expect(f.fecha).toBe('10/08/2026');
    expect(f.mes).toBe('agosto de 2026');
  });

  it('no cuenta lo que esté marcado en el futuro', () => {
    const d = almacen(DOS, { '2026-08': { g: [10, 20, 25], l: [] } });
    expect(foto(d, AHORA).hechos).toBe(1);
  });

  it('dice de cada hábito si hoy está hecho', () => {
    const d = almacen(DOS, { '2026-08': { g: [10], l: [9] } });
    const f = foto(d, AHORA);
    expect(f.habitos.find(h => h.nombre === 'Gimnasio')?.hoy).toBe(true);
    expect(f.habitos.find(h => h.nombre === 'Leer')?.hoy).toBe(false);
  });

  it('la media del estado mental ignora los días en blanco', () => {
    // No haber apuntado el ánimo no es tener el ánimo por los suelos: contar
    // los huecos como cero convertiría un mes a medio apuntar en un mes malo.
    const d = almacen(DOS, {});
    d.meses['2026-08'] = { marcas: {}, animo: { 1: 8, 2: 6 }, motivacion: {} };
    const f = foto(d, AHORA);
    expect(f.animo).toBe(7);
    expect(f.motivacion).toBe(null);
  });
});

describe('el resumen que lee Perseo en voz alta', () => {
  const AHORA = new Date(2026, 7, 10, 12, 0, 0);

  it('abre con la cifra global y la fecha a la que corresponde', () => {
    const d = almacen(
      [{ id: 'g', nombre: 'Gimnasio' }, { id: 'l', nombre: 'Leer' }],
      { '2026-08': { g: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], l: [10] } },
    );
    const t = resumen(d, AHORA);
    expect(t).toContain('10/08/2026');
    expect(t).toContain('agosto de 2026');
    expect(t).toContain('11 casillas de 20');
  });

  it('lo que hoy falta va delante del detalle: es lo accionable', () => {
    const d = almacen(
      [{ id: 'g', nombre: 'Gimnasio' }, { id: 'l', nombre: 'Leer' }],
      { '2026-08': { g: [10], l: [] } },
    );
    const t = resumen(d, AHORA);
    expect(t).toContain('Hoy le faltan por hacer: Leer.');
    expect(t.indexOf('Hoy le faltan')).toBeLessThan(t.indexOf('Detalle del mes'));
  });

  it('distingue «los lleva todos» de «no ha marcado ninguno»', () => {
    const dos = [{ id: 'g', nombre: 'Gimnasio' }, { id: 'l', nombre: 'Leer' }];
    expect(resumen(almacen(dos, { '2026-08': { g: [10], l: [10] } }), AHORA))
      .toContain('Hoy los lleva todos hechos.');
    expect(resumen(almacen(dos, { '2026-08': { g: [9], l: [9] } }), AHORA))
      .toContain('Hoy todavía no ha marcado ninguno.');
  });

  it('saca las rachas vivas, y solo a partir de tres días', () => {
    const d = almacen(
      [{ id: 'g', nombre: 'Gimnasio' }, { id: 'l', nombre: 'Leer' }],
      { '2026-08': { g: [8, 9, 10], l: [10] } },
    );
    const t = resumen(d, AHORA);
    expect(t).toContain('Gimnasio, 3 días seguidos');
    // Una racha de un día no es una racha y no merece que se hable de ella.
    expect(t).not.toContain('Leer, 1 día seguidos');
  });

  it('sin ningún hábito lo dice y no devuelve un cero triunfal', () => {
    expect(resumen({ habitos: [], meses: {} }, AHORA))
      .toContain('no tiene ningún hábito dado de alta');
  });
});

describe('el almacén', () => {
  /* Un `localStorage` de mentira, y no un DOM entero: estas pruebas no son
     sobre el navegador sino sobre lo que `leer()` hace con lo que encuentre
     dentro —nada, basura, o hábitos con el emoji de antes—. Traerse `jsdom`
     para cuatro `getItem` sería una dependencia de más en el CI por cuatro
     líneas de aquí. */
  beforeEach(() => {
    const caja = new Map<string, string>();
    vi.stubGlobal('localStorage', {
      getItem: (k: string) => caja.get(k) ?? null,
      setItem: (k: string, v: string) => { caja.set(k, v); },
      removeItem: (k: string) => { caja.delete(k); },
      clear: () => caja.clear(),
    });
  });

  it('sin nada guardado arranca con los doce de fábrica', () => {
    expect(leer().habitos).toHaveLength(12);
    expect(leer().habitos[0].nombre).toBe('Levantarse a las 06:00');
  });

  it('un almacén ilegible se sustituye en vez de dejar la pantalla en blanco', () => {
    localStorage.setItem(ALMACEN, '{roto');
    expect(leer().habitos).toHaveLength(12);
  });

  it('descarta el emoji que los hábitos llevaban hasta el 2026-08-25', () => {
    // Se cae aquí, en la puerta, y no en cada sitio donde se dibuja un hábito.
    localStorage.setItem(ALMACEN, JSON.stringify({
      habitos: [{ id: 'g', nombre: 'Gimnasio', icono: '🏋️' }],
      meses: {},
    }));
    expect(leer().habitos[0]).toEqual({ id: 'g', nombre: 'Gimnasio' });
  });

  it('lo guardado vuelve entero', () => {
    const d = almacen(GIMNASIO, { '2026-08': { g: [1, 2] } });
    guardar(d);
    expect(leer()).toEqual(d);
  });
});
