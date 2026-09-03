/**
 * Las reglas del corcho.
 *
 * Un tablero de tareas parece que no tiene lógica hasta que se le pide lo que
 * de verdad se le pide: que una nota caiga donde la soltaste, que el orden
 * dentro de la columna sea el que se ve, y que lo tirado se pueda recuperar en
 * su sitio. Las tres cosas se rompen en silencio —la nota aparece al final de
 * otra columna y nadie sabe cuándo pasó—, así que se prueban aquí y no a ojo.
 *
 * Lo que se vigila con más saña es la papelera: es la única puerta por la que
 * se pierde trabajo, y tiene que estar cerrada salvo cuando se abre a mano.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';

import {
  ALMACEN,
  anadir, aplicarOrden, borrar, crear, crearDeFuera, cuenta, deColumna,
  diasDesde, editar, foto, guardar, haceCuanto, leer, mover, moverDeFuera,
  porTitulo, restaurar, resumen, tirar, vaciarPapelera,
  type Columna, type Datos,
} from '../src/lib/tareas';

/** Un tablero a medida: títulos por columna, en orden. */
function tablero(porColumna: Record<string, string[]>): Datos {
  const tareas = [];
  for (const [columna, titulos] of Object.entries(porColumna)) {
    for (const titulo of titulos) {
      tareas.push({ ...crear({ titulo, columna: columna as any }), id: titulo });
    }
  }
  return { tareas };
}

const titulos = (d: Datos, c: any) => deColumna(d, c).map(t => t.titulo);

describe('el almacén', () => {
  /* Un `localStorage` de mentira, como en las pruebas de hábitos: lo que se
     prueba es qué hace `leer()` con lo que encuentre —nada, basura, o notas
     con campos que ya no existen—, no el navegador. */
  beforeEach(() => {
    const caja = new Map<string, string>();
    vi.stubGlobal('localStorage', {
      getItem: (k: string) => caja.get(k) ?? null,
      setItem: (k: string, v: string) => { caja.set(k, v); },
      removeItem: (k: string) => { caja.delete(k); },
      clear: () => caja.clear(),
    });
  });

  it('estrena tablero cuando no hay nada guardado', () => {
    const d = leer();
    expect(d.tareas.length).toBeGreaterThan(0);
    expect(cuenta(d, 'sin_hacer')).toBeGreaterThan(0);
  });

  it('respeta un tablero vacío guardado a propósito', () => {
    guardar({ tareas: [] });
    expect(leer().tareas).toEqual([]);
  });

  it('sustituye un almacén ilegible en vez de reventar', () => {
    localStorage.setItem(ALMACEN, '{esto no es json');
    expect(leer().tareas.length).toBeGreaterThan(0);
  });

  it('salva la nota aunque su color o su columna no se reconozcan', () => {
    localStorage.setItem(ALMACEN, JSON.stringify({
      tareas: [{ id: 'x', titulo: 'Sobrevive', columna: 'inventada', color: 'fucsia' }],
    }));
    const [t] = leer().tareas;
    expect(t.titulo).toBe('Sobrevive');
    expect(t.columna).toBe('sin_hacer');
    expect(t.color).toBe('amarillo');
  });

  it('descarta lo que no llega a ser una nota', () => {
    localStorage.setItem(ALMACEN, JSON.stringify({ tareas: [{ id: 'x' }, { titulo: 'sin id' }] }));
    expect(leer().tareas).toEqual([]);
  });
});

describe('mover', () => {
  it('lleva la nota a otra columna, al final', () => {
    const d = mover(tablero({ sin_hacer: ['a', 'b'], en_proceso: ['c'] }), 'a', 'en_proceso');
    expect(titulos(d, 'sin_hacer')).toEqual(['b']);
    expect(titulos(d, 'en_proceso')).toEqual(['c', 'a']);
  });

  it('la coloca delante de la nota indicada', () => {
    const d = mover(tablero({ sin_hacer: ['a'], en_proceso: ['b', 'c'] }), 'a', 'en_proceso', 'c');
    expect(titulos(d, 'en_proceso')).toEqual(['b', 'a', 'c']);
  });

  it('ordena dentro de la misma columna', () => {
    const d = mover(tablero({ sin_hacer: ['a', 'b', 'c'] }), 'c', 'sin_hacer', 'a');
    expect(titulos(d, 'sin_hacer')).toEqual(['c', 'a', 'b']);
  });

  it('no hace nada si la nota se suelta sobre sí misma', () => {
    const antes = tablero({ sin_hacer: ['a', 'b'] });
    expect(mover(antes, 'a', 'sin_hacer', 'a')).toBe(antes);
  });

  it('ignora un id que no existe', () => {
    const antes = tablero({ sin_hacer: ['a'] });
    expect(mover(antes, 'fantasma', 'completadas')).toBe(antes);
  });

  it('apunta la hora del movimiento', () => {
    const antes = tablero({ sin_hacer: ['a'] });
    antes.tareas[0].movida = '2020-01-01T00:00:00.000Z';
    const d = mover(antes, 'a', 'completadas');
    expect(d.tareas[0].movida).not.toBe('2020-01-01T00:00:00.000Z');
    expect(d.tareas[0].creada).toBe(antes.tareas[0].creada);
  });
});

describe('la papelera', () => {
  it('tirar mueve, no borra', () => {
    const d = tirar(tablero({ en_proceso: ['a'] }), 'a');
    expect(d.tareas).toHaveLength(1);
    expect(titulos(d, 'papelera')).toEqual(['a']);
  });

  it('recuperar devuelve a la columna de donde salió', () => {
    const d = restaurar(tirar(tablero({ completadas: ['a'] }), 'a'), 'a');
    expect(titulos(d, 'completadas')).toEqual(['a']);
  });

  it('recordar de dónde vino sobrevive a tirarla dos veces', () => {
    let d = tirar(tablero({ en_proceso: ['a'] }), 'a');
    d = tirar(d, 'a');
    expect(restaurar(d, 'a').tareas[0].columna).toBe('en_proceso');
  });

  it('recuperar una nota sin origen la deja en «sin hacer»', () => {
    const d = tablero({ papelera: ['a'] });
    expect(restaurar(d, 'a').tareas[0].columna).toBe('sin_hacer');
  });

  it('salir de la papelera olvida el origen', () => {
    const d = mover(tirar(tablero({ en_proceso: ['a'] }), 'a'), 'a', 'completadas');
    expect(d.tareas[0].previa).toBeUndefined();
  });

  it('borrar sí borra, y solo la pedida', () => {
    const d = borrar(tablero({ papelera: ['a', 'b'] }), 'a');
    expect(titulos(d, 'papelera')).toEqual(['b']);
  });

  it('vaciar se lleva la papelera y nada más', () => {
    const d = vaciarPapelera(tablero({ sin_hacer: ['a'], papelera: ['b', 'c'] }));
    expect(titulos(d, 'sin_hacer')).toEqual(['a']);
    expect(titulos(d, 'papelera')).toEqual([]);
  });
});

describe('editar y añadir', () => {
  it('cambia lo que se edita y deja quieto lo demás', () => {
    const antes = tablero({ sin_hacer: ['a'] });
    const d = editar(antes, 'a', { titulo: 'otro', color: 'lila' });
    expect(d.tareas[0].titulo).toBe('otro');
    expect(d.tareas[0].color).toBe('lila');
    expect(d.tareas[0].columna).toBe('sin_hacer');
    expect(antes.tareas[0].titulo).toBe('a');
  });

  it('la nota nueva nace en su columna, con giro y fechas', () => {
    const t = crear({ titulo: '  con espacios  ', columna: 'en_proceso' });
    expect(t.titulo).toBe('con espacios');
    expect(t.columna).toBe('en_proceso');
    expect(Math.abs(t.giro)).toBeLessThanOrEqual(3);
    expect(t.creada).toBe(t.movida);
    expect(titulos(anadir(tablero({ en_proceso: ['a'] }), t), 'en_proceso'))
      .toEqual(['a', 'con espacios']);
  });

  it('dos notas seguidas no comparten identificador', () => {
    expect(crear({ titulo: 'x' }).id).not.toBe(crear({ titulo: 'x' }).id);
  });
});

describe('lo que se lee en pantalla', () => {
  const ahora = new Date('2026-08-30T12:00:00Z');
  const hace = (min: number) => new Date(ahora.getTime() - min * 60000).toISOString();

  it('cuenta el tiempo como lo diría una persona', () => {
    expect(haceCuanto(hace(0), ahora)).toBe('ahora mismo');
    expect(haceCuanto(hace(5), ahora)).toBe('hace 5 min');
    expect(haceCuanto(hace(180), ahora)).toBe('hace 3 h');
    expect(haceCuanto(hace(60 * 26), ahora)).toBe('ayer');
    expect(haceCuanto(hace(60 * 24 * 3), ahora)).toBe('hace 3 días');
    expect(haceCuanto('2026-08-12T10:00:00Z', ahora)).toBe('12/08');
  });

  it('una fecha rota no pinta nada en vez de «Invalid Date»', () => {
    expect(haceCuanto('mañana', ahora)).toBe('');
  });

  it('el resumen dice lo que hay, y lo que no hay lo dice también', () => {
    expect(resumen({ tareas: [] })).toContain('vacío');
    const texto = resumen(tablero({ sin_hacer: ['comprar pan'], en_proceso: ['el panel'] }));
    expect(texto).toContain('1 sin hacer');
    expect(texto).toContain('el panel');
    expect(texto).toContain('comprar pan');
  });

  it('la papelera no cuenta como pendiente', () => {
    expect(resumen(tablero({ papelera: ['olvidada'] }))).toContain('vacío');
  });
});

/**
 * Lo que Perseo lee.
 *
 * Este texto no lo mira nadie en pantalla: se le entrega al modelo, en la
 * llamada y —por el espejo del núcleo— en el chat escrito. Una nota que se
 * cuela aquí es una nota que Perseo dice en voz alta, así que lo que se vigila
 * es qué entra y qué no: la papelera se cuenta y no se enumera, y lo que lleva
 * parado sale con su cifra en vez de perdido en la lista.
 */
describe('lo que Perseo lee', () => {
  const ahora = new Date('2026-09-03T12:00:00Z');
  /** Días atrás, en ISO. */
  const hace = (dias: number) =>
    new Date(ahora.getTime() - dias * 86400000).toISOString();

  /** Un tablero con las fechas puestas a mano: lo que se prueba aquí es
   *  precisamente cuánto tiempo lleva cada nota donde está. */
  function conFechas(
    notas: { titulo: string; columna: Columna; dias: number; detalle?: string }[],
  ): Datos {
    return {
      tareas: notas.map(n => ({
        ...crear({ titulo: n.titulo, detalle: n.detalle, columna: n.columna }),
        id: n.titulo,
        creada: hace(n.dias),
        movida: hace(n.dias),
      })),
    };
  }

  it('cuenta días cumplidos, no empezados', () => {
    expect(diasDesde(new Date(ahora.getTime() - 20 * 3600000).toISOString(), ahora)).toBe(0);
    expect(diasDesde(hace(1), ahora)).toBe(1);
    expect(diasDesde(hace(9), ahora)).toBe(9);
    // Una fecha ilegible cuenta como hoy: decir «parado desde 1970» sería peor
    // que no decir nada.
    expect(diasDesde('mañana', ahora)).toBe(0);
    // Y una del futuro tampoco da negativos.
    expect(diasDesde(new Date(ahora.getTime() + 86400000).toISOString(), ahora)).toBe(0);
  });

  it('la foto cuenta por columnas y deja la papelera fuera de la lista', () => {
    const d = conFechas([
      { titulo: 'a', columna: 'sin_hacer', dias: 0 },
      { titulo: 'b', columna: 'en_proceso', dias: 4 },
      { titulo: 'c', columna: 'completadas', dias: 1 },
      { titulo: 'd', columna: 'papelera', dias: 2 },
    ]);
    const f = foto(d, ahora);

    expect([f.sinHacer, f.enProceso, f.completadas, f.papelera]).toEqual([1, 1, 1, 1]);
    expect(f.tareas.map(t => t.titulo)).toEqual(['a', 'b', 'c']);
    expect(f.tareas.find(t => t.titulo === 'b')!.dias).toBe(4);
  });

  it('dice lo que tiene entre manos con su detalle y desde cuándo', () => {
    const texto = resumen(conFechas([
      { titulo: 'el panel', columna: 'en_proceso', dias: 5, detalle: 'falta la gráfica' },
    ]), ahora);

    expect(texto).toContain('el panel');
    expect(texto).toContain('falta la gráfica');
    expect(texto).toContain('sin moverse desde hace 5 días');
    expect(texto).toContain('Lleva tiempo parado');
  });

  it('lo movido hoy no se cuenta como atascado', () => {
    const texto = resumen(conFechas([
      { titulo: 'recién cogida', columna: 'en_proceso', dias: 0 },
    ]), ahora);

    expect(texto).toContain('movida hoy');
    expect(texto).not.toContain('Lleva tiempo parado');
  });

  it('un pendiente viejo lleva su antigüedad y uno de ayer no', () => {
    const texto = resumen(conFechas([
      { titulo: 'la mudanza', columna: 'sin_hacer', dias: 20 },
      { titulo: 'comprar pan', columna: 'sin_hacer', dias: 1 },
    ]), ahora);

    expect(texto).toContain('la mudanza (apuntada hace 20 días)');
    expect(texto).toContain('comprar pan');
    expect(texto).not.toContain('comprar pan (apuntada');
  });

  it('reconoce lo cerrado esta semana y olvida lo de hace un mes', () => {
    const texto = resumen(conFechas([
      { titulo: 'la factura', columna: 'completadas', dias: 2 },
      { titulo: 'lo de agosto', columna: 'completadas', dias: 30 },
    ]), ahora);

    expect(texto).toContain('Cerradas en los últimos siete días: la factura.');
    expect(texto).not.toContain('lo de agosto');
  });

  it('la papelera se cuenta pero no se enumera', () => {
    const texto = resumen(conFechas([
      { titulo: 'comprar pan', columna: 'sin_hacer', dias: 0 },
      { titulo: 'lo que tiró', columna: 'papelera', dias: 1 },
    ]), ahora);

    expect(texto).toContain('En la papelera hay 1 nota');
    expect(texto).not.toContain('lo que tiró');
  });

  it('el resumen guardado y el de pantalla cuentan lo mismo', () => {
    // No hay dos contabilidades: `resumenGuardado` es `resumen` leyendo el
    // almacén, y el espejo del núcleo recibe exactamente este texto.
    const d = conFechas([{ titulo: 'comprar pan', columna: 'sin_hacer', dias: 0 }]);
    expect(resumen(d, ahora)).toContain('1 sin hacer');
  });
});

/**
 * Lo que Perseo escribe.
 *
 * Aquí hay un segundo escritor sobre el tablero, y eso es lo que se vigila: que
 * escriba donde debe, que no mueva la nota equivocada cuando dos se parecen, y
 * que una orden mal formada —vienen por HTTP, de fuera de esta ventana— se
 * cuente en vez de aplicarse a medias.
 */
describe('lo que Perseo escribe', () => {
  beforeEach(() => {
    const caja = new Map<string, string>();
    vi.stubGlobal('localStorage', {
      getItem: (k: string) => caja.get(k) ?? null,
      setItem: (k: string, v: string) => { caja.set(k, v); },
      removeItem: (k: string) => { caja.delete(k); },
      clear: () => caja.clear(),
    });
    guardar({ tareas: [] });
  });

  it('clava la nota y la deja guardada', () => {
    const dicho = crearDeFuera({ titulo: '  Llamar al fontanero  ', detalle: 'el del bajo' });

    expect(dicho).toContain('Llamar al fontanero');
    const [t] = leer().tareas;
    expect(t.titulo).toBe('Llamar al fontanero');
    expect(t.detalle).toBe('el del bajo');
    expect(t.columna).toBe('sin_hacer');
  });

  it('no clava nada en la papelera: nacer en la basura no es nacer', () => {
    crearDeFuera({ titulo: 'x', columna: 'papelera' });
    expect(leer().tareas[0].columna).toBe('sin_hacer');
  });

  it('una nota sin título no se clava', () => {
    expect(crearDeFuera({ titulo: '   ' })).toContain('No se ha clavado');
    expect(leer().tareas).toEqual([]);
  });

  it('encuentra la nota aunque él la diga sin tildes ni mayúsculas', () => {
    guardar({ tareas: [crear({ titulo: 'Revisar el Presupuesto Anual' })] });
    expect(porTitulo(leer(), 'revisar el presupuesto anual')).not.toBeNull();
    expect(porTitulo(leer(), 'REVISAR EL')).not.toBeNull();
    expect(porTitulo(leer(), 'presupuesto')).not.toBeNull();

    guardar({ tareas: [crear({ titulo: 'Llamar a Jesús' })] });
    expect(porTitulo(leer(), 'llamar a jesus')).not.toBeNull();
  });

  it('con dos notas que encajan no mueve ninguna', () => {
    guardar({
      tareas: [crear({ titulo: 'Llamar al fontanero' }), crear({ titulo: 'Llamar al seguro' })],
    });
    expect(porTitulo(leer(), 'llamar')).toBe('ambigua');

    const dicho = moverDeFuera('llamar', 'completadas');
    expect(dicho).toContain('más de una');
    expect(cuenta(leer(), 'completadas')).toBe(0);
  });

  it('no resucita lo que él tiró aunque el título encaje', () => {
    guardar({ tareas: [{ ...crear({ titulo: 'Comprar pan' }), columna: 'papelera' }] });
    expect(porTitulo(leer(), 'comprar pan')).toBeNull();
    expect(moverDeFuera('comprar pan', 'en_proceso')).toContain('No hay ninguna nota');
  });

  it('mueve por título y lo cuenta', () => {
    guardar({ tareas: [crear({ titulo: 'El panel' })] });

    expect(moverDeFuera('el panel', 'completadas')).toContain('completadas');
    expect(leer().tareas[0].columna).toBe('completadas');
    // Y moverla adonde ya está no es un fallo, pero se dice.
    expect(moverDeFuera('el panel', 'completadas')).toContain('ya estaba');
  });

  it('a la papelera se dice que se recupera, porque se recupera', () => {
    guardar({ tareas: [crear({ titulo: 'Comprar pan' })] });
    const dicho = moverDeFuera('comprar pan', 'papelera');

    expect(dicho).toContain('no se ha borrado');
    expect(leer().tareas[0].columna).toBe('papelera');
    expect(restaurar(leer(), leer().tareas[0].id).tareas[0].columna).toBe('sin_hacer');
  });

  it('una nota que no existe se dice, no se inventa', () => {
    expect(moverDeFuera('la que no está', 'completadas')).toContain('No hay ninguna nota');
  });

  describe('las órdenes que llegan de fuera de la ventana', () => {
    it('crea y mueve', () => {
      aplicarOrden({ accion: 'crear', titulo: 'Comprar pan', detalle: 'integral' });
      expect(leer().tareas[0].detalle).toBe('integral');

      aplicarOrden({ accion: 'mover', titulo: 'comprar pan', columna: 'en_proceso' });
      expect(leer().tareas[0].columna).toBe('en_proceso');
    });

    it('una orden sin columna válida no toca nada', () => {
      guardar({ tareas: [crear({ titulo: 'Comprar pan' })] });

      expect(aplicarOrden({ accion: 'mover', titulo: 'comprar pan', columna: 'inventada' }))
        .toContain('no se ha tocado nada');
      expect(leer().tareas[0].columna).toBe('sin_hacer');
    });

    it('una orden que no se entiende se cuenta y no se aplica', () => {
      expect(aplicarOrden({ accion: 'borrar', titulo: 'Comprar pan' }))
        .toContain('Orden desconocida');
      expect(aplicarOrden(null)).toContain('Orden desconocida');
      expect(leer().tareas).toEqual([]);
    });
  });
});
