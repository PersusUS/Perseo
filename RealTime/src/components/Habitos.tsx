/**
 * El seguimiento de hábitos, dentro de la ventana de la app.
 *
 * Salió de una plantilla de hoja de cálculo y se rehízo el 2026-08-25 por
 * encargo del señor Persus —*«más intuitivo, sin emoticonos, todo más grande y
 * con el aire de Perseo»*—. Lo que cambió, y por qué:
 *
 *  1. **Dos aires, una pantalla.** `estilo` decide la paleta: `perseo` —negro,
 *     versalitas, monoespaciada— o `plantilla` —el beige del que salió—. Mismo
 *     reparto, mismas cifras, mismos gestos; solo cambian los colores. Se elige
 *     en Ajustes y se aplica al pulsar.
 *  2. **Sin iconos.** Cada hábito llevaba un emoji al lado del nombre. En una
 *     rejilla de 31 columnas era ruido de colores compitiendo con lo único que
 *     importa: si la casilla está marcada o no. Un `icono` guardado de antes se
 *     descarta al leer.
 *  3. **Se lee a un metro.** La plantilla original tenía tipografías de 7 a
 *     12 px porque en una hoja de cálculo se hace zoom; una ventana no. Toda la
 *     escala vive en variables (`--hb-t-*`, `--hb-celda`) y subió de golpe.
 *  4. **Lo que un tracker necesita y una hoja no tiene:** hoy va señalado, los
 *     días que aún no han llegado se apagan, los fines de semana se distinguen,
 *     se pinta arrastrando el ratón por varias casillas, la cabecera de un día
 *     marca la columna entera y cada hábito enseña su racha viva.
 *
 * Todo se edita: nombres, casillas, ánimo, motivación, alta y baja de hábitos.
 * Las cifras de la derecha —objetivo, hechos, faltan, porcentajes, el anillo,
 * las barras y el ranking— no se escriben nunca a mano: salen de las casillas.
 * Dos sitios donde teclear el mismo dato acaban discrepando.
 *
 * Dónde vive el dato: en el `localStorage` de la ventana, bajo `ALMACEN`. Es la
 * excepción consciente a «las caras no piensan» —el núcleo no tiene agente de
 * hábitos y no se le va a inventar uno para guardar doce casillas—, y está
 * aislada en una sola constante para el día que se mude a `perseo_core`.
 */

import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';

import type { EstiloHabitos } from '../lib/config';

const ALMACEN = 'perseo.habitos.v1';

type Habito = { id: string; nombre: string };

/** Un mes de datos. Las marcas van en un objeto plano con clave
 *  `idHabito|día` en vez de una matriz: añadir o quitar un hábito no tiene
 *  entonces que recolocar nada, y un hábito borrado se lleva sus marcas al
 *  filtrarlas por id. */
type Mes = {
  marcas: Record<string, boolean>;
  animo: Record<string, number>;
  motivacion: Record<string, number>;
};

type Datos = { habitos: Habito[]; meses: Record<string, Mes> };

const MESES = [
  'enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
  'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre',
];

/** Los rótulos del encabezado, empezando en domingo porque `getDay()` devuelve
 *  0 para el domingo y así el índice es el propio día de la semana. */
const DIAS_SEMANA = ['D', 'L', 'M', 'X', 'J', 'V', 'S'];
const DIAS_SEMANA_LARGO = ['domingo', 'lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado'];

/** Los doce de la plantilla, traducidos. Es la lista con la que se estrena la
 *  pantalla; a partir de ahí manda lo que haya guardado. */
const HABITOS_INICIALES: Habito[] = [
  { id: 'h1', nombre: 'Levantarse a las 06:00' },
  { id: 'h2', nombre: 'Meditar' },
  { id: 'h3', nombre: 'Gimnasio' },
  { id: 'h4', nombre: 'Ducha fría' },
  { id: 'h5', nombre: 'Trabajo' },
  { id: 'h6', nombre: 'Leer 10 páginas' },
  { id: 'h7', nombre: 'Aprender algo nuevo' },
  { id: 'h8', nombre: 'Sin azúcar' },
  { id: 'h9', nombre: 'Sin alcohol' },
  { id: 'h10', nombre: 'Una hora de redes' },
  { id: 'h11', nombre: 'Planificar el día' },
  { id: 'h12', nombre: 'Dormir antes de las 23:00' },
];

const MES_VACIO: Mes = { marcas: {}, animo: {}, motivacion: {} };

function clave(anio: number, mes: number): string {
  return `${anio}-${String(mes + 1).padStart(2, '0')}`;
}

function diasDelMes(anio: number, mes: number): number {
  return new Date(anio, mes + 1, 0).getDate();
}

/** Lee lo guardado, y si no hay nada —o hay algo roto— arranca con la lista de
 *  fábrica en vez de dejar la pantalla en blanco.
 *
 *  Los hábitos se normalizan de paso: hasta el 2026-08-25 llevaban un campo
 *  `icono` con un emoji, y lo guardado de entonces sigue en el disco. Se cae
 *  aquí, en la puerta, y no en cada sitio donde se dibuja un hábito. */
function leer(): Datos {
  try {
    const crudo = localStorage.getItem(ALMACEN);
    if (crudo) {
      const d = JSON.parse(crudo) as { habitos?: any[]; meses?: Record<string, Mes> };
      if (Array.isArray(d.habitos) && d.meses) {
        return {
          habitos: d.habitos.map(h => ({ id: String(h.id), nombre: String(h.nombre ?? '') })),
          meses: d.meses,
        };
      }
    }
  } catch {
    // Un almacén ilegible se sustituye; avisar de esto no le sirve a nadie.
  }
  return { habitos: HABITOS_INICIALES, meses: {} };
}

/** Un mes sin nada marcado da 0 en vez de `NaN`. */
function porcentaje(parte: number, total: number): number {
  if (!total) return 0;
  return (parte / total) * 100;
}

/** Las semanas son bloques de siete días desde el 1, no semanas naturales: es
 *  como están agrupadas las columnas de la rejilla, y la gráfica de la derecha
 *  tiene que contar lo mismo que se ve debajo. */
function semanaDe(dia: number): number {
  return Math.ceil(dia / 7);
}

/** Días seguidos marcados que llegan hasta `hasta`, contando hacia atrás.
 *
 *  Es la cifra que se mira de verdad en un seguimiento de hábitos: no cuántas
 *  veces lo hiciste este mes, sino si la cadena sigue viva. Por eso cuenta
 *  desde hoy hacia atrás y se rompe en el primer hueco. */
function racha(marcas: Record<string, boolean>, id: string, hasta: number): number {
  let n = 0;
  for (let d = hasta; d >= 1; d--) {
    if (!marcas[`${id}|${d}`]) break;
    n++;
  }
  return n;
}

function acotar(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

/** Lo que la fila de abajo —estado mental y ranking— reserva para sí antes de
 *  que la rejilla se reparta el resto. Es el `min-height` de `.hb-fila-baja`, y
 *  las dos cifras tienen que decir lo mismo. */
const ALTO_MINIMO_ABAJO = 168;
/** Los dos huecos entre las tres filas del lienzo (`gap: var(--e3)`). */
const HUECOS = 24;
/** Suelo y techo del alto de fila. Por debajo del suelo una casilla deja de ser
 *  pulsable; por encima del techo la rejilla se hincha y una fila parece un
 *  cartel. */
const FILA_MINIMA = 13;
const FILA_MAXIMA = 40;

/** Encaja la rejilla en el hueco que le toca, en vez de dejarla desbordar.
 *
 *  La pantalla no se desplaza: cabe entera o no cabe. Pero cuánto mide «entera»
 *  depende de dos cosas que el CSS no sabe —cuántos hábitos hay y cuántos días
 *  tiene el mes—, así que el alto de fila, el ancho de columna y el tamaño de la
 *  casilla se calculan aquí y se escriben como variables sobre la raíz `.hb`.
 *  Trece hábitos en un portátil y cinco en un monitor grande salen con la misma
 *  pantalla, apretada o desahogada.
 *
 *  Los topes no son decorativos: por debajo de 15 px de fila no se acierta con
 *  el ratón, y por encima de 30 la rejilla se hincha y deja calvas. Si con el
 *  mínimo aún no cupiera —cuarenta hábitos en una ventana pequeña—, lo que sobra
 *  se recorta, que es visible; una barra de scroll no lo sería.
 *
 *  Para que estos números manden de verdad, en la hoja no queda un solo relleno
 *  vertical dentro de la rejilla: un `padding: 5px` en el campo del nombre
 *  bastaba para que la fila se negara a bajar de 27 px.
 *
 *  `useLayoutEffect` y no `useEffect`: esto corrige medidas antes de pintar. Con
 *  el segundo se vería un fotograma con la rejilla del tamaño anterior. */
function useEncaje(
  raiz: React.RefObject<HTMLDivElement | null>,
  hueco: React.RefObject<HTMLDivElement | null>,
  cabecera: React.RefObject<HTMLTableSectionElement | null>,
  habitos: number,
  dias: number,
) {
  useLayoutEffect(() => {
    const nodoRaiz = raiz.current;
    const nodoHueco = hueco.current;
    if (!nodoRaiz || !nodoHueco || !habitos || !dias) return;

    const ajustar = () => {
      const ancho = nodoHueco.clientWidth;
      if (!ancho) return;

      /* Cuánto alto le toca a la rejilla. No se le pregunta a su propia caja:
         ahora esa caja mide lo que la rejilla decida (`grid-template-rows` la
         pone en `auto`), así que preguntarle sería preguntarse a uno mismo. Se
         calcula restando al lienzo lo que se llevan la fila de arriba —que tiene
         altura propia— y el mínimo que reserva la de abajo. */
      const lienzo = nodoHueco.closest('.hb-lienzo') as HTMLElement | null;
      const filaAlta = lienzo?.firstElementChild as HTMLElement | null;
      if (!lienzo || !filaAlta) return;
      const alto = lienzo.clientHeight - filaAlta.offsetHeight - ALTO_MINIMO_ABAJO - HUECOS;
      if (alto <= 0) return;

      // Lo que ocupan las dos filas de cabecera de la tabla se mide, no se
      // supone: cambia con el estilo elegido y con la fuente del sistema.
      const altoCabecera = cabecera.current?.offsetHeight ?? 44;
      // Los dos píxeles son los filos de la celda, que se suman al alto pedido:
      // sin descontarlos, doce filas se pasan de su hueco por veinticuatro.
      // Una primera estimación, que las vueltas de abajo afinan.
      const fila = acotar(Math.floor((alto - altoCabecera) / habitos) - 2, FILA_MINIMA, FILA_MAXIMA);

      // El ancho de los días lo reparte la tabla (`table-layout: fixed`); aquí
      // solo se decide cuánto se queda la columna de nombres. Cede ella primero
      // —un nombre se lee igual algo más estrecho— hasta dejar a los días en 24,
      // y de ahí en adelante ya se aprietan los días.
      const nombre = acotar(ancho - 24 * dias, 120, 232);
      const celda = (ancho - nombre) / dias;

      nodoRaiz.style.setProperty('--hb-nombre', `${nombre}px`);
      nodoRaiz.style.setProperty('--hb-marca', `${acotar(Math.floor(Math.min(fila, celda)) - 8, 8, 18)}px`);

      /* Y ahora se comprueba contra la única medida que importa: si el lienzo
         entero desborda, hay barra de scroll. La estimación de arriba no puede
         acertar al píxel —no sabe lo que miden el título de cada caja, los
         filos ni los redondeos—, así que se pone, se mira y se corrige. Leer
         `scrollHeight` fuerza el recálculo, de modo que cada vuelta ve el
         resultado de la anterior y no el de antes.

         Las vueltas están contadas: esto corre al abrir y al redimensionar, no
         en cada repintado, y un tope bajo vale más que un bucle elegante que
         algún día no converja. */
      const poner = (v: number) => nodoRaiz.style.setProperty('--hb-fila', `${v}px`);
      const desborda = () => lienzo.scrollHeight > lienzo.clientHeight;

      let actual = fila;
      poner(actual);

      // Primero se baja hasta que quepa.
      for (let vuelta = 0; vuelta < 8 && desborda() && actual > FILA_MINIMA; vuelta++) {
        const exceso = lienzo.scrollHeight - lienzo.clientHeight;
        actual = acotar(actual - Math.max(1, Math.ceil(exceso / habitos)), FILA_MINIMA, FILA_MAXIMA);
        poner(actual);
      }

      // Y después se sube mientras siga cabiendo, para no dejar media caja en
      // negro con la rejilla flotando arriba.
      for (let vuelta = 0; vuelta < 30 && actual < FILA_MAXIMA; vuelta++) {
        poner(actual + 1);
        if (desborda()) { poner(actual); break; }
        actual++;
      }
    };

    /* Dos pasadas: la primera calcula con la cabecera que había —que puede ser
       de la medida anterior—, y la segunda corrige ya con la definitiva. Sin
       esto, al estrechar la ventana la rejilla se quedaba unos píxeles por
       encima de su hueco y asomaba por debajo. */
    let pendiente = 0;
    const ajustarDosVeces = () => {
      ajustar();
      cancelAnimationFrame(pendiente);
      pendiente = requestAnimationFrame(ajustar);
    };

    ajustarDosVeces();
    const observador = new ResizeObserver(ajustarDosVeces);
    observador.observe(nodoHueco);
    // La ventana avisa además del observador: en el arranque de Tauri, el paso
    // a pantalla completa llega antes de que este nodo tenga su tamaño final.
    window.addEventListener('resize', ajustarDosVeces);
    return () => {
      cancelAnimationFrame(pendiente);
      observador.disconnect();
      window.removeEventListener('resize', ajustarDosVeces);
    };
  }, [raiz, hueco, cabecera, habitos, dias]);
}

/** Una barra de las gráficas de arriba. */
const BarraCol: React.FC<{
  valor: number;
  rotulo: string;
  titulo: string;
  finde?: boolean;
  hoy?: boolean;
  futuro?: boolean;
}> = ({ valor, rotulo, titulo, finde, hoy, futuro }) => (
  <div
    className={'hb-col' + (finde ? ' finde' : '') + (hoy ? ' hoy' : '') + (futuro ? ' futuro' : '')}
    title={titulo}
  >
    <div className="hb-col-carril">
      <div className="hb-col-relleno" style={{ height: `${Math.max(0, Math.min(100, valor))}%` }} />
    </div>
    <span className="hb-col-rotulo">{rotulo}</span>
  </div>
);

/** El anillo del mes, con el porcentaje dentro.
 *
 *  La cifra va en el hueco del centro y no a un lado: es el número más grande
 *  de la pantalla y el que contesta la única pregunta que se hace uno al
 *  abrirla. */
const Anillo: React.FC<{ hecho: number; total: number }> = ({ hecho, total }) => {
  const pct = porcentaje(hecho, total);
  const r = 46;
  const vuelta = 2 * Math.PI * r;
  const arco = (pct / 100) * vuelta;

  return (
    <div className="hb-anillo">
      <svg viewBox="0 0 120 120" aria-hidden>
        <circle className="hb-anillo-resto" cx="60" cy="60" r={r} />
        <circle
          className="hb-anillo-hecho"
          cx="60" cy="60" r={r}
          strokeDasharray={`${arco} ${vuelta - arco}`}
          /* Arranca arriba y no a las tres en punto: un anillo que empieza en el
             costado se lee como si le faltara un trozo. */
          transform="rotate(-90 60 60)"
        />
      </svg>
      <div className="hb-anillo-centro">
        <b>{Math.round(pct)}<span className="hb-anillo-pct">%</span></b>
        <span className="hb-anillo-pie">del mes</span>
      </div>
    </div>
  );
};

/** El estado mental: las dos filas de cifras y, debajo, las dos áreas.
 *
 *  Las cifras y el dibujo comparten anchura de columna a propósito —cada punto
 *  cae en el centro de su casilla—, que es lo que permite leer un pico de la
 *  línea y bajar el dedo hasta el número que lo produjo. */
const EstadoMental: React.FC<{
  dias: number;
  animo: Record<string, number>;
  motivacion: Record<string, number>;
  hoy: number | null;
  onCambiar: (serie: 'animo' | 'motivacion', dia: number, valor: number | null) => void;
}> = ({ dias, animo, motivacion, hoy, onCambiar }) => {
  const ancho = dias * 10;

  const camino = (fuente: Record<string, number>) => {
    const puntos: [number, number][] = [];
    for (let d = 1; d <= dias; d++) {
      const v = fuente[d];
      if (typeof v === 'number') {
        puntos.push([(d - 1) * 10 + 5, 100 - (Math.max(0, Math.min(10, v)) / 10) * 100]);
      }
    }
    return puntos;
  };

  const linea = (puntos: [number, number][]) =>
    puntos.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ');

  const area = (puntos: [number, number][]) =>
    puntos.length < 2 ? '' : `${linea(puntos)} L${puntos[puntos.length - 1][0]},100 L${puntos[0][0]},100 Z`;

  const pAnimo = camino(animo);
  const pMoti = camino(motivacion);

  const fila = (serie: 'animo' | 'motivacion', fuente: Record<string, number>, etiqueta: string) => (
    <div className="hb-mental-fila">
      <span className={`hb-mental-etiqueta ${serie}`}>
        <i className="hb-muestra" aria-hidden />
        {etiqueta}
      </span>
      <div className="hb-mental-cifras" style={{ gridTemplateColumns: `repeat(${dias}, 1fr)` }}>
        {Array.from({ length: dias }, (_, i) => i + 1).map(d => (
          <input
            key={d}
            className={'hb-mental-celda' + (d === hoy ? ' hoy' : '')}
            value={fuente[d] ?? ''}
            inputMode="numeric"
            maxLength={2}
            title={`Día ${d} · ${etiqueta.toLowerCase()} de 0 a 10`}
            onChange={e => {
              const t = e.target.value.trim();
              if (t === '') return onCambiar(serie, d, null);
              const n = Number(t);
              // Fuera de 0..10 no se guarda nada: la escala del dibujo es esa, y
              // un 47 suelto aplastaría las dos líneas contra el suelo.
              if (Number.isFinite(n) && n >= 0 && n <= 10) onCambiar(serie, d, n);
            }}
          />
        ))}
      </div>
    </div>
  );

  return (
    <div className="hb-caja hb-mental">
      <div className="hb-caja-titulo">
        Estado mental
        <span className="hb-caja-pie">de 0 a 10, un número por día</span>
      </div>
      {fila('animo', animo, 'Ánimo')}
      {fila('motivacion', motivacion, 'Motivación')}
      <div className="hb-mental-grafica">
        {/* La escala a la izquierda: sin ella, dos áreas superpuestas dicen
            «arriba» y «abajo» pero no cuánto. */}
        <div className="hb-mental-escala"><span>10</span><span>5</span><span>0</span></div>
        <svg viewBox={`0 0 ${ancho} 100`} preserveAspectRatio="none" aria-hidden>
          <line className="hb-guia" x1="0" y1="50" x2={ancho} y2="50" vectorEffect="non-scaling-stroke" />
          <path className="hb-area suave" d={area(pMoti)} />
          <path className="hb-linea suave" d={linea(pMoti)} vectorEffect="non-scaling-stroke" />
          <path className="hb-area" d={area(pAnimo)} />
          <path className="hb-linea" d={linea(pAnimo)} vectorEffect="non-scaling-stroke" />
        </svg>
      </div>
    </div>
  );
};

export const Habitos: React.FC<{ onCerrar: () => void; estilo: EstiloHabitos }> = ({
  onCerrar, estilo,
}) => {
  const hoy = useMemo(() => new Date(), []);
  const [anio, setAnio] = useState(hoy.getFullYear());
  const [mes, setMes] = useState(hoy.getMonth());
  const [datos, setDatos] = useState<Datos>(leer);

  // Cada cambio se guarda entero. Son unos pocos kilobytes y el mes cabe en una
  // escritura: llevar un diario de cambios aquí sería fontanería sin cliente.
  useEffect(() => {
    try {
      localStorage.setItem(ALMACEN, JSON.stringify(datos));
    } catch {
      // Almacén lleno o bloqueado: la pantalla sigue funcionando en memoria.
    }
  }, [datos]);

  const k = clave(anio, mes);
  const dias = diasDelMes(anio, mes);
  const mesActual = datos.meses[k] ?? MES_VACIO;
  const habitos = datos.habitos;

  const esMesDeHoy = anio === hoy.getFullYear() && mes === hoy.getMonth();
  const diaDeHoy = esMesDeHoy ? hoy.getDate() : null;
  /** Hasta dónde tiene sentido contar: hoy en el mes en curso, el último día en
   *  cualquier otro. Sin esto, la racha de un mes pasado se cortaría siempre en
   *  el día de hoy aunque el mes entero estuviera lleno.
   *
   *  Y manda también sobre el objetivo: a mitad de mes, «faltan» no puede
   *  incluir los días que aún no han llegado — el anillo no llegaría nunca al
   *  100 % aunque fueras perfecto hasta hoy, y el objetivo contaría días que
   *  no existen. */
  const ultimoContable = diaDeHoy ?? dias;

  const tocarMes = useCallback((cambio: (m: Mes) => Mes) => {
    setDatos(d => {
      const k = clave(anio, mes);
      const previo = d.meses[k] ?? MES_VACIO;
      return { ...d, meses: { ...d.meses, [k]: cambio(previo) } };
    });
  }, [anio, mes]);

  const poner = useCallback((idHabito: string, dia: number, valor: boolean) => {
    tocarMes(m => {
      const c = `${idHabito}|${dia}`;
      if (!!m.marcas[c] === valor) return m;
      const marcas = { ...m.marcas };
      if (valor) marcas[c] = true; else delete marcas[c];
      return { ...m, marcas };
    });
  }, [tocarMes]);

  /** Pintar arrastrando: se pulsa en una casilla y, sin soltar, las que se
   *  crucen toman ESE mismo valor —no el contrario del suyo—. Marcar catorce
   *  días de gimnasio a catorce clics era el gesto que más se repetía.
   *
   *  El valor en curso vive en una `ref` y no en el estado: cambia en cada
   *  `pointerenter` y no debe repintar nada. */
  const pintando = useRef<boolean | null>(null);
  useEffect(() => {
    const soltar = () => { pintando.current = null; };
    window.addEventListener('pointerup', soltar);
    window.addEventListener('pointercancel', soltar);
    return () => {
      window.removeEventListener('pointerup', soltar);
      window.removeEventListener('pointercancel', soltar);
    };
  }, []);

  /** La cabecera de un día marca su columna entera, y la desmarca si ya estaba
   *  completa. Es el atajo del «hoy lo he hecho todo». */
  const alternarColumna = useCallback((dia: number) => {
    tocarMes(m => {
      const lleno = habitos.length > 0 && habitos.every(h => m.marcas[`${h.id}|${dia}`]);
      const marcas = { ...m.marcas };
      for (const h of habitos) {
        const c = `${h.id}|${dia}`;
        if (lleno) delete marcas[c]; else marcas[c] = true;
      }
      return { ...m, marcas };
    });
  }, [tocarMes, habitos]);

  const cambiarMental = useCallback((serie: 'animo' | 'motivacion', dia: number, valor: number | null) => {
    tocarMes(m => {
      const fuente = { ...m[serie] };
      if (valor === null) delete fuente[dia]; else fuente[dia] = valor;
      return { ...m, [serie]: fuente };
    });
  }, [tocarMes]);

  const renombrar = (id: string, nombre: string) => {
    setDatos(d => ({ ...d, habitos: d.habitos.map(h => (h.id === id ? { ...h, nombre } : h)) }));
  };

  const anadir = () => {
    setDatos(d => ({ ...d, habitos: [...d.habitos, { id: `h${Date.now()}`, nombre: 'Hábito nuevo' }] }));
  };

  /** Quitar un hábito se lleva sus marcas de todos los meses. Dejarlas
   *  guardadas «por si acaso» hace que un hábito nuevo con el mismo id reviva
   *  casillas de hace medio año. */
  const quitar = (id: string) => {
    setDatos(d => {
      const meses: Record<string, Mes> = {};
      for (const [mk, m] of Object.entries(d.meses)) {
        const marcas: Record<string, boolean> = {};
        for (const [c, v] of Object.entries(m.marcas)) {
          if (!c.startsWith(`${id}|`)) marcas[c] = v;
        }
        meses[mk] = { ...m, marcas };
      }
      return { habitos: d.habitos.filter(h => h.id !== id), meses };
    });
  };

  const moverMes = (paso: number) => {
    const d = new Date(anio, mes + paso, 1);
    setAnio(d.getFullYear());
    setMes(d.getMonth());
  };

  const irAHoy = () => {
    setAnio(hoy.getFullYear());
    setMes(hoy.getMonth());
  };

  // ── Todo lo que se enseña a la derecha sale de aquí ────────────────────────
  const cuentas = useMemo(() => {
    const porHabito = new Map<string, number>();
    const rachas = new Map<string, number>();
    const porDia = new Map<number, number>();
    const porSemana = new Map<number, number>();
    let total = 0;

    for (const h of habitos) {
      let n = 0;
      for (let d = 1; d <= dias; d++) {
        if (mesActual.marcas[`${h.id}|${d}`]) {
          // Las cifras de «cuánto llevas» llegan hasta hoy; las gráficas por
          // día y por semana enseñan el mes entero, futuro incluido.
          if (d <= ultimoContable) n++;
          porDia.set(d, (porDia.get(d) ?? 0) + 1);
          porSemana.set(semanaDe(d), (porSemana.get(semanaDe(d)) ?? 0) + 1);
        }
      }
      porHabito.set(h.id, n);
      rachas.set(h.id, racha(mesActual.marcas, h.id, ultimoContable));
      total += n;
    }
    return { porHabito, rachas, porDia, porSemana, total };
  }, [habitos, mesActual, dias, ultimoContable]);

  const objetivo = habitos.length * ultimoContable;
  const hecho = cuentas.total;
  const restante = Math.max(0, objetivo - hecho);

  const semanas = useMemo(() => {
    const lista: { nombre: string; pct: number; hecho: number; tope: number; desde: number; hasta: number }[] = [];
    for (let s = 1; s <= semanaDe(dias); s++) {
      const desde = (s - 1) * 7 + 1;
      const hasta = Math.min(dias, s * 7);
      // La semana en curso no pide los días que le faltan por llegar: su tope
      // es lo que ya pasó de ella, o su barra nunca se llenaría.
      const tope = Math.max(0, Math.min(hasta, ultimoContable) - desde + 1) * habitos.length;
      const n = cuentas.porSemana.get(s) ?? 0;
      lista.push({ nombre: `S${s}`, pct: porcentaje(n, tope), hecho: n, tope, desde, hasta });
    }
    return lista;
  }, [dias, habitos.length, cuentas, ultimoContable]);

  const ranking = useMemo(
    () => [...habitos]
      .sort((a, b) => (cuentas.porHabito.get(b.id) ?? 0) - (cuentas.porHabito.get(a.id) ?? 0))
      .slice(0, 10),
    [habitos, cuentas],
  );

  const anios = useMemo(() => {
    const base = hoy.getFullYear();
    return Array.from({ length: 11 }, (_, i) => base - 5 + i);
  }, [hoy]);

  const listaDias = useMemo(() => Array.from({ length: dias }, (_, i) => i + 1), [dias]);

  const raiz = useRef<HTMLDivElement>(null);
  const huecoRejilla = useRef<HTMLDivElement>(null);
  const cabeceraTabla = useRef<HTMLTableSectionElement>(null);
  useEncaje(raiz, huecoRejilla, cabeceraTabla, habitos.length, dias);

  const finde = (d: number) => {
    const s = new Date(anio, mes, d).getDay();
    return s === 0 || s === 6;
  };
  const claseDia = (d: number) =>
    (finde(d) ? ' finde' : '') +
    (d === diaDeHoy ? ' hoy' : '') +
    (diaDeHoy !== null && d > diaDeHoy ? ' futuro' : '');

  return (
    <div className="hb" data-estilo={estilo} ref={raiz}>
      <header className="hb-cabecera">
        <h2>Hábitos</h2>
        <span className="hb-cabecera-pie">
          {hecho} de {objetivo} casillas {esMesDeHoy ? 'hasta hoy' : 'este mes'}
        </span>
        <button className="hb-volver" onClick={onCerrar}>Volver a la llamada</button>
      </header>

      <div className="hb-lienzo">
        {/* ── Fila de arriba ─────────────────────────────────────────────── */}
        <div className="hb-fila hb-fila-alta">
          <div className="hb-caja hb-mes">
            <div className="hb-mes-nav">
              <button onClick={() => moverMes(-1)} title="El mes anterior" aria-label="El mes anterior">‹</button>
              <div className="hb-mes-nombre">
                <b>{MESES[mes]}</b>
                <span>{anio}</span>
              </div>
              <button onClick={() => moverMes(1)} title="El mes siguiente" aria-label="El mes siguiente">›</button>
            </div>
            <div className="hb-mes-saltos">
              <select value={mes} onChange={e => setMes(Number(e.target.value))} title="Ir a un mes">
                {MESES.map((m, i) => <option key={m} value={i}>{m}</option>)}
              </select>
              <select value={anio} onChange={e => setAnio(Number(e.target.value))} title="Ir a un año">
                {anios.map(a => <option key={a} value={a}>{a}</option>)}
              </select>
            </div>
            {/* El botón solo existe cuando sirve de algo: en el mes en curso no
                lleva a ninguna parte y sería un mando muerto. */}
            {!esMesDeHoy && <button className="hb-hoy" onClick={irAHoy}>Volver a hoy</button>}
          </div>

          <div className="hb-caja hb-grafica">
            <div className="hb-caja-titulo">
              Cada día
              <span className="hb-caja-pie">cuántos de {habitos.length}</span>
            </div>
            <div className="hb-barras">
              {listaDias.map(d => {
                const n = cuentas.porDia.get(d) ?? 0;
                return (
                  <BarraCol
                    key={d}
                    valor={porcentaje(n, habitos.length)}
                    rotulo={String(d)}
                    finde={finde(d)}
                    hoy={d === diaDeHoy}
                    futuro={diaDeHoy !== null && d > diaDeHoy}
                    titulo={`${d} de ${MESES[mes]}, ${DIAS_SEMANA_LARGO[new Date(anio, mes, d).getDay()]} · ${n} de ${habitos.length}`}
                  />
                );
              })}
            </div>
          </div>

          <div className="hb-caja hb-grafica hb-grafica-semanal">
            <div className="hb-caja-titulo">
              Cada semana
              <span className="hb-caja-pie">bloques de siete días</span>
            </div>
            <div className="hb-barras hb-barras-anchas">
              {semanas.map(s => (
                <BarraCol
                  key={s.nombre}
                  valor={s.pct}
                  rotulo={s.nombre}
                  titulo={`Días ${s.desde} a ${s.hasta} · ${s.hecho} de ${s.tope} (${Math.round(s.pct)} %)`}
                />
              ))}
            </div>
          </div>

          <div className="hb-caja hb-resumen">
            <div className="hb-caja-titulo">El mes</div>
            <div className="hb-resumen-cuerpo">
              <Anillo hecho={hecho} total={objetivo} />
              <dl className="hb-cifras">
                <div><dt>Hechos</dt><dd>{hecho}</dd></div>
                <div><dt>Faltan</dt><dd>{restante}</dd></div>
                <div><dt>Objetivo</dt><dd>{objetivo}</dd></div>
              </dl>
            </div>
          </div>
        </div>

        {/* ── Fila del centro: la rejilla y el análisis ───────────────────── */}
        <div className="hb-fila hb-fila-rejilla">
          <div className="hb-caja hb-rejilla">
            <div className="hb-caja-titulo hb-rejilla-cabeza">
              Mis hábitos
              <span className="hb-caja-pie">
                pulsa una casilla —o arrastra para varias—; pulsa el número de un día para su columna
              </span>
              <button className="hb-anadir" onClick={anadir}>Añadir hábito</button>
            </div>
            <div className="hb-rejilla-lienzo" ref={huecoRejilla}>
              <table className="hb-tabla">
                <thead ref={cabeceraTabla}>
                  <tr>
                    <th className="hb-th-nombre" rowSpan={2}>Hábito</th>
                    {listaDias.map(d => (
                      <th key={d} className={'hb-th-dia' + claseDia(d)}>
                        {DIAS_SEMANA[new Date(anio, mes, d).getDay()]}
                      </th>
                    ))}
                  </tr>
                  <tr>
                    {listaDias.map(d => (
                      <th key={d} className={'hb-th-numero' + claseDia(d)}>
                        <button onClick={() => alternarColumna(d)} title={`Marcar o desmarcar todo el día ${d}`}>
                          {d}
                        </button>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {habitos.map(h => (
                    <tr key={h.id}>
                      <td className="hb-td-nombre">
                        <div className="hb-habito">
                          <input
                            className="hb-habito-nombre"
                            value={h.nombre}
                            onChange={e => renombrar(h.id, e.target.value)}
                            aria-label="Nombre del hábito"
                          />
                          <button
                            className="hb-quitar"
                            title={`Quitar «${h.nombre}»`}
                            aria-label={`Quitar ${h.nombre}`}
                            onClick={() => quitar(h.id)}
                          >
                            ×
                          </button>
                        </div>
                      </td>
                      {listaDias.map(d => {
                        const marcado = !!mesActual.marcas[`${h.id}|${d}`];
                        return (
                          <td key={d} className={'hb-td-celda' + claseDia(d)}>
                            <button
                              className={'hb-casilla' + (marcado ? ' marcada' : '')}
                              role="checkbox"
                              aria-checked={marcado}
                              aria-label={`${h.nombre}, día ${d}`}
                              onPointerDown={e => {
                                // El arrastre pinta; sin esto el navegador
                                // empieza a seleccionar texto de la tabla y la
                                // rejilla se queda azul a medio camino.
                                e.preventDefault();
                                pintando.current = !marcado;
                                poner(h.id, d, !marcado);
                              }}
                              onPointerEnter={() => {
                                if (pintando.current !== null) poner(h.id, d, pintando.current);
                              }}
                              /* El teclado no arrastra: para él, pulsar es
                                 alternar, que es lo que hace una casilla. */
                              onKeyDown={e => {
                                if (e.key === ' ' || e.key === 'Enter') {
                                  e.preventDefault();
                                  poner(h.id, d, !marcado);
                                }
                              }}
                            />
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="hb-caja hb-analisis">
            <div className="hb-caja-titulo">
              Hábito a hábito
              <span className="hb-caja-pie">
                racha: días seguidos hasta {diaDeHoy !== null ? 'hoy' : `el ${dias}`}
              </span>
            </div>
            <div className="hb-analisis-lienzo">
              <table className="hb-tabla-analisis">
                <thead>
                  <tr>
                    <th className="hb-col-hab">Hábito</th>
                    <th>Hechos</th>
                    <th className="hb-col-barra">Progreso</th>
                    <th>Racha</th>
                  </tr>
                </thead>
                <tbody>
                  {habitos.map(h => {
                    const n = cuentas.porHabito.get(h.id) ?? 0;
                    const pct = porcentaje(n, ultimoContable);
                    const r = cuentas.rachas.get(h.id) ?? 0;
                    return (
                      <tr key={h.id}>
                        <td className="hb-col-hab" title={h.nombre}>{h.nombre}</td>
                        <td className="hb-num">{n}<span className="hb-de">/{ultimoContable}</span></td>
                        <td className="hb-col-barra">
                          <div className="hb-barra-carril" title={`${Math.round(pct)} %`}>
                            <span style={{ width: `${pct}%` }} />
                          </div>
                        </td>
                        <td className={'hb-num' + (r >= 7 ? ' viva' : '')}>{r}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* ── Fila de abajo: estado mental y ranking ──────────────────────── */}
        <div className="hb-fila hb-fila-baja">
          <EstadoMental
            dias={dias}
            animo={mesActual.animo}
            motivacion={mesActual.motivacion}
            hoy={diaDeHoy}
            onCambiar={cambiarMental}
          />

          <div className="hb-caja hb-top">
            <div className="hb-caja-titulo">
              Los que mejor van
              <span className="hb-caja-pie">este mes</span>
            </div>
            <ol className="hb-top-lista">
              {ranking.map((h, i) => {
                const n = cuentas.porHabito.get(h.id) ?? 0;
                return (
                  <li key={h.id}>
                    <span className="hb-top-puesto">{i + 1}</span>
                    <span className="hb-top-nombre" title={h.nombre}>{h.nombre}</span>
                    <span className="hb-top-barra" aria-hidden>
                      <span style={{ width: `${porcentaje(n, ultimoContable)}%` }} />
                    </span>
                    <span className="hb-top-cuenta">{n}</span>
                  </li>
                );
              })}
            </ol>
          </div>
        </div>
      </div>
    </div>
  );
};
