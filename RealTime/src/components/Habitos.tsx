/**
 * El seguimiento de hábitos: la plantilla original, en la ventana de la app.
 *
 * ── Qué es esto y por qué está así ──────────────────────────────────────────
 *
 * Esto salió de una hoja de cálculo —un «habit tracker» de plantilla— que el
 * señor Persus ya usaba. Se rehízo tres veces buscando algo mejor y la tercera
 * acabó en una sola tabla sin cajas; él miró la foto de la plantilla y dijo la
 * frase que cierra el asunto: *«mejor vuélvelo al diseño original de esta foto
 * […] mantén los colores y todo, solo quita los emoticonos»*.
 *
 * Así que el reparto es el de la plantilla, literal, y lo que se conserva de
 * las vueltas anteriores es **solo lo que no se ve**: el manejo. La lección,
 * apuntada para no repetirla: una plantilla que alguien ya usa a diario no es
 * un borrador que mejorar, es un requisito. Lo que sobraba no era el reparto.
 *
 * ── El reparto, que es el de la foto ────────────────────────────────────────
 *
 *   ┌──────────┬─────────────────────────┬──────────────────┐
 *   │ título   │ progreso diario         │ objetivo/hechas  │
 *   │ ajustes  │ progreso semanal        │ resumen (rosco)  │
 *   ├──────────┼─────────────────────────┼──────────────────┤
 *   │ mis      │ semana 1 … semana 5     │ análisis         │
 *   │ hábitos  │ (la rejilla de casillas)│ por hábito       │
 *   ├──────────┴─────────────────────────┼──────────────────┤
 *   │ estado mental (ánimo, motivación)  │ los diez mejores │
 *   └────────────────────────────────────┴──────────────────┘
 *
 * La columna de la izquierda mide lo mismo en las tres filas (`--hb-izq`): el
 * bloque del título, la columna de nombres de la rejilla y las etiquetas del
 * estado mental están a plomo, como en la hoja. Es lo primero que se nota si se
 * descuadra, y por eso la medida es una sola variable y no tres.
 *
 * ── Lo único que NO viene de la plantilla ───────────────────────────────────
 *
 * Sin emoticonos. En la hoja cada hábito llevaba uno al lado del nombre y el
 * «top 10» los repetía. En una rejilla de 31 columnas es ruido de colores
 * compitiendo con lo único que importa: si la casilla está marcada o no. Un
 * `icono` guardado de antes se descarta al leer (ver `lib/habitos.ts`).
 *
 * ── Y lo que se conserva de las vueltas intermedias, porque no se ve ────────
 *
 *  - **La casilla es la celda entera**, no un cuadrito de 16 px en el centro de
 *    un hueco de 26 × 27. Mismo aspecto, casi el triple de diana.
 *  - **Cruceta**: al pasar el ratón se encienden la fila y la columna. En un mes
 *    de 31 columnas es la diferencia entre marcar el 17 y marcar el 18.
 *  - **Teclado**: flechas para moverse, espacio para marcar, `Shift`+flecha para
 *    pintar sin soltar — el arrastre del ratón, para quien no lo usa.
 *  - **`Ctrl+Z`**. Se pinta arrastrando, y un arrastre torcido borra una semana
 *    en un gesto. Un arrastre entero cuenta como un solo paso.
 *  - **Marcar un hábito entero** hasta hoy, **marcar un día entero**, y
 *    **reordenar** la lista arrastrando por el asa.
 *  - **La racha cruza el mes** (ver `racha()` en `lib/habitos.ts`): contada
 *    dentro del mes valía como mucho 1 cada día 1.
 *
 * Todo se edita; ninguna cifra se teclea. Las cuentas de la derecha —objetivo,
 * hechas, faltan, porcentajes, el rosco, las barras y el ranking— salen de las
 * casillas, porque dos sitios donde escribir el mismo dato acaban discrepando.
 *
 * Dónde vive el dato: en `lib/habitos.ts`, sobre el `localStorage` de la
 * ventana. Esta pantalla es una de sus dos lectoras; la otra es Perseo, que lo
 * consulta en llamada con la herramienta `consultar_habitos`.
 */
import React, {
  useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState,
} from 'react';

import { invoke } from '@tauri-apps/api/core';

import type { EstiloHabitos } from '../lib/config';
import {
  DIAS_SEMANA, DIAS_SEMANA_LARGO, MESES, MES_VACIO,
  acotar, clave, diasDelMes, foto, guardar, leer, porcentaje, racha, resumen, semanaDe,
  type Datos, type Mes,
} from '../lib/habitos';
import '../styles/habitos.css';

/** Suelo y techo del alto de fila. Por debajo del suelo una casilla deja de ser
 *  pulsable con el ratón; por encima del techo la rejilla se hincha y deja
 *  calvas entre las casillas y sus filos. */
const FILA_MINIMA = 14;
/* El techo es alto a propósito: con cinco hábitos en vez de doce hay alto de
   sobra, y una rejilla desahogada se maneja mejor. Por encima de esto no sube,
   porque una fila de sesenta píxeles para una casilla parece un cartel — y
   además el ancho de columna suele topar antes: el cuadrito es un cuadrado y
   manda la medida más corta de las dos. */
const FILA_MAXIMA = 44;
/** Lo que reservan la fila de arriba y la de abajo del lienzo, y los dos huecos
 *  entre las tres. Entran en la cuenta del encaje, así que tienen que decir lo
 *  mismo que la hoja (`.hb-lienzo`): si se cambian ahí, se cambian aquí. */
const ALTO_ARRIBA = 128;
const ALTO_ABAJO = 186;
const HUECOS = 16;
/** Cuántos pasos atrás guarda `Ctrl+Z`. Un arrastre largo es UN paso —el estado
 *  entero se apila una vez por gesto—, así que veinte son veinte gestos. */
const PASOS_ATRAS = 20;
/** Cuánto se espera, sin que nadie toque nada, antes de mandarle la copia al
 *  núcleo. Pintar arrastrando cambia el estado decenas de veces en un segundo y
 *  cada cambio sería un POST; con la espera, un arrastre entero manda uno. */
const ESPERA_ESPEJO = 2000;

/**
 * Encaja la rejilla en el hueco que le toca, en vez de dejarla desbordar.
 *
 * La pantalla no se desplaza: cabe entera o no cabe. Pero cuánto mide «entera»
 * depende de dos cosas que el CSS no sabe —cuántos hábitos hay y cuántos días
 * tiene el mes—, así que el alto de fila, el lado del cuadrito y el alto de la
 * cabecera de la tabla se calculan aquí y se escriben como variables sobre la
 * raíz `.hb`. Trece hábitos en un portátil y cinco en un monitor grande salen
 * con la misma pantalla, apretada o desahogada.
 *
 * `--hb-cabeza` no es decorativa: la tabla del análisis, a la derecha, tiene que
 * empezar sus filas a la misma altura que la rejilla o las dos listas de doce
 * dejan de estar a plomo, que es lo primero que se ve mal en esta pantalla.
 *
 * `useLayoutEffect` y no `useEffect`: esto corrige medidas antes de pintar. Con
 * el segundo se vería un fotograma con la rejilla del tamaño anterior.
 */
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

      /* Cuánto alto le toca a la rejilla. No se le pregunta a su propia caja
         —ahora esa caja mide lo que la rejilla decida—, así que se calcula
         restando al lienzo lo que se llevan la fila de arriba y la de abajo. */
      const lienzo = nodoHueco.closest('.hb-lienzo') as HTMLElement | null;
      if (!lienzo) return;
      const cajaLienzo = getComputedStyle(lienzo);
      const relleno = parseFloat(cajaLienzo.paddingTop) + parseFloat(cajaLienzo.paddingBottom);
      /* Cuánto alto puede pedir la rejilla como mucho. No es lo que va a ocupar:
         su fila de la retícula es `auto`, así que si con el techo de fila le
         sobra sitio, la fila encoge y el sobrante se lo lleva el estado mental.
         Esto solo pone el LÍMITE, para que con veinte hábitos no desborde. */
      const alto = lienzo.clientHeight - relleno - ALTO_ARRIBA - ALTO_ABAJO - HUECOS;
      if (alto <= 0) return;

      // Lo que ocupan las tres filas de cabecera —semanas, letras y números— se
      // mide, no se supone: cambia con el estilo elegido y con la fuente.
      const altoCabecera = cabecera.current?.offsetHeight ?? 58;
      nodoRaiz.style.setProperty('--hb-cabeza', `${altoCabecera}px`);

      /* Y lo que le toca a la cabecera del análisis, que arranca más abajo: su
         caja lleva encima la barra del título («Análisis») y la rejilla no —su
         rótulo, «Mis hábitos», va DENTRO de la tabla—. Sin descontarla, las dos
         listas de doce empezaban con veintitrés píxeles de desfase, que en dos
         tablas puestas la una al lado de la otra se ve a la primera. */
      const tituloAnalisis = (lienzo.querySelector('.hb-analisis .hb-caja-titulo') as HTMLElement | null)
        ?.offsetHeight ?? 23;
      nodoRaiz.style.setProperty(
        '--hb-cabeza-ana', `${Math.max(16, altoCabecera - tituloAnalisis)}px`,
      );

      /* El alto de fila, calculado y no tanteado.
         
         Antes esto era un bucle: se ponía una altura, se miraba si la caja
         desbordaba y se corregía. Dejó de valer en cuanto la fila del centro
         pasó a medir lo que necesita (`auto`), porque entonces la pregunta se
         muerde la cola — encoger la fila encoge la tabla, que encoge la caja,
         que vuelve a desbordar—. Y medir el lienzo entero tampoco vale: sus
         filas están acotadas, así que nunca desborda y el bucle subía hasta el
         techo escondiendo lo que sobraba tras una barra de desplazamiento.

         Con el hueco disponible conocido, la cuenta es directa. El píxel por
         fila es el filo de la celda, que se suma al alto pedido. */
      const altoPie = (nodoHueco.parentElement?.querySelector('.hb-rejilla-pie') as HTMLElement | null)
        ?.offsetHeight ?? 30;

      /* Y el hueco que hay que reservarle al análisis por abajo para que su
         cajón de desplazamiento mida EXACTAMENTE lo mismo que el de la rejilla.
         La rejilla cede alto por los botones del pie y el análisis por su barra
         de título; como no miden lo mismo, sin igualarlos los dos cajones
         desplazan distinto y al arrastrar la rejilla con treinta hábitos el
         análisis se quedaba veinte píxeles atrás — es decir, una fila y media
         de desfase entre un hábito y sus cifras. */
      nodoRaiz.style.setProperty(
        '--hb-hueco-ana', `${Math.max(0, altoPie - tituloAnalisis)}px`,
      );
      const fila = acotar(
        Math.floor((alto - altoCabecera - altoPie) / habitos) - 1,
        FILA_MINIMA, FILA_MAXIMA,
      );
      nodoRaiz.style.setProperty('--hb-fila', `${fila}px`);

      /* El lado del cuadrito. Manda el más corto de los dos —alto de fila o
         ancho de columna—, porque es un cuadrado y tiene que caber en los dos
         sentidos. El hueco es proporcional (18 %) con un suelo de 4 px, así que
         crece con la rejilla en vez de quedarse plantado: un cuadrito de 18 px
         en una celda de 37 dejaba más hueco muerto que marca, y trescientas
         setenta y dos celdas medio vacías se leen como puntos sueltos. */
      const celda = (ancho - parseFloat(getComputedStyle(nodoRaiz).getPropertyValue('--hb-izq') || '152')) / dias;
      const lado = Math.min(fila, celda);
      nodoRaiz.style.setProperty(
        '--hb-marca',
        `${acotar(Math.floor(lado - Math.max(4, lado * 0.18)), 8, 26)}px`,
      );
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

/** Una barra de las dos gráficas de arriba.
 *
 *  El carril se dibuja siempre, tenga la barra la altura que tenga. Sin él, un
 *  mes recién empezado enseñaba dos cajas tituladas y absolutamente vacías, y
 *  una caja vacía no se lee como «llevas cero»: se lee como «esto está roto». */
const Barra: React.FC<{
  valor: number; rotulo: string; titulo: string;
  finde?: boolean; hoy?: boolean; futuro?: boolean;
}> = ({ valor, rotulo, titulo, finde, hoy, futuro }) => (
  <div
    className={'hb-col' + (finde ? ' finde' : '') + (hoy ? ' hoy' : '') + (futuro ? ' futuro' : '')}
    title={titulo}
  >
    <div className="hb-col-carril">
      <div className="hb-col-relleno" style={{ height: `${acotar(valor, 0, 100)}%` }} />
    </div>
    <span className="hb-col-rotulo">{rotulo}</span>
  </div>
);

/** El rosco del resumen, con los dos porcentajes fuera y el grande dentro.
 *
 *  Como en la hoja: el trozo hecho y el que falta, con su cifra cada uno. Un
 *  anillo que empieza a las tres en punto se lee como si le faltara un trozo,
 *  así que arranca arriba (`rotate(-90)`). */
const Rosco: React.FC<{ hecho: number; total: number }> = ({ hecho, total }) => {
  const pct = porcentaje(hecho, total);
  const r = 42;
  const vuelta = 2 * Math.PI * r;
  const arco = (pct / 100) * vuelta;

  return (
    <div className="hb-rosco">
      {/* El anillo y su cifra van en el MISMO cajón, y la cifra se centra sobre
          él con `inset: 0`. Antes la cifra se colocaba con un ancho calculado a
          mano contra la caja entera —`calc(100% - 16px - 62px)`— y bastaba con
          que la leyenda de al lado cambiara de ancho para que el número se
          descolgara del agujero del rosco. */}
      <div className="hb-rosco-anillo">
        <svg viewBox="0 0 120 120" aria-hidden>
          <circle className="hb-rosco-resto" cx="60" cy="60" r={r} />
          <circle
            className="hb-rosco-hecho"
            cx="60" cy="60" r={r}
            strokeDasharray={`${arco} ${vuelta - arco}`}
            transform="rotate(-90 60 60)"
          />
        </svg>
        <div className="hb-rosco-centro">
          <b>{Math.round(pct)}<span>%</span></b>
        </div>
      </div>
      <div className="hb-rosco-pies">
        <span className="hecho">{Math.round(pct)} % hechas</span>
        <span className="resto">{Math.round(100 - pct)} % faltan</span>
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

  useEffect(() => { guardar(datos); }, [datos]);

  /** Y una copia al núcleo, para que Perseo lo sepa también fuera de la llamada.
   *
   *  Los hábitos viven en el `localStorage` de esta ventana, que es lo correcto
   *  —marcar una casilla no puede depender de que el núcleo esté encendido— y a
   *  la vez deja fuera a media casa: el Perseo de la llamada corre AQUÍ y los
   *  lee sin más, pero el chat escrito y los agentes son Python y no ven dentro
   *  de un navegador. Esto es el puente, y va en una sola dirección.
   *
   *  Si falla, se calla: el núcleo apagado es un estado normal de esta app, y un
   *  aviso rojo por no haber podido mandar una copia que nadie ha pedido sería
   *  alarmar por nada. El cambio siguiente la manda otra vez. */
  useEffect(() => {
    const t = setTimeout(() => {
      invoke('habitos_espejo', { texto: resumen(datos), foto: foto(datos) })
        .catch(e => console.debug('[Hábitos] El núcleo no recogió la copia:', e));
    }, ESPERA_ESPEJO);
    return () => clearTimeout(t);
  }, [datos]);

  /** Lo que había antes de cada gesto, para `Ctrl+Z`.
   *
   *  La pila se llena en el manejador y no dentro del actualizador de estado:
   *  React invoca los actualizadores dos veces en modo estricto, y apilar ahí
   *  dentro dejaría la mitad de los pasos duplicados. Por eso hace falta también
   *  el espejo `datosAhora`: el manejador necesita ver el estado de este
   *  instante, no el que se cerró cuando se creó la función. */
  const historia = useRef<Datos[]>([]);
  const datosAhora = useRef(datos);
  datosAhora.current = datos;
  const [hayQueDeshacer, setHayQueDeshacer] = useState(false);

  const aplicar = useCallback((cambio: (d: Datos) => Datos) => {
    historia.current.push(datosAhora.current);
    if (historia.current.length > PASOS_ATRAS) historia.current.shift();
    setHayQueDeshacer(true);
    setDatos(cambio);
  }, []);

  const deshacer = useCallback(() => {
    const previo = historia.current.pop();
    if (!previo) return;
    setDatos(previo);
    setHayQueDeshacer(historia.current.length > 0);
  }, []);

  useEffect(() => {
    const teclas = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        deshacer();
      }
    };
    window.addEventListener('keydown', teclas);
    return () => window.removeEventListener('keydown', teclas);
  }, [deshacer]);

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
   *  incluir los días que aún no han llegado — el rosco no llegaría nunca al
   *  100 % aunque fueras perfecto hasta hoy. */
  const ultimoContable = diaDeHoy ?? dias;

  const tocarMes = useCallback((cambio: (m: Mes) => Mes) => {
    aplicar(d => {
      const k = clave(anio, mes);
      const previo = d.meses[k] ?? MES_VACIO;
      return { ...d, meses: { ...d.meses, [k]: cambio(previo) } };
    });
  }, [aplicar, anio, mes]);

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
   *  Un arrastre entero es **un** paso de deshacer y no cuarenta: el primer
   *  `pointerdown` apila, y mientras `pintando` siga vivo los demás escriben
   *  sobre el mismo estado sin volver a apilar. */
  const pintando = useRef<boolean | null>(null);
  const ponerPintando = useCallback((idHabito: string, dia: number, valor: boolean) => {
    setDatos(d => {
      const k = clave(anio, mes);
      const previo = d.meses[k] ?? MES_VACIO;
      const c = `${idHabito}|${dia}`;
      if (!!previo.marcas[c] === valor) return d;
      const marcas = { ...previo.marcas };
      if (valor) marcas[c] = true; else delete marcas[c];
      return { ...d, meses: { ...d.meses, [k]: { ...previo, marcas } } };
    });
  }, [anio, mes]);

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

  /** Y la fila entera, hasta hoy. El gesto simétrico: «este mes he ido al
   *  gimnasio todos los días y no lo había apuntado».
   *
   *  Llega hasta `ultimoContable` y no hasta fin de mes a propósito: marcar por
   *  adelantado días que no han pasado no es un descuido que se arregle luego,
   *  es meter datos falsos en la única cifra que hace que esto sirva. */
  const alternarFila = useCallback((id: string) => {
    tocarMes(m => {
      let lleno = true;
      for (let d = 1; d <= ultimoContable; d++) {
        if (!m.marcas[`${id}|${d}`]) { lleno = false; break; }
      }
      const marcas = { ...m.marcas };
      for (let d = 1; d <= ultimoContable; d++) {
        const c = `${id}|${d}`;
        if (lleno) delete marcas[c]; else marcas[c] = true;
      }
      return { ...m, marcas };
    });
  }, [tocarMes, ultimoContable]);

  const cambiarMental = useCallback((serie: 'animo' | 'motivacion', dia: number, valor: number | null) => {
    tocarMes(m => {
      const fuente = { ...m[serie] };
      if (valor === null) delete fuente[dia]; else fuente[dia] = valor;
      return { ...m, [serie]: fuente };
    });
  }, [tocarMes]);

  const renombrar = (id: string, nombre: string) => {
    aplicar(d => ({ ...d, habitos: d.habitos.map(h => (h.id === id ? { ...h, nombre } : h)) }));
  };

  const anadir = () => {
    aplicar(d => ({ ...d, habitos: [...d.habitos, { id: `h${Date.now()}`, nombre: 'Hábito nuevo' }] }));
  };

  /** Quitar un hábito se lleva sus marcas de todos los meses. Dejarlas
   *  guardadas «por si acaso» hace que un hábito nuevo con el mismo id reviva
   *  casillas de hace medio año. Se puede deshacer, que es lo que convierte una
   *  cruz sin confirmación en algo aceptable. */
  const quitar = (id: string) => {
    aplicar(d => {
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

  /** Reordenar arrastrando por el asa. El orden de la lista es información —lo
   *  de la mañana arriba, lo de la noche abajo— y en la hoja estaba clavado.
   *
   *  El asa es lo arrastrable y no la fila entera: con la fila arrastrable, el
   *  campo del nombre deja de poder seleccionarse con el ratón. */
  const arrastrado = useRef<string | null>(null);
  const [sobre, setSobre] = useState<string | null>(null);

  const soltarSobre = (destino: string) => {
    const origen = arrastrado.current;
    arrastrado.current = null;
    setSobre(null);
    if (!origen || origen === destino) return;
    aplicar(d => {
      const lista = [...d.habitos];
      const desde = lista.findIndex(h => h.id === origen);
      const hasta = lista.findIndex(h => h.id === destino);
      if (desde < 0 || hasta < 0) return d;
      const [movido] = lista.splice(desde, 1);
      lista.splice(hasta, 0, movido);
      return { ...d, habitos: lista };
    });
  };

  const moverMes = (paso: number) => {
    const d = new Date(anio, mes + paso, 1);
    setAnio(d.getFullYear());
    setMes(d.getMonth());
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
          // Las cifras de «cuánto llevas» llegan hasta hoy; las gráficas por día
          // y por semana enseñan el mes entero, futuro incluido.
          if (d <= ultimoContable) n++;
          porDia.set(d, (porDia.get(d) ?? 0) + 1);
          porSemana.set(semanaDe(d), (porSemana.get(semanaDe(d)) ?? 0) + 1);
        }
      }
      porHabito.set(h.id, n);
      // La racha se pide al almacén entero y no solo al mes en pantalla: es la
      // única cifra de aquí que tiene sentido cruzando el cambio de mes.
      rachas.set(h.id, racha(datos, h.id, anio, mes, ultimoContable));
      total += n;
    }
    return { porHabito, rachas, porDia, porSemana, total };
  }, [habitos, mesActual, dias, ultimoContable, datos, anio, mes]);

  const objetivo = habitos.length * ultimoContable;
  const hecho = cuentas.total;
  const restante = Math.max(0, objetivo - hecho);

  /** Las semanas de la hoja: bloques de siete días desde el 1, no semanas
   *  naturales. Es como se agrupan las columnas de la rejilla, y la gráfica de
   *  arriba tiene que contar lo mismo que se ve debajo. */
  const semanas = useMemo(() => {
    const lista: { n: number; desde: number; hasta: number; largo: number; pct: number; hecho: number; tope: number }[] = [];
    for (let s = 1; s <= semanaDe(dias); s++) {
      const desde = (s - 1) * 7 + 1;
      const hasta = Math.min(dias, s * 7);
      // La semana en curso no pide los días que le faltan por llegar: su tope es
      // lo que ya pasó de ella, o su barra nunca se llenaría.
      const tope = Math.max(0, Math.min(hasta, ultimoContable) - desde + 1) * habitos.length;
      const n = cuentas.porSemana.get(s) ?? 0;
      lista.push({ n: s, desde, hasta, largo: hasta - desde + 1, hecho: n, tope, pct: porcentaje(n, tope) });
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
  const huecoAnalisis = useRef<HTMLDivElement>(null);
  const cabeceraTabla = useRef<HTMLTableSectionElement>(null);
  useEncaje(raiz, huecoRejilla, cabeceraTabla, habitos.length, dias);

  /** La cruceta: al pasar el ratón se enciende la columna entera del día, de la
   *  cabecera al último hábito. La fila la enciende la hoja sola (`tr:hover`);
   *  la columna no puede, porque en CSS no hay forma de decir «la celda que está
   *  encima de esta» sin saber cuántas columnas hay.
   *
   *  Se toca el DOM a mano y no el estado: pasar el ratón por la rejilla
   *  cambiaría de columna decenas de veces por segundo, y repintar 372 celdas de
   *  React en cada una se nota en el ratón. Aquí son dos vueltas de `classList`
   *  y solo al cruzar de columna, no en cada píxel. */
  const cuerpoRejilla = useRef<HTMLTableSectionElement>(null);
  useEffect(() => {
    const nodo = huecoRejilla.current;
    if (!nodo) return;

    let vivo: string | null = null;
    const encender = (dia: string | null) => {
      if (dia === vivo) return;
      if (vivo) nodo.querySelectorAll(`[data-dia="${vivo}"]`).forEach(n => n.classList.remove('cruz'));
      if (dia) nodo.querySelectorAll(`[data-dia="${dia}"]`).forEach(n => n.classList.add('cruz'));
      vivo = dia;
    };
    const mover = (e: PointerEvent) => {
      const celda = (e.target as HTMLElement).closest?.<HTMLElement>('[data-dia]');
      encender(celda?.dataset.dia ?? null);
    };
    const salir = () => encender(null);

    nodo.addEventListener('pointermove', mover);
    nodo.addEventListener('pointerleave', salir);
    return () => {
      nodo.removeEventListener('pointermove', mover);
      nodo.removeEventListener('pointerleave', salir);
      encender(null);
    };
  }, [dias, habitos.length]);

  /** Moverse por la rejilla con las flechas, y pintar con `Shift` puesto.
   *
   *  Va en el `tbody` y no en cada casilla —un solo oyente para 372— y encuentra
   *  el destino por los `data-` de la celda en vez de por un mapa de refs: la
   *  rejilla ya está en el DOM, y mantener 372 referencias al día con las altas
   *  y bajas de hábitos sería un segundo estado que puede discrepar. */
  const navegar = (e: React.KeyboardEvent<HTMLTableSectionElement>) => {
    const origen = e.target as HTMLElement;
    const f = Number(origen.dataset.f);
    const d = Number(origen.dataset.d);
    if (!Number.isFinite(f) || !Number.isFinite(d)) return;

    let nf = f;
    let nd = d;
    switch (e.key) {
      case 'ArrowLeft': nd = d - 1; break;
      case 'ArrowRight': nd = d + 1; break;
      case 'ArrowUp': nf = f - 1; break;
      case 'ArrowDown': nf = f + 1; break;
      case 'Home': nd = 1; break;
      case 'End': nd = dias; break;
      default: return;
    }
    if (nf < 0 || nf >= habitos.length || nd < 1 || nd > dias) return;
    e.preventDefault();

    const destino = cuerpoRejilla.current?.querySelector<HTMLButtonElement>(
      `[data-f="${nf}"][data-d="${nd}"]`,
    );
    if (!destino) return;
    // `Shift` puesto: el destino toma el valor de donde se viene. Es el arrastre
    // del ratón, para quien no lo usa.
    if (e.shiftKey) poner(habitos[nf].id, nd, origen.getAttribute('aria-checked') === 'true');
    destino.focus();
  };

  /** Con muchos hábitos la rejilla se desplaza —hay un suelo por debajo del cual
   *  una casilla deja de ser pulsable, y con treinta ni ese suelo alcanza— y el
   *  análisis de al lado tiene que irse con ella.
   *
   *  Antes el análisis estaba a `overflow: hidden`, así que a partir del hábito
   *  que no cupiera sus cifras simplemente **desaparecían**, sin barra y sin
   *  aviso: la rejilla decía que hay treinta hábitos y la tabla de al lado
   *  enseñaba veinticuatro. Un dato escondido es peor que un dato feo.
   *
   *  Manda la rejilla y el análisis la sigue; su barra va oculta por la hoja,
   *  para que no haya dos barras diciendo lo mismo. */
  const seguirElScroll = () => {
    const a = huecoAnalisis.current;
    const r = huecoRejilla.current;
    if (a && r) a.scrollTop = r.scrollTop;
  };

  const finde = (d: number) => {
    const s = new Date(anio, mes, d).getDay();
    return s === 0 || s === 6;
  };
  const claseDia = (d: number) =>
    (finde(d) ? ' finde' : '') +
    (d === diaDeHoy ? ' hoy' : '') +
    (diaDeHoy !== null && d > diaDeHoy ? ' futuro' : '');

  /** Una de las dos filas del estado mental. Las cifras y el dibujo de abajo
   *  comparten anchura de columna a propósito —cada punto cae en el centro de su
   *  casilla—, que es lo que permite leer un pico de la línea y bajar el dedo
   *  hasta el número que lo produjo.
   *
   *  Y se teclea con flechas: subir el ánimo de 6 a 7 son dos pulsaciones de
   *  borrar y escribir, o una de `↑`. */
  const filaMental = (serie: 'animo' | 'motivacion', etiqueta: string) => {
    const fuente = mesActual[serie];
    return (
      <div className="hb-mental-fila">
        <span className={`hb-mental-etiqueta ${serie}`}>{etiqueta}</span>
        <div className="hb-mental-cifras" style={{ gridTemplateColumns: `repeat(${dias}, 1fr)` }}>
          {listaDias.map(d => {
            const v = fuente[d];
            const tiene = typeof v === 'number';
            return (
              <input
                key={d}
                className={'hb-mental-celda' + (d === diaDeHoy ? ' hoy' : '') + (finde(d) ? ' finde' : '')}
                value={tiene ? v : ''}
                inputMode="numeric"
                maxLength={2}
                title={`Día ${d} · ${etiqueta.toLowerCase()} de 0 a 10 · flechas ↑ ↓ para ajustar`}
                onKeyDown={e => {
                  if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
                  e.preventDefault();
                  // Sin número puesto, la primera flecha entra por el 5 —la
                  // mitad de la escala— en vez de por un extremo.
                  const base = tiene ? v : 5;
                  cambiarMental(serie, d, acotar(base + (e.key === 'ArrowUp' ? 1 : -1), 0, 10));
                }}
                onChange={e => {
                  const t = e.target.value.trim();
                  if (t === '') return cambiarMental(serie, d, null);
                  const n = Number(t);
                  // Fuera de 0..10 no se guarda: la escala del dibujo es esa, y
                  // un 47 suelto aplastaría las dos líneas contra el suelo.
                  if (Number.isFinite(n) && n >= 0 && n <= 10) cambiarMental(serie, d, n);
                }}
              />
            );
          })}
        </div>
      </div>
    );
  };

  // El dibujo de áreas del estado mental, como en la hoja.
  const anchoMental = dias * 10;
  const camino = (fuente: Record<string, number>) => {
    const puntos: [number, number][] = [];
    for (let d = 1; d <= dias; d++) {
      const v = fuente[d];
      if (typeof v === 'number') puntos.push([(d - 1) * 10 + 5, 100 - (acotar(v, 0, 10) / 10) * 100]);
    }
    return puntos;
  };
  const linea = (p: [number, number][]) =>
    p.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
  const area = (p: [number, number][]) =>
    p.length < 2 ? '' : `${linea(p)} L${p[p.length - 1][0]},100 L${p[0][0]},100 Z`;
  const pAnimo = camino(mesActual.animo);
  const pMoti = camino(mesActual.motivacion);

  return (
    <div className="hb" data-estilo={estilo} ref={raiz}>
      <div className="hb-lienzo">

        {/* ── Arriba a la izquierda: el título y los ajustes del calendario ── */}
        <div className="hb-izq-alta">
          <div className="hb-titulo">
            <b>Seguimiento de hábitos</b>
            <span className="hb-titulo-mes">
              <button onClick={() => moverMes(-1)} title="El mes anterior" aria-label="El mes anterior">‹</button>
              — {MESES[mes]} —
              <button onClick={() => moverMes(1)} title="El mes siguiente" aria-label="El mes siguiente">›</button>
            </span>
          </div>
          <div className="hb-ajustes">
            <div className="hb-ajustes-titulo">Calendario</div>
            <label>
              <span>Año</span>
              <select value={anio} onChange={e => setAnio(Number(e.target.value))}>
                {anios.map(a => <option key={a} value={a}>{a}</option>)}
              </select>
            </label>
            <label>
              <span>Mes</span>
              <select value={mes} onChange={e => setMes(Number(e.target.value))}>
                {MESES.map((m, i) => <option key={m} value={i}>{m}</option>)}
              </select>
            </label>
          </div>
        </div>

        {/* ── Arriba en el centro: las dos gráficas ── */}
        {/* Van dentro de un envoltorio y no sueltas en la retícula: la
            columna del centro es UNA, y dos cajas puestas a pelo se
            reparten filas distintas por colocación automática — que es
            justo lo que pasó y desmontó el reparto entero. */}
        <div className="hb-arriba-centro">
        <div className="hb-caja hb-grafica hb-grafica-dia">
          <div className="hb-caja-titulo">Progreso diario</div>
          <div className="hb-barras">
            {listaDias.map(d => {
              const n = cuentas.porDia.get(d) ?? 0;
              return (
                <Barra
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

        <div className="hb-caja hb-grafica hb-grafica-semana">
          <div className="hb-caja-titulo">Progreso semanal</div>
          <div className="hb-barras hb-barras-anchas">
            {semanas.map(s => (
              <Barra
                key={s.n}
                valor={s.pct}
                rotulo={`sem. ${s.n}`}
                titulo={`Días ${s.desde} a ${s.hasta} · ${s.hecho} de ${s.tope} (${Math.round(s.pct)} %)`}
              />
            ))}
          </div>
        </div>
        </div>

        {/* ── Arriba a la derecha: las tres cifras y el rosco ── */}
        <div className="hb-arriba-dcha">
        <div className="hb-metas">
          <div><span>Objetivo</span><b>{objetivo}</b></div>
          <div><span>Hechas</span><b>{hecho}</b></div>
          <div><span>Faltan</span><b>{restante}</b></div>
        </div>

        <div className="hb-caja hb-resumen">
          <div className="hb-caja-titulo">Resumen del mes</div>
          <Rosco hecho={hecho} total={objetivo} />
        </div>
        </div>

        {/* ── El centro: «mis hábitos» y la rejilla ── */}
        <div className="hb-caja hb-rejilla">
          <div className="hb-rejilla-lienzo" ref={huecoRejilla} onScroll={seguirElScroll}>
            {habitos.length === 0 ? (
              <p className="hb-vacio">No hay ningún hábito. Pulsa «Añadir» aquí abajo y ponle nombre.</p>
            ) : (
              <table className="hb-tabla">
                <thead ref={cabeceraTabla}>
                  <tr>
                    <th className="hb-th-mis" rowSpan={3}>
                      Mis hábitos
                      <span className="hb-pista">pulsa o arrastra · flechas y espacio</span>
                    </th>
                    {semanas.map(s => (
                      <th key={s.n} className="hb-th-semana" colSpan={s.largo}>Semana {s.n}</th>
                    ))}
                  </tr>
                  <tr>
                    {listaDias.map(d => (
                      <th key={d} className={'hb-th-dia' + claseDia(d)} data-dia={d}>
                        {DIAS_SEMANA[new Date(anio, mes, d).getDay()]}
                      </th>
                    ))}
                  </tr>
                  <tr>
                    {listaDias.map(d => (
                      <th key={d} className={'hb-th-numero' + claseDia(d)} data-dia={d}>
                        <button onClick={() => alternarColumna(d)} title={`Marcar o desmarcar todo el día ${d}`}>
                          {d}
                        </button>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody ref={cuerpoRejilla} onKeyDown={navegar}>
                  {habitos.map((h, f) => (
                    <tr
                      key={h.id}
                      className={sobre === h.id ? 'hb-tr-destino' : undefined}
                      onDragOver={e => { e.preventDefault(); setSobre(h.id); }}
                      onDragLeave={() => setSobre(s => (s === h.id ? null : s))}
                      onDrop={e => { e.preventDefault(); soltarSobre(h.id); }}
                    >
                      <td className="hb-td-nombre">
                        <div className="hb-habito">
                          <span
                            className="hb-asa"
                            draggable
                            title="Arrastra para cambiarlo de sitio"
                            aria-hidden
                            onDragStart={() => { arrastrado.current = h.id; }}
                            onDragEnd={() => { arrastrado.current = null; setSobre(null); }}
                          >
                            ⠿
                          </span>
                          <button
                            className="hb-fila-toda"
                            title={`Marcar o desmarcar «${h.nombre}» hasta ${diaDeHoy !== null ? 'hoy' : `el ${dias}`}`}
                            aria-label={`Marcar o desmarcar ${h.nombre} en todo el mes`}
                            onClick={() => alternarFila(h.id)}
                          >
                            ▤
                          </button>
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
                          <td key={d} className={'hb-td-celda' + claseDia(d)} data-dia={d}>
                            <button
                              className={'hb-casilla' + (marcado ? ' marcada' : '')}
                              role="checkbox"
                              aria-checked={marcado}
                              aria-label={`${h.nombre}, día ${d}`}
                              /* El foco del tabulador entra por el día 1 y de
                                 ahí se sigue con flechas: 372 paradas de
                                 tabulador no son navegación, son un castigo. */
                              tabIndex={d === 1 ? 0 : -1}
                              data-f={f}
                              data-d={d}
                              onPointerDown={e => {
                                // El arrastre pinta; sin esto el navegador
                                // empieza a seleccionar texto de la tabla y la
                                // rejilla se queda azul a medio camino.
                                e.preventDefault();
                                (e.currentTarget as HTMLButtonElement).focus();
                                pintando.current = !marcado;
                                poner(h.id, d, !marcado);
                              }}
                              onPointerEnter={() => {
                                // Las casillas del arrastre no vuelven a apilar
                                // en la historia: el gesto entero es un paso.
                                if (pintando.current !== null) ponerPintando(h.id, d, pintando.current);
                              }}
                              /* El teclado no arrastra: para él, pulsar es
                                 alternar, que es lo que hace una casilla. */
                              onKeyDown={e => {
                                if (e.key === ' ' || e.key === 'Enter') {
                                  e.preventDefault();
                                  poner(h.id, d, !marcado);
                                }
                              }}
                            >
                              <i className="hb-marca" aria-hidden />
                            </button>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
          {/* Los mandos van al pie y en pequeño: en la hoja no existen, y arriba
              competirían con el título por la atención. */}
          <div className="hb-rejilla-pie">
            {hayQueDeshacer && (
              <button onClick={deshacer} title="Deshacer el último cambio (Ctrl+Z)">Deshacer</button>
            )}
            <button onClick={anadir}>Añadir hábito</button>
            <button className="hb-volver" onClick={onCerrar}>Volver a la llamada</button>
          </div>
        </div>

        {/* ── A la derecha: el análisis, hábito a hábito ── */}
        <div className="hb-caja hb-analisis">
          <div className="hb-caja-titulo">Análisis</div>
          <div className="hb-analisis-lienzo" ref={huecoAnalisis}>
            <table className="hb-tabla-analisis">
              <thead>
                <tr>
                  {/* «Meta» y no «Objetivo»: la columna mide 36 px y el rótulo
                      largo se cortaba por la mitad. */}
                  <th>Meta</th>
                  <th>Hechas</th>
                  <th>Faltan</th>
                  <th className="hb-col-barra">Progreso</th>
                  <th>%</th>
                </tr>
              </thead>
              <tbody>
                {habitos.map(h => {
                  const n = cuentas.porHabito.get(h.id) ?? 0;
                  const pct = porcentaje(n, ultimoContable);
                  const r = cuentas.rachas.get(h.id) ?? 0;
                  return (
                    <tr key={h.id} title={`${h.nombre} · racha de ${r} ${r === 1 ? 'día' : 'días'}`}>
                      <td>{ultimoContable}</td>
                      <td>{n}</td>
                      <td>{Math.max(0, ultimoContable - n)}</td>
                      <td className="hb-col-barra">
                        <div className="hb-barra-carril">
                          <span style={{ width: `${pct}%` }} />
                        </div>
                      </td>
                      <td className={'hb-pct' + (r >= 7 ? ' viva' : '')}>{Math.round(pct)}%</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* ── Abajo: el estado mental ── */}
        <div className="hb-caja hb-mental">
          <div className="hb-caja-titulo">Estado mental</div>
          {filaMental('animo', 'Ánimo')}
          {filaMental('motivacion', 'Motivación')}
          <div className="hb-mental-grafica">
            {/* La escala a la izquierda: sin ella, dos áreas superpuestas dicen
                «arriba» y «abajo» pero no cuánto. */}
            <div className="hb-mental-escala"><span>10</span><span>5</span><span>0</span></div>
            <svg viewBox={`0 0 ${anchoMental} 100`} preserveAspectRatio="none" aria-hidden>
              <line className="hb-guia" x1="0" y1="50" x2={anchoMental} y2="50" vectorEffect="non-scaling-stroke" />
              <path className="hb-area suave" d={area(pMoti)} />
              <path className="hb-linea suave" d={linea(pMoti)} vectorEffect="non-scaling-stroke" />
              <path className="hb-area" d={area(pAnimo)} />
              <path className="hb-linea" d={linea(pAnimo)} vectorEffect="non-scaling-stroke" />
            </svg>
          </div>
        </div>

        {/* ── Abajo a la derecha: los diez que mejor van ── */}
        <div className="hb-caja hb-top">
          <div className="hb-caja-titulo">Los diez que mejor van</div>
          <ol className="hb-top-lista">
            {ranking.map((h, i) => (
              <li key={h.id}>
                <span className="hb-top-puesto">{i + 1}</span>
                <span className="hb-top-nombre" title={h.nombre}>{h.nombre}</span>
                <span className="hb-top-cuenta">{cuentas.porHabito.get(h.id) ?? 0}</span>
              </li>
            ))}
          </ol>
        </div>
      </div>
    </div>
  );
};
