/**
 * Las piezas de la pantalla de hábitos: el encaje de la rejilla y los dos
 * dibujos que la acompañan.
 *
 * Salieron de `Habitos.tsx` el 2026-09-12, cuando el fichero pasaba de mil
 * líneas. Ninguna de las tres sabe de hábitos: `useEncaje` mide un hueco, y
 * `Barra` y `Rosco` pintan un número. Lo que sí sabe de hábitos —qué hay
 * marcado, qué falta hoy, las rachas— se queda en la pantalla.
 */

import React, { useLayoutEffect } from 'react';

import { acotar, porcentaje } from '../lib/habitos';

const FILA_MINIMA = 14;

const FILA_MAXIMA = 44;

const ALTO_ARRIBA = 128;

const ALTO_ABAJO = 186;

const HUECOS = 16;

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
export function useEncaje(
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
export const Barra: React.FC<{
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
export const Rosco: React.FC<{ hecho: number; total: number }> = ({ hecho, total }) => {
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
