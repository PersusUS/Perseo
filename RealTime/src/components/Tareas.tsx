/**
 * El corcho: la pantalla de tareas de la app.
 *
 * Pedida por el señor Persus (2026-08-30) y **rehecha el mismo día** con tres
 * correcciones suyas, que son las que explican por qué esto es como es:
 *
 *  1. **Pantalla propia, al lado de Hábitos**, no una pestaña del panel. El
 *     panel es para mirar lo que hace el núcleo —cola, correo, encargos—; esto
 *     no lo hace el núcleo, lo hace él con la mano. Se abre desde el riel
 *     (`components/Marco.tsx`) como los hábitos y tapa la llamada igual.
 *  2. **Las notas se arrastran de verdad.** El arrastre nativo del navegador
 *     (`draggable`) da una imagen fantasma pálida, no deja soltar fuera de una
 *     zona declarada y en WebView2 se pelea con la región de arrastre de la
 *     ventana. Aquí el gesto va con eventos de puntero: la nota se despega,
 *     sigue al ratón inclinada, y el hueco donde va a caer se abre delante.
 *     De paso funciona igual con el dedo.
 *  3. **La papelera es una papelera**, ladeada en el rincón de abajo a la
 *     derecha y casi fuera de la pantalla: en reposo solo asoma un canto. Entra
 *     cuando pasas por encima y cuando coges una nota —para eso está—, abre la
 *     tapa si le acercas una, y se pulsa para mirar dentro y recuperar lo
 *     tirado.
 *
 * El dato sigue fuera, en `lib/tareas.ts`, con sus reglas y sus pruebas. Esta
 * pantalla arrastra, pinta y pide; no cuenta nada.
 *
 * Y la regla que no se rompe aunque el gesto sea de ratón: **todo lo que se
 * hace arrastrando se puede hacer sin arrastrar**. La ficha abierta lleva sus
 * botones de columna y su «a la papelera».
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';

import {
  COLORES, NOMBRES_COLUMNA,
  anadir, borrar, crear, deColumna, editar, foto, guardar, haceCuanto, leer,
  mover, restaurar, resumen, tirar, vaciarPapelera,
  type Color, type Columna, type Datos, type Tarea,
} from '../lib/tareas';
import '../styles/tareas.css';

/** Las tres zonas del corcho. La papelera existe en el dato pero **no es una
 *  columna del tablero**: es el cubo de abajo a la izquierda. */
const ZONAS: Columna[] = ['sin_hacer', 'en_proceso', 'completadas'];

/** Cuántos píxeles hay que mover el ratón para que esto deje de ser un clic y
 *  pase a ser un arrastre. Sin umbral, abrir la ficha de una nota es imposible:
 *  la mano siempre tiembla un píxel entre pulsar y soltar. */
const UMBRAL = 5;

/** Autoscroll: a qué distancia del filo de una columna empieza a correr la
 *  lista, y cuánto corre por fotograma. Sin esto, una columna larga no se puede
 *  llenar por abajo — no hay forma de llegar con la nota en la mano. */
const BORDE_SCROLL = 48;
const PASO_SCROLL = 12;

/** Cuánto se espera, sin que nadie toque nada, antes de mandarle la copia al
 *  núcleo. Igual que en los hábitos: arrastrar una nota cambia el estado
 *  decenas de veces en un segundo y cada cambio sería un POST; con la espera,
 *  un arrastre entero manda uno. */
const ESPERA_ESPEJO = 2000;

/** El gesto en curso. `activo` distingue el clic del arrastre; hasta que no se
 *  supera el umbral, la nota no se ha despegado del corcho. */
type Arrastre = {
  id: string;
  activo: boolean;
  /** Dónde se agarró la nota, dentro de la nota. Es lo que hace que no salte
   *  al centro del cursor al cogerla. */
  dx: number;
  dy: number;
  ancho: number;
  alto: number;
  giro: number;
  /** Dónde estaba el puntero al despegar. Sin esto, la nota se pinta una vez en
   *  la esquina de la pantalla antes de que llegue el primer movimiento. */
  x: number;
  y: number;
};

/** Dónde caería: en qué zona y en qué posición de ella. */
type Destino = { columna: Columna; indice: number };

const ICONOS: Record<Columna, React.ReactElement> = {
  sin_hacer: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="square">
      <rect x="4.5" y="4.5" width="15" height="15" />
    </svg>
  ),
  en_proceso: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="square">
      <circle cx="12" cy="12" r="7.5" />
      <path d="M12 7.5v5l3.5 2" />
    </svg>
  ),
  completadas: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="square">
      <rect x="4.5" y="4.5" width="15" height="15" />
      <path d="m8.5 12 2.5 2.5 4.5-5.5" />
    </svg>
  ),
  papelera: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="square">
      <path d="M4 7h16M9.5 7V4.5h5V7M6.5 7l1 13h9l1-13" />
    </svg>
  ),
};

/**
 * La papelera del rincón de abajo a la derecha.
 *
 * Dibujada, no un icono: cubo con estrías, tapa aparte y un asa. La tapa es un
 * grupo con su propio giro para que se abra desde la bisagra de la izquierda,
 * como una de verdad; girar el cubo entero la haría parecer una caja volcada.
 *
 * Vive ladeada y casi fuera de la pantalla —en reposo solo asoma un canto— y
 * entra en tres casos: al pasar el ratón por encima, mientras haya una nota en
 * el aire, y cuando está abierta enseñando lo que guarda. La inclinación no se
 * pierde en ninguno: una papelera que se endereza al acercarse deja de parecer
 * un objeto y pasa a parecer un botón.
 */
const Papelera: React.FC<{
  cuenta: number;
  /** Hay una nota en el aire: la papelera se asoma sola. */
  alerta: boolean;
  /** La nota en el aire está justo encima: abre la tapa. */
  apuntada: boolean;
  abierta: boolean;
  onPulsar: () => void;
}> = ({ cuenta, alerta, apuntada, abierta, onPulsar }) => (
  <div className="tb-papelera-zona" data-papelera="1">
  <button
    type="button"
    className={
      'tb-papelera' +
      (alerta ? ' tb-papelera-alerta' : '') +
      (apuntada ? ' tb-papelera-apuntada' : '') +
      (abierta ? ' tb-papelera-abierta' : '')
    }
    onClick={onPulsar}
    title={
      cuenta
        ? `Papelera — ${cuenta} nota${cuenta === 1 ? '' : 's'} dentro`
        : 'Papelera — vacía'
    }
  >
    <svg viewBox="0 0 80 96" aria-hidden="true">
      {/* La tapa, con su bisagra a la izquierda. */}
      <g className="tb-papelera-tapa">
        <rect x="8" y="22" width="64" height="9" rx="1.5" />
        <rect x="33" y="15" width="14" height="6" rx="2" />
      </g>
      {/* El cubo, un poco más estrecho abajo. */}
      <path d="M14 33h52l-5 55a4 4 0 0 1-4 3.6H23a4 4 0 0 1-4-3.6z" />
      <path className="tb-papelera-estria" d="M31 43v41M40 43v41M49 43v41" />
    </svg>
    <span className="tb-papelera-rotulo">
      Papelera{cuenta > 0 && <b> · {cuenta}</b>}
    </span>
  </button>
  </div>
);

/** Una nota en el corcho. Memorizada: durante un arrastre esto se repinta
 *  muchas veces por segundo y son las hojas del árbol. */
const Nota = React.memo<{
  tarea: Tarea;
  onAbrir: (id: string) => void;
  onCoger: (e: React.PointerEvent<HTMLDivElement>, tarea: Tarea) => void;
}>(({ tarea, onAbrir, onCoger }) => (
  <div
    className={'tb-nota tb-papel-' + tarea.color}
    style={{ '--giro': tarea.giro + 'deg' } as React.CSSProperties}
    data-nota={tarea.id}
    tabIndex={0}
    role="button"
    aria-label={`${tarea.titulo} — ${NOMBRES_COLUMNA[tarea.columna]}`}
    onPointerDown={e => onCoger(e, tarea)}
    onKeyDown={e => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        onAbrir(tarea.id);
      }
    }}
  >
    <span className="tb-chincheta" aria-hidden="true" />
    <p className="tb-nota-titulo">{tarea.titulo}</p>
    {tarea.detalle.trim() && <p className="tb-nota-detalle">{tarea.detalle.trim()}</p>}
    <span className="tb-nota-pie">{haceCuanto(tarea.movida)}</span>
  </div>
));
Nota.displayName = 'Nota';

export const Tareas: React.FC<{ onCerrar: () => void }> = ({ onCerrar }) => {
  const [datos, setDatos] = useState<Datos>(leer);
  const [abierta, setAbierta] = useState<string | null>(null);
  const [nueva, setNueva] = useState('');
  const [arrastre, setArrastre] = useState<Arrastre | null>(null);
  const [destino, setDestino] = useState<Destino | null>(null);
  const [sobrePapelera, setSobrePapelera] = useState(false);
  const [papeleraAbierta, setPapeleraAbierta] = useState(false);

  /** La nota que vuela. Se mueve tocando su `style` a mano y no con estado:
   *  un `setState` por cada `pointermove` repinta el tablero entero sesenta
   *  veces por segundo para mover un rectángulo doscientos píxeles. */
  const volando = useRef<HTMLDivElement>(null);
  /** Lo mismo que `arrastre` y `destino`, a mano, para poder leerlos dentro de
   *  los oyentes de ventana sin volver a suscribirlos en cada movimiento. */
  const gesto = useRef<Arrastre | null>(null);
  const caida = useRef<Destino | null>(null);
  const enPapelera = useRef(false);
  const tituloFicha = useRef<HTMLInputElement>(null);
  /** El tablero de ahora mismo, legible desde dentro de un oyente de ventana:
   *  el estado de React que ve `soltar` es el del render donde nació el gesto. */
  const datosRef = useRef(datos);
  useEffect(() => { datosRef.current = datos; }, [datos]);

  /** Y una copia al núcleo, para que Perseo lo sepa también fuera de la llamada.
   *
   *  El tablero vive en el `localStorage` de esta ventana, que es lo correcto
   *  —mover una nota no puede depender de que el núcleo esté encendido— y a la
   *  vez deja fuera a media casa: el Perseo de la llamada corre AQUÍ y lo lee
   *  sin más, pero el chat escrito y los agentes son Python y no ven dentro de
   *  un navegador. Esto es el puente, y va en una sola dirección: nadie mueve
   *  notas desde el núcleo (ver `perseo_core/tareas.py`).
   *
   *  Si falla, se calla: el núcleo apagado es un estado normal de esta app, y
   *  un aviso rojo por no haber podido mandar una copia que nadie ha pedido
   *  sería alarmar por nada. El cambio siguiente la manda otra vez. */
  useEffect(() => {
    const t = setTimeout(() => {
      invoke('tareas_espejo', { texto: resumen(datos), foto: foto(datos) })
        .catch(e => console.debug('[Tareas] El núcleo no recogió la copia:', e));
    }, ESPERA_ESPEJO);
    return () => clearTimeout(t);
  }, [datos]);

  const aplicar = useCallback((cambio: (d: Datos) => Datos) => {
    setDatos(previos => {
      const siguientes = cambio(previos);
      guardar(siguientes);
      return siguientes;
    });
  }, []);

  const porColumna = useMemo(() => {
    const mapa = {} as Record<Columna, Tarea[]>;
    for (const c of [...ZONAS, 'papelera' as Columna]) mapa[c] = deColumna(datos, c);
    return mapa;
  }, [datos]);

  const tarea = abierta ? datos.tareas.find(t => t.id === abierta) ?? null : null;
  const enPapel = porColumna.papelera;

  // ── El arrastre ───────────────────────────────────────────────────────────

  /** Qué hay bajo el puntero: una zona del corcho y en qué hueco de ella, o la
   *  papelera. Se pregunta al documento en vez de llevar la cuenta con eventos
   *  de entrada y salida por caja: con el elemento en la mano tapando el ratón,
   *  esos eventos llegan en un orden que no se puede razonar. */
  const mirarDebajo = useCallback((x: number, y: number) => {
    const bajo = document.elementFromPoint(x, y) as HTMLElement | null;
    const cubo = bajo?.closest('[data-papelera]');
    if (cubo) {
      enPapelera.current = true;
      setSobrePapelera(true);
      caida.current = null;
      setDestino(null);
      return;
    }
    if (enPapelera.current) {
      enPapelera.current = false;
      setSobrePapelera(false);
    }

    const zona = bajo?.closest('[data-columna]') as HTMLElement | null;
    if (!zona) {
      caida.current = null;
      setDestino(d => (d === null ? d : null));
      return;
    }
    const columna = zona.dataset.columna as Columna;
    const pila = zona.querySelector('.tb-pila') as HTMLElement | null;

    // El hueco es cuántas notas quedan ANTES del puntero, leyendo la zona como
    // se lee un texto: primero las filas de arriba enteras, y dentro de la
    // fila donde está el puntero, las que quedan a su izquierda. Contar solo
    // por la altura funcionaba mientras las notas iban en una sola columna;
    // desde que se reparten en filas, dejaba caer todo al final.
    let indice = 0;
    if (pila) {
      const notas = Array.from(pila.querySelectorAll<HTMLElement>('[data-nota]'));
      for (const n of notas) {
        const caja = n.getBoundingClientRect();
        const suFila = y >= caja.top && y <= caja.bottom;
        if (y > caja.bottom || (suFila && x > caja.left + caja.width / 2)) indice++;
      }
      // Autoscroll cuando se arrastra pegado a un filo de la columna.
      const caja = pila.getBoundingClientRect();
      if (y < caja.top + BORDE_SCROLL) pila.scrollTop -= PASO_SCROLL;
      else if (y > caja.bottom - BORDE_SCROLL) pila.scrollTop += PASO_SCROLL;
    }
    const siguiente = { columna, indice };
    const previo = caida.current;
    if (!previo || previo.columna !== columna || previo.indice !== indice) {
      caida.current = siguiente;
      setDestino(siguiente);
    }
  }, []);

  const coger = useCallback((e: React.PointerEvent<HTMLDivElement>, t: Tarea) => {
    // Solo el botón principal, y nunca desde un campo de texto.
    if (e.button !== 0) return;
    const caja = (e.currentTarget as HTMLElement).getBoundingClientRect();
    gesto.current = {
      id: t.id,
      activo: false,
      dx: e.clientX - caja.left,
      dy: e.clientY - caja.top,
      ancho: caja.width,
      alto: caja.height,
      giro: t.giro,
      x: e.clientX,
      y: e.clientY,
    };
    // El puntero se sigue en la ventana y no en la nota: la nota se va a
    // repintar en otro sitio del árbol en cuanto el gesto se active.
    const desde = { x: e.clientX, y: e.clientY };

    const limpiar = () => {
      setArrastre(null);
      setDestino(null);
      setSobrePapelera(false);
      caida.current = null;
      enPapelera.current = false;
    };

    const mover_ = (ev: PointerEvent) => {
      const g = gesto.current;
      if (!g) return;
      if (!g.activo) {
        if (Math.hypot(ev.clientX - desde.x, ev.clientY - desde.y) < UMBRAL) return;
        g.activo = true;
        g.x = ev.clientX;
        g.y = ev.clientY;
        setArrastre({ ...g });
      }
      const nodo = volando.current;
      if (nodo) {
        nodo.style.left = `${ev.clientX - g.dx}px`;
        nodo.style.top = `${ev.clientY - g.dy}px`;
      }
      mirarDebajo(ev.clientX, ev.clientY);
    };

    const soltar = () => {
      window.removeEventListener('pointermove', mover_);
      window.removeEventListener('pointerup', soltar);
      window.removeEventListener('pointercancel', soltar);
      const g = gesto.current;
      gesto.current = null;

      if (!g) return;
      if (!g.activo) {
        // No llegó a ser un arrastre: es un clic, y un clic abre la ficha.
        setAbierta(g.id);
        return;
      }
      if (enPapelera.current) {
        aplicar(d => tirar(d, g.id));
      } else if (caida.current) {
        const { columna, indice } = caida.current;
        const lista = deColumna(datosRef.current, columna).filter(x => x.id !== g.id);
        aplicar(d => mover(d, g.id, columna, lista[indice]?.id));
      }
      // Y si se suelta en el marco de madera o fuera de la ventana, no pasa
      // nada: la nota vuelve a su sitio. Perder una tarea por soltar mal sería
      // la forma más rápida de dejar de fiarse del tablero.
      limpiar();
    };

    window.addEventListener('pointermove', mover_);
    window.addEventListener('pointerup', soltar);
    window.addEventListener('pointercancel', soltar);
    e.preventDefault();
  }, [aplicar, mirarDebajo]);

  // ── Teclado ───────────────────────────────────────────────────────────────

  useEffect(() => {
    const tecla = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      if (abierta) setAbierta(null);
      else if (papeleraAbierta) setPapeleraAbierta(false);
      else onCerrar();
    };
    window.addEventListener('keydown', tecla);
    return () => window.removeEventListener('keydown', tecla);
  }, [abierta, papeleraAbierta, onCerrar]);

  // ── Altas ─────────────────────────────────────────────────────────────────

  const crearEn = (columna: Columna, titulo: string, abrir: boolean) => {
    const t = crear({ titulo, columna });
    aplicar(d => anadir(d, t));
    if (abrir) {
      setAbierta(t.id);
      requestAnimationFrame(() => tituloFicha.current?.select());
    }
  };

  const enviarNueva = (e: React.FormEvent) => {
    e.preventDefault();
    const titulo = nueva.trim();
    if (!titulo) return;
    crearEn('sin_hacer', titulo, false);
    setNueva('');
  };

  const volada = arrastre ? datos.tareas.find(t => t.id === arrastre.id) ?? null : null;
  const pendientes = porColumna.sin_hacer.length + porColumna.en_proceso.length;

  return (
    <div className={'tb' + (arrastre ? ' tb-en-vuelo' : '')}>
      <header className="tb-cabecera">
        <h2>Tablero</h2>
        <form onSubmit={enviarNueva} className="tb-alta">
          <input
            value={nueva}
            onChange={e => setNueva(e.target.value)}
            placeholder="Escribe una tarea y pulsa Intro…"
            aria-label="Nueva tarea"
          />
          <button type="submit" disabled={!nueva.trim()}>Clavar</button>
        </form>
        <span className="tb-cuenta-total">
          {pendientes === 0
            ? 'Nada pendiente'
            : `${pendientes} pendiente${pendientes === 1 ? '' : 's'}`}
        </span>
        <button className="tb-volver" onClick={onCerrar}>Volver a la llamada</button>
      </header>

      {/* El marco de madera y, dentro, el corcho. Son dos cajas y no una con
          borde: la sombra del marco tiene que caer SOBRE el corcho, y un borde
          no proyecta hacia dentro. */}
      <div className="tb-marco">
        <div className="tb-corcho">
          {ZONAS.map(columna => (
            <section
              key={columna}
              data-columna={columna}
              className={'tb-zona' + (destino?.columna === columna ? ' tb-zona-diana' : '')}
            >
              <header className="tb-zona-cabecera">
                <span className="tb-zona-icono">{ICONOS[columna]}</span>
                <h3>{NOMBRES_COLUMNA[columna]}</h3>
                <span className="tb-zona-cuenta">{porColumna[columna].length}</span>
                <button
                  className="tb-zona-accion"
                  onClick={() => crearEn(columna, 'Nueva tarea', true)}
                  title={`Clavar una nota en «${NOMBRES_COLUMNA[columna]}»`}
                >
                  + Nota
                </button>
              </header>

              <div className="tb-pila">
                {porColumna[columna]
                  .filter(t => t.id !== arrastre?.id)
                  .map((t, i) => (
                    <React.Fragment key={t.id}>
                      {destino?.columna === columna && destino.indice === i && (
                        <div className="tb-hueco" aria-hidden="true" />
                      )}
                      <Nota tarea={t} onAbrir={setAbierta} onCoger={coger} />
                    </React.Fragment>
                  ))}
                {destino?.columna === columna
                  && destino.indice >= porColumna[columna].filter(t => t.id !== arrastre?.id).length && (
                  <div className="tb-hueco" aria-hidden="true" />
                )}
                {!porColumna[columna].length && destino?.columna !== columna && (
                  <p className="tb-vacia">Suelta aquí una nota.</p>
                )}
              </div>
            </section>
          ))}
        </div>
      </div>

      {/* ── La papelera y su cajón ── */}
      {papeleraAbierta && (
        <div className="tb-cajon" role="dialog" aria-label="Papelera">
          <header>
            <h3>Papelera</h3>
            <button
              disabled={!enPapel.length}
              onClick={() => aplicar(vaciarPapelera)}
              title="Vaciar — esto sí borra"
            >
              Vaciar
            </button>
            <button onClick={() => setPapeleraAbierta(false)}>Cerrar</button>
          </header>
          {!enPapel.length ? (
            <p className="tb-vacia">No has tirado nada.</p>
          ) : (
            <ul>
              {enPapel.map(t => (
                <li key={t.id}>
                  <span className={'tb-mota tb-papel-' + t.color} aria-hidden="true" />
                  <span className="tb-cajon-titulo">{t.titulo}</span>
                  <button onClick={() => aplicar(d => restaurar(d, t.id))}>Recuperar</button>
                  <button className="tb-peligro" onClick={() => aplicar(d => borrar(d, t.id))}>
                    Borrar
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      <Papelera
        cuenta={enPapel.length}
        alerta={!!arrastre}
        apuntada={sobrePapelera}
        abierta={papeleraAbierta}
        onPulsar={() => setPapeleraAbierta(v => !v)}
      />

      {/* ── La nota en el aire ── */}
      {arrastre && volada && (
        <div
          ref={volando}
          className={'tb-volando tb-papel-' + volada.color}
          style={{
            width: arrastre.ancho,
            height: arrastre.alto,
            left: arrastre.x - arrastre.dx,
            top: arrastre.y - arrastre.dy,
          }}
          aria-hidden="true"
        >
          <span className="tb-chincheta" />
          <p className="tb-nota-titulo">{volada.titulo}</p>
          {volada.detalle.trim() && <p className="tb-nota-detalle">{volada.detalle.trim()}</p>}
        </div>
      )}

      {/* ── La ficha ── */}
      {tarea && (
        <div className="tb-velo" onClick={() => setAbierta(null)}>
          <div
            className={'tb-ficha tb-papel-' + tarea.color}
            role="dialog"
            aria-label="Detalle de la tarea"
            onClick={e => e.stopPropagation()}
          >
            <input
              ref={tituloFicha}
              className="tb-ficha-titulo"
              value={tarea.titulo}
              placeholder="Título de la tarea"
              onChange={e => aplicar(d => editar(d, tarea.id, { titulo: e.target.value }))}
            />
            <textarea
              className="tb-ficha-detalle"
              value={tarea.detalle}
              placeholder="El detalle: qué hay que hacer, con quién, para cuándo…"
              onChange={e => aplicar(d => editar(d, tarea.id, { detalle: e.target.value }))}
            />

            <div className="tb-ficha-fila">
              <span className="tb-ficha-rotulo">Papel</span>
              <div className="tb-colores">
                {COLORES.map(c => (
                  <button
                    key={c}
                    className={'tb-color tb-papel-' + c + (tarea.color === c ? ' tb-color-puesto' : '')}
                    aria-label={`Papel ${c}`}
                    aria-pressed={tarea.color === c}
                    onClick={() => aplicar(d => editar(d, tarea.id, { color: c as Color }))}
                  />
                ))}
              </div>
            </div>

            <div className="tb-ficha-fila">
              <span className="tb-ficha-rotulo">Columna</span>
              <div className="tb-mover">
                {ZONAS.map(c => (
                  <button
                    key={c}
                    aria-pressed={tarea.columna === c}
                    onClick={() => aplicar(d => mover(d, tarea.id, c))}
                  >
                    {NOMBRES_COLUMNA[c]}
                  </button>
                ))}
              </div>
            </div>

            <p className="tb-ficha-fechas">
              Clavada {haceCuanto(tarea.creada)} · movida {haceCuanto(tarea.movida)}
            </p>

            <footer className="tb-ficha-pie">
              {tarea.columna === 'papelera' ? (
                <>
                  <button onClick={() => aplicar(d => restaurar(d, tarea.id))}>Recuperar</button>
                  <button
                    className="tb-peligro"
                    onClick={() => { aplicar(d => borrar(d, tarea.id)); setAbierta(null); }}
                  >
                    Borrar del todo
                  </button>
                </>
              ) : (
                <button onClick={() => aplicar(d => tirar(d, tarea.id))}>A la papelera</button>
              )}
              <button className="tb-ficha-cerrar" onClick={() => setAbierta(null)}>Cerrar</button>
            </footer>
          </div>
        </div>
      )}
    </div>
  );
};
