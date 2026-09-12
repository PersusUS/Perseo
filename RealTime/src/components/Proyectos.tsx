/**
 * El riel de proyectos, en el borde izquierdo de la pantalla de la llamada.
 *
 * Rediseño pedido por el señor Persus (2026-08-24): fuera la tira horizontal
 * de abajo («no es adecuada con el diseño»); un carril **vertical**, pegado a
 * la izquierda y de arriba a abajo, que nace sobre su propia pestaña del marco.
 * Y **solo lanza aplicaciones**: las fichas que no son del modo `servicio` no
 * aparecen — para eso están las carpetas y los editores.
 *
 * Segunda vuelta, del mismo día: un carril que solo dijera «LANZAR» estaba
 * muerto — y lo quieto parece colgado. Así que el núcleo pregunta por cada
 * servicio si su puerto respira (`GET /proyectos` trae `vivo`) y cada ficha se
 * pinta como se pinta el resto del sistema:
 *
 *   ● EN MARCHA    punto lleno, la app ya corre — pulsarla abre su pestaña
 *   ○ PARADO       punto hueco — pulsarla arranca los procesos y abre sola
 *   ◉ ARRANCANDO…  punto latiendo, entre el clic y el puerto abierto
 *
 * La lista se pide CADA VEZ que se abre el riel: un estado vivo servido de la
 * caché es una mentira con retraso. Cuesta dos sondeos locales de décimas.
 *
 * Cada app lleva su **icono único** — definidos a mano aquí abajo, porque la
 * CSP no deja traer fuentes de fuera y un icono genérico por modo no dice de
 * qué va cada cosa de un vistazo. Una app nueva sin icono propio cae al
 * comodín hasta que alguien le dibuje el suyo.
 *
 * Y la regla de seguridad de siempre: **por aquí viaja el `id` y nada más**.
 * Qué se ejecuta lo decide `<datos>/proyectos.json` en el disco, lo valida el
 * núcleo (`perseo_core/servicios/proyectos.py`) y Rust solo hace de puente.
 */

import { invoke } from '@tauri-apps/api/core';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { defaultConfig, guardarAjuste, type AspectoLive } from '../lib/config';

export type Proyecto = {
  id: string;
  nombre: string;
  modo: string;
  destino?: string;
  carpeta?: string;
  descripcion?: string;
  /** Solo en modo `servicio`: ¿está su puerto respirando ahora mismo? */
  vivo?: boolean;
  /** Solo en modo `servicio`: el tamaño que el proyecto pide para su ventana. */
  ventana?: { ancho: number; alto: number } | null;
  /** Su color de identidad (#rrggbb, validado en el núcleo). Va a la BARRA de
   *  su ventana (Corteza.tsx); las fichas del riel siguen en blanco y negro,
   *  que es como las pidió el señor Persus. */
  color?: string;
};

/** Cuánto se enseña el resultado en la ficha antes de que el riel se cierre. */
const MS_RESULTADO = 2200;

/** A partir de cuántos píxeles un ratón que se mueve deja de ser un clic. */
const PIXELES_DE_ARRASTRE = 6;

/** Percha: Armario es ropa. Trazo fino, esquinas vivas, nada redondeado. */
const IconArmario = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="square">
    <path d="M12 9V7.5c0-1.2 1-1.6 1.7-2A2.1 2.1 0 1 0 10 3.4" />
    <path d="m12 9 8.5 6.5H3.5L12 9Z" />
    <path d="M3.5 20.5h17" />
  </svg>
);

/** Tres nodos en triángulo: MELCHIOR, BALTHASAR y CASPER deliberando. */
const IconMagi = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="square">
    <rect x="9" y="2.5" width="6" height="5" />
    <rect x="2.5" y="16.5" width="6" height="5" />
    <rect x="15.5" y="16.5" width="6" height="5" />
    <path d="M12 7.5v4.5m0 0-6.5 4.5m6.5-4.5 6.5 4.5" />
  </svg>
);

/** Luna creciente: Night Shift trabaja mientras la casa duerme. */
const IconNightshift = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="square">
    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79Z" />
  </svg>
);

/** Torii: la puerta de un santuario — Kotoba enseña japonés. Dintel, travesaño,
 *  el poste corto del centro y las dos columnas: nada más, que a 24 píxeles un
 *  torii con detalles es una mancha. */
const IconKotoba = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="square">
    <path d="M2 4.5h20" />
    <path d="M4.5 8.5h15" />
    <path d="M12 4.5v4" />
    <path d="M6.5 4.5v16" />
    <path d="M17.5 4.5v16" />
  </svg>
);

/** Micrófono de sobremesa: ClassTranscriber escucha la clase entera. Cápsula,
 *  el arco del soporte y el pie — a 24 píxeles, un micro con rejilla es una
 *  mancha gris. */
const IconClassTranscriber = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="square">
    <rect x="9" y="2.5" width="6" height="11" />
    <path d="M5 11.5a7 7 0 0 0 14 0" />
    <path d="M12 18.5v3" />
    <path d="M8 21.5h8" />
  </svg>
);

/** Globo: el comodín de toda app que aún no tenga icono propio. */
const IconComodin = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
    <circle cx="12" cy="12" r="9" />
    <path d="M3 12h18" />
    <path d="M12 3a13.5 13.5 0 0 1 0 18a13.5 13.5 0 0 1 0-18Z" />
  </svg>
);

/** Constelación: un nodo central con satélites — el grafo del vault. */
const IconGrafo = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="square">
    <circle cx="12" cy="12" r="2.4" />
    <circle cx="4.5" cy="5" r="1.8" />
    <circle cx="19.5" cy="5" r="1.8" />
    <circle cx="12" cy="20" r="1.8" />
    <path d="M6 6.2 10.3 10.4M18 6.2 13.7 10.4M12 14.4V18.2" />
  </svg>
);

/** Un icono POR PROYECTO, no por modo: de un vistazo se sabe qué es qué. */
const ICONOS_PROPIOS: Record<string, () => React.ReactElement> = {
  'armario-app': IconArmario,
  'classtranscriber-app': IconClassTranscriber,
  'kotoba-app': IconKotoba,
  'magi-app': IconMagi,
  'nightshift-app': IconNightshift,
};

/** Los límites del arrastre: el riel nunca sale de la pantalla. */
const X_MIN = 12, X_MAX = 88, Y_MIN = 18, Y_MAX = 82;

/** La posición guardada para un aspecto, saneada; sin guarda válida, la de fábrica. */
function posicionDe(aspecto: AspectoLive): { x: number; y: number } {
  const p = defaultConfig.posicionRiel?.[aspecto];
  if (p && Number.isFinite(p.x) && Number.isFinite(p.y)) {
    return {
      x: Math.min(X_MAX, Math.max(X_MIN, p.x)),
      y: Math.min(Y_MAX, Math.max(Y_MIN, p.y)),
    };
  }
  return { x: 25, y: 50 };
}

export const Proyectos: React.FC<{
  abierto: boolean;
  onCerrar: () => void;
  aspecto: AspectoLive;
}> = ({ abierto, onCerrar, aspecto }) => {
  const [lista, setLista] = useState<Proyecto[] | null>(null);
  const [fichero, setFichero] = useState('');
  const [fallo, setFallo] = useState('');
  const [lanzando, setLanzando] = useState('');
  const [resultado, setResultado] = useState<{ id: string; texto: string; malo: boolean } | null>(null);
  // Dónde flota el riel: lo decide el aspecto activo, y arrastrando la
  // cabecera se coloca donde convenga — se recuerda por aspecto.
  const [posicion, setPosicion] = useState(() => posicionDe(aspecto));

  const tira = useRef<HTMLDivElement>(null);
  // El arrastre se lleva en una `ref` y no en estado: cambia en cada movimiento
  // del ratón, y repintar el carril sesenta veces por segundo lo haría ir a
  // tirones justo mientras se desliza.
  const arrastre = useRef({ activo: false, desdeY: 0, desdeScroll: 0, movido: 0, cogido: false });
  // La «ronda» de apertura: sube cada vez que el riel se abre o se cierra, y
  // cualquier espera en marcha de una ronda vieja se aborta sola con ella.
  const ronda = useRef(0);

  // ── Arrastrar el riel por su cabecera ─────────────────────────────────────

  const arrastreRiel = useRef({ activo: false, desdeX: 0, desdeY: 0, origenX: 25, origenY: 50 });
  // Espejo de `posicion` para el momento de guardar: el cierre del arrastre
  // lee la última posición conocida sin depender de un cierre de React.
  const posRef = useRef(posicion);
  posRef.current = posicion;

  useEffect(() => {
    // Cambiar de aspecto recoloca el riel donde ese aspecto lo tenga guardado.
    setPosicion(posicionDe(aspecto));
  }, [aspecto]);

  const empezarArrastreRiel = (e: React.PointerEvent) => {
    e.stopPropagation();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    arrastreRiel.current = {
      activo: true,
      desdeX: e.clientX,
      desdeY: e.clientY,
      origenX: posRef.current.x,
      origenY: posRef.current.y,
    };
  };

  const moverRiel = (e: React.PointerEvent) => {
    if (!arrastreRiel.current.activo) return;
    e.stopPropagation();
    const dx = ((e.clientX - arrastreRiel.current.desdeX) / window.innerWidth) * 100;
    const dy = ((e.clientY - arrastreRiel.current.desdeY) / window.innerHeight) * 100;
    setPosicion({
      x: Math.min(X_MAX, Math.max(X_MIN, arrastreRiel.current.origenX + dx)),
      y: Math.min(Y_MAX, Math.max(Y_MIN, arrastreRiel.current.origenY + dy)),
    });
  };

  const soltarRiel = useCallback(() => {
    if (!arrastreRiel.current.activo) return;
    arrastreRiel.current.activo = false;
    // Se recuerda POR ASPECTO: lo que convenga en «mira» no es lo que convenga
    // en «mando», donde la columna izquierda ya tiene instrumentos.
    void guardarAjuste('posicionRiel', {
      ...defaultConfig.posicionRiel,
      [aspecto]: posRef.current,
    }).catch(e => console.warn('[Proyectos] no se pudo guardar la posición:', e));
  }, [aspecto]);

  // El grafo del segundo cerebro: no es un proyecto del disco ni arranca
  // servidores — es una ventana propia de Perseo que mira el vault en vivo.
  // La abre Rust con el token ya puesto, que el frontend no lo ve nunca.
  const lanzarGrafo = useCallback(async () => {
    setLanzando('grafo');
    setResultado(null);
    try {
      // 950×640: la constelación no pide más. La ventana entera cabe en un
      // portátil sin tapar la mitad del escritorio, y con el encuadre automático
      // del grafo da igual que haya menos sitio: él solo se mira dentro.
      await invoke('ventana_grafo', { ancho: 950, alto: 640 });
    } catch (e) {
      setResultado({ id: 'grafo', texto: String(e), malo: true });
    } finally {
      setLanzando('');
    }
  }, []);

  // Solo se muestran las apps que se LANZAN (modo `servicio`). Las carpetas,
  // URLs y editores se quedan en el fichero del disco, fuera del carril.
  const lanzables = lista?.filter(p => p.modo === 'servicio') ?? null;

  // La lista se pide CADA VEZ que se abre: trae el estado vivo de cada puerto,
  // y un «EN MARCHA» servido de la memoria sería una mentira con retraso.
  useEffect(() => {
    if (!abierto) return;
    let vigente = true;
    invoke<{ proyectos: Proyecto[]; fichero: string }>('panel_proyectos')
      .then(d => {
        if (!vigente) return;
        setLista(d.proyectos ?? []);
        setFichero(d.fichero ?? '');
        setFallo('');
      })
      .catch(e => {
        if (!vigente) return;
        setLista([]);
        setFallo(String(e));
      });
    return () => {
      vigente = false;
    };
  }, [abierto]);

  // Escape cierra, que es lo que espera cualquiera con algo desplegado delante.
  useEffect(() => {
    if (!abierto) return;
    const alPulsar = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onCerrar();
    };
    window.addEventListener('keydown', alPulsar);
    return () => window.removeEventListener('keydown', alPulsar);
  }, [abierto, onCerrar]);

  const abrir = useCallback(async (p: Proyecto) => {
    setLanzando(p.id);
    setResultado(null);
    let malo: boolean;
    let texto: string;
    try {
      const r = await invoke<any>('panel_abrir_proyecto', { id: p.id });
      // El núcleo contesta con un texto que empieza por «Éxito» o por «Error»:
      // se enseña tal cual, porque ya está escrito para leerse.
      texto = typeof r === 'string' ? r : r?.resultado ?? r?.detalle ?? 'Hecho';
      malo = String(texto).startsWith('Error');
      setResultado({ id: p.id, texto, malo });
    } catch (e) {
      setResultado({ id: p.id, texto: String(e), malo: true });
      setLanzando('');
      return;
    }
    if (malo) {
      setLanzando('');
      return;
    }

    // La ventana la abre ESTA pantalla, no el núcleo: nace enfocada y con el
    // tamaño que el proyecto pidió. Si ya estaba en marcha, al momento; si
    // estaba despegando, se espera a que su puerto respire — el estado vivo
    // de `/proyectos` es la señal — con plazo de 90 s, como el vigilante.
    const miRonda = ++ronda.current;
    const limite = Date.now() + 90_000;
    let vivo = false;
    while (!vivo && Date.now() < limite) {
      if (ronda.current !== miRonda) return; // cerraron el riel mientras esperaba
      const d = await invoke<{ proyectos: Proyecto[] }>('panel_proyectos').catch(() => null);
      if (d?.proyectos.find(x => x.id === p.id)?.vivo === true) {
        vivo = true;
        break;
      }
      await new Promise(resolver => setTimeout(resolver, 700));
    }
    if (ronda.current !== miRonda) return;
    if (!vivo) {
      setResultado({
        id: p.id,
        texto: `Error: ${p.nombre} no abrió su puerto en 90 s`,
        malo: true,
      });
      setLanzando('');
      return;
    }
    try {
      await invoke('ventana_proyecto', {
        url: p.destino,
        titulo: p.nombre,
        color: p.color ?? '',
        ancho: p.ventana?.ancho ?? 1280,
        alto: p.ventana?.alto ?? 900,
      });
    } catch (e) {
      setResultado({ id: p.id, texto: String(e), malo: true });
      setLanzando('');
      return;
    }
    // La ventana está fuera y enfocada: el riel ya hizo su trabajo.
    setLanzando('');
    onCerrar();
  }, [onCerrar]);

  // Cuando sale bien, el riel se va solo: lo que no se está usando no ocupa
  // sitio. Cuando sale mal se queda, porque hay algo que leer. Y mientras hay
  // un lanzamiento esperando su puerto, tampoco — la ficha está latiendo.
  useEffect(() => {
    if (!resultado || resultado.malo || lanzando) return;
    const t = setTimeout(onCerrar, MS_RESULTADO);
    return () => clearTimeout(t);
  }, [resultado, lanzando, onCerrar]);

  useEffect(() => {
    if (abierto) return;
    // Al cerrarse se olvida lo último que pasó y aborta las esperas vivas,
    // o reaparecerían al abrir otra vez.
    ronda.current += 1;
    setResultado(null);
    setLanzando('');
  }, [abierto]);

  // ── Deslizar con el ratón, en vertical ────────────────────────────────────

  const empezarArrastre = (e: React.PointerEvent) => {
    const nodo = tira.current;
    if (!nodo) return;
    arrastre.current = {
      activo: true,
      desdeY: e.clientY,
      desdeScroll: nodo.scrollTop,
      movido: 0,
      cogido: false,
    };
  };

  const mover = (e: React.PointerEvent) => {
    const nodo = tira.current;
    if (!nodo || !arrastre.current.activo) return;
    const recorrido = e.clientY - arrastre.current.desdeY;
    arrastre.current.movido = Math.max(arrastre.current.movido, Math.abs(recorrido));

    // El puntero se captura **solo cuando ya se está arrastrando de verdad**, no
    // al apoyarlo: con la captura puesta desde el principio, el `click` se lo
    // llevaba el carril y pulsar una ficha no hacía absolutamente nada.
    if (arrastre.current.movido >= PIXELES_DE_ARRASTRE && !arrastre.current.cogido) {
      nodo.setPointerCapture(e.pointerId);
      arrastre.current.cogido = true;
    }

    nodo.scrollTop = arrastre.current.desdeScroll - recorrido;
  };

  const soltar = (e: React.PointerEvent) => {
    const nodo = tira.current;
    arrastre.current.activo = false;
    if (arrastre.current.cogido && nodo?.hasPointerCapture(e.pointerId)) {
      nodo.releasePointerCapture(e.pointerId);
    }
    arrastre.current.cogido = false;
  };

  // Un ratón que se ha movido estaba deslizando, no eligiendo. Sin esto, soltar
  // el arrastre encima de una ficha lanzaría ese proyecto sin querer.
  const eraUnClic = () => arrastre.current.movido < PIXELES_DE_ARRASTRE;

  const teclas = (e: React.KeyboardEvent) => {
    const nodo = tira.current;
    if (!nodo) return;
    if (e.key === 'ArrowDown') nodo.scrollTop += 90;
    if (e.key === 'ArrowUp') nodo.scrollTop -= 90;
  };

  if (!abierto) return null;

  return (
    <div className="proyectos-capa" onClick={e => e.target === e.currentTarget && onCerrar()}>
      <aside
        className="proyectos-riel"
        style={{ left: `${posicion.x}%`, top: `${posicion.y}%` }}
      >
        <div
          className="proyectos-cabecera"
          title="Arrástrame · doble clic para recentrar"
          onPointerDown={empezarArrastreRiel}
          onPointerMove={moverRiel}
          onPointerUp={soltarRiel}
          onPointerCancel={soltarRiel}
          onDoubleClick={() => {
            // Volver al sitio de fábrica en este aspecto, y que se recuerde así.
            const origen = { x: 25, y: 50 };
            setPosicion(origen);
            posRef.current = origen;
            void guardarAjuste('posicionRiel', {
              ...defaultConfig.posicionRiel,
              [aspecto]: origen,
            }).catch(e => console.warn('[Proyectos] no se pudo guardar la posición:', e));
          }}
        >
          <span className="proyectos-titulo">Lanzar · Apps</span>
          <span className="proyectos-marca-viva" />
        </div>

        <div
          className="proyectos-tira"
          ref={tira}
          role="listbox"
          tabIndex={0}
          onPointerDown={empezarArrastre}
          onPointerMove={mover}
          onPointerUp={soltar}
          onPointerCancel={soltar}
          onKeyDown={teclas}
        >
          {/* El grafo del vault va SIEMPRE, antes que las apps del disco: es
              una herramienta de esta casa, no un proyecto declarado. */}
          <button
            className={`proyecto-ficha${lanzando === 'grafo' ? ' arrancando' : ''}`}
            title="El segundo cerebro como constelación, en vivo"
            onClick={() => eraUnClic() && !lanzando && lanzarGrafo()}
          >
            <span className="proyecto-icono"><IconGrafo /></span>
            <span className="proyecto-textos">
              <span className="proyecto-nombre">Cerebro · Grafo</span>
            </span>
            <span className={`proyecto-punto ${lanzando === 'grafo' ? 'latiendo' : 'vivo'}`} />
          </button>

          {lanzables === null && <div className="proyectos-vacio">Preguntando a los puertos…</div>}

          {lanzables?.length === 0 && (
            <div className="proyectos-vacio">
              {fallo
                ? `No se pudo pedir la lista: ${fallo}`
                : `Ninguna app declarada. Se añaden en ${fichero || 'datos/proyectos.json'}`}
            </div>
          )}

          {lanzables?.map(p => {            const Icono = ICONOS_PROPIOS[p.id] ?? IconComodin;
            const suyo = resultado?.id === p.id ? resultado : null;
            const enMarcha = !suyo && !lanzando && p.vivo === true;

            let claseEstado = 'parado';
            let textoEstado = 'PARADO';
            if (suyo) {
              claseEstado = suyo.malo ? 'fallo' : 'hecho';
              textoEstado = suyo.texto.toUpperCase();
            } else if (lanzando === p.id) {
              claseEstado = 'latiendo';
              textoEstado = 'ARRANCANDO…';
            } else if (enMarcha) {
              claseEstado = 'vivo';
              textoEstado = 'EN MARCHA';
            }

            return (
              <button
                key={p.id}
                className={`proyecto-ficha${claseEstado === 'fallo' ? ' con-fallo' : ''}${
                  claseEstado === 'hecho' ? ' hecho' : ''
                }`}
                title={p.descripcion || p.destino || p.nombre}
                onClick={() => eraUnClic() && !lanzando && abrir(p)}
              >
                {/* Sin franja ni icono teñidos: el color de cada app vive en
                    SU ventana, y el riel se queda en blanco y negro. */}
                <span className="proyecto-icono"><Icono /></span>
                <span className="proyecto-textos">
                  <span className="proyecto-nombre">{p.nombre}</span>
                </span>
                <span className={`proyecto-punto ${claseEstado}`} title={textoEstado} />
              </button>
            );
          })}
        </div>
      </aside>
    </div>
  );
};
