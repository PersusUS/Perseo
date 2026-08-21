/**
 * El lanzador de proyectos, en la pantalla de la llamada.
 *
 * Encargo del señor Persus (2026-08-21): *«abrir los proyectos con una interfaz
 * chula deslizable en la zona del live»*, y que arranque los programas, no solo
 * que abra carpetas. Antes vivía en una tarjeta de la pestaña Estado del panel,
 * que es donde no se está mirando cuando hace falta.
 *
 * Tres reglas, que son las que hacen que esto quepa encima de una llamada:
 *
 *  1. **Lo que no se usa no ocupa sitio.** Se despliega, se usa y se va. La cara
 *     de Perseo no se tapa nunca: esto vive abajo, sobre la barra de controles.
 *  2. **Se maneja con ratón.** Aquí no hay dedos: se arrastra con el ratón, se
 *     gira la rueda, y las flechas del teclado también mueven la tira. Un
 *     carrusel que solo entiende de móviles no sirve de nada en un PC.
 *  3. **La llamada manda.** Abrir un proyecto no toca la sesión de voz.
 *
 * Y la regla de siempre, que aquí es de seguridad: **por aquí viaja el `id` y
 * nada más**. Qué se ejecuta lo decide `<datos>/proyectos.json` en el disco, lo
 * valida el núcleo (`perseo_core/proyectos.py`) y Rust solo hace de puente.
 */

import { invoke } from '@tauri-apps/api/core';
import React, { useCallback, useEffect, useRef, useState } from 'react';

export type Proyecto = {
  id: string;
  nombre: string;
  modo: string;
  destino?: string;
  carpeta?: string;
  descripcion?: string;
  arranque?: string[];
};

/** Cuánto se enseña el resultado en la tarjeta antes de que la tira se cierre. */
const MS_RESULTADO = 2200;

/** A partir de cuántos píxeles un ratón que se mueve deja de ser un clic. */
const PIXELES_DE_ARRASTRE = 6;

const IconCarpeta = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z" /></svg>
);

const IconEnlace = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><path d="M10 13a5 5 0 0 0 7 0l3-3a5 5 0 0 0-7-7l-1 1" /><path d="M14 11a5 5 0 0 0-7 0l-3 3a5 5 0 0 0 7 7l1-1" /></svg>
);

const IconArranque = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><path d="m8 5 11 7-11 7Z" /></svg>
);

const IconPrograma = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="4" width="18" height="16" rx="2" /><path d="m8 10 2.5 2L8 14" /><line x1="13" y1="14" x2="16" y2="14" /></svg>
);

const ICONOS: Record<string, () => React.ReactElement> = {
  carpeta: IconCarpeta,
  url: IconEnlace,
  arranque: IconArranque,
  programa: IconPrograma,
};

/** Qué se lee debajo del nombre cuando el proyecto no trae descripción. */
function subtitulo(p: Proyecto): string {
  if (p.descripcion) return p.descripcion;
  if (p.modo === 'arranque' && p.arranque?.length) return p.arranque.join(' ');
  if (p.modo === 'url') return p.destino ?? '';
  return p.destino ?? '';
}

export const Proyectos: React.FC<{ abierto: boolean; onCerrar: () => void }> = ({
  abierto,
  onCerrar,
}) => {
  const [lista, setLista] = useState<Proyecto[] | null>(null);
  const [fichero, setFichero] = useState('');
  const [fallo, setFallo] = useState('');
  const [lanzando, setLanzando] = useState('');
  const [resultado, setResultado] = useState<{ id: string; texto: string; malo: boolean } | null>(null);

  const tira = useRef<HTMLDivElement>(null);
  // El arrastre se lleva en una `ref` y no en estado: cambia en cada movimiento
  // del ratón, y repintar la tira sesenta veces por segundo la haría ir a
  // tirones justo mientras se desliza.
  const arrastre = useRef({ activo: false, desdeX: 0, desdeScroll: 0, movido: 0 });

  // La lista sale de un fichero que casi nunca cambia: se pide la primera vez
  // que se abre y se guarda. Volver a pedirla en cada despliegue sería un viaje
  // al núcleo para leer lo mismo.
  useEffect(() => {
    if (!abierto || lista !== null) return;
    invoke<{ proyectos: Proyecto[]; fichero: string }>('panel_proyectos')
      .then(d => {
        setLista(d.proyectos ?? []);
        setFichero(d.fichero ?? '');
      })
      .catch(e => {
        setLista([]);
        setFallo(String(e));
      });
  }, [abierto, lista]);

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
    try {
      const r = await invoke<any>('panel_abrir_proyecto', { id: p.id });
      // El núcleo contesta con un texto que empieza por «Éxito» o por «Error»:
      // se enseña tal cual, porque ya está escrito para leerse.
      const texto = typeof r === 'string' ? r : r?.resultado ?? r?.detalle ?? 'Hecho';
      setResultado({ id: p.id, texto: String(texto), malo: String(texto).startsWith('Error') });
    } catch (e) {
      setResultado({ id: p.id, texto: String(e), malo: true });
    } finally {
      setLanzando('');
    }
  }, []);

  // Cuando sale bien, la tira se va sola: es lo que se pidió —lo que no se está
  // usando no ocupa sitio—. Cuando sale mal se queda, porque hay algo que leer.
  useEffect(() => {
    if (!resultado || resultado.malo) return;
    const t = setTimeout(onCerrar, MS_RESULTADO);
    return () => clearTimeout(t);
  }, [resultado, onCerrar]);

  useEffect(() => {
    if (abierto) return;
    // Al cerrarse se olvida lo último que pasó, o reaparecería al abrir otra vez.
    setResultado(null);
    setLanzando('');
  }, [abierto]);

  // ── Deslizar con el ratón ──────────────────────────────────────────────── //

  const empezarArrastre = (e: React.PointerEvent) => {
    const nodo = tira.current;
    if (!nodo) return;
    arrastre.current = { activo: true, desdeX: e.clientX, desdeScroll: nodo.scrollLeft, movido: 0 };
    nodo.setPointerCapture(e.pointerId);
  };

  const mover = (e: React.PointerEvent) => {
    const nodo = tira.current;
    if (!nodo || !arrastre.current.activo) return;
    const recorrido = e.clientX - arrastre.current.desdeX;
    arrastre.current.movido = Math.max(arrastre.current.movido, Math.abs(recorrido));
    nodo.scrollLeft = arrastre.current.desdeScroll - recorrido;
  };

  const soltar = (e: React.PointerEvent) => {
    const nodo = tira.current;
    arrastre.current.activo = false;
    if (nodo?.hasPointerCapture(e.pointerId)) nodo.releasePointerCapture(e.pointerId);
  };

  // Un ratón que se ha movido estaba deslizando, no eligiendo. Sin esto, soltar
  // el arrastre encima de una tarjeta abriría ese proyecto sin querer.
  const eraUnClic = () => arrastre.current.movido < PIXELES_DE_ARRASTRE;

  const rueda = (e: React.WheelEvent) => {
    const nodo = tira.current;
    if (!nodo) return;
    // La rueda de un ratón de PC solo da eje vertical: aquí se gasta en mover la
    // tira de lado, que es lo único que se puede hacer con ella.
    if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) nodo.scrollLeft += e.deltaY;
  };

  const teclas = (e: React.KeyboardEvent) => {
    const nodo = tira.current;
    if (!nodo) return;
    if (e.key === 'ArrowRight') nodo.scrollLeft += 220;
    if (e.key === 'ArrowLeft') nodo.scrollLeft -= 220;
  };

  if (!abierto) return null;

  return (
    <div className="proyectos-capa" onPointerDown={e => e.target === e.currentTarget && onCerrar()}>
      <div className="proyectos-tira-marco">
        <div
          className="proyectos-tira"
          ref={tira}
          role="listbox"
          tabIndex={0}
          onPointerDown={empezarArrastre}
          onPointerMove={mover}
          onPointerUp={soltar}
          onPointerCancel={soltar}
          onWheel={rueda}
          onKeyDown={teclas}
        >
          {lista === null && <div className="proyectos-vacio">Buscando proyectos…</div>}

          {lista?.length === 0 && (
            <div className="proyectos-vacio">
              {fallo
                ? `No se pudo pedir la lista: ${fallo}`
                : `Ninguno declarado. Se añaden en ${fichero || 'datos/proyectos.json'}`}
            </div>
          )}

          {lista?.map(p => {
            const Icono = ICONOS[p.modo] ?? IconCarpeta;
            const suyo = resultado?.id === p.id ? resultado : null;
            return (
              <button
                key={p.id}
                className={`proyecto-ficha${lanzando === p.id ? ' arrancando' : ''}${
                  suyo ? (suyo.malo ? ' con-fallo' : ' hecho') : ''
                }`}
                title={subtitulo(p)}
                onClick={() => eraUnClic() && abrir(p)}
              >
                <span className="proyecto-icono"><Icono /></span>
                <span className="proyecto-nombre">{p.nombre}</span>
                <span className="proyecto-detalle">
                  {lanzando === p.id ? 'Arrancando…' : suyo ? suyo.texto : subtitulo(p)}
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default Proyectos;
