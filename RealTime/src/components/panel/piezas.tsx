/**
 * Las piezas que dibujan las pestañas del panel: barras, lecturas, tarjetas.
 *
 * Ninguna decide nada. Reciben lo que el núcleo contestó y lo pintan; si aquí
 * dentro aparece una decisión de negocio, está en el sitio equivocado. Ver la
 * cabecera de `Panel.tsx`.
 */

import { invoke } from '@tauri-apps/api/core';
import React, { useCallback, useEffect, useState } from 'react';

import { CONSTRUCCION, EN_DESARROLLO } from '../../lib/version';

import {
  CLASES_CORREO,
  ESTADOS_ABIERTOS,
  ESTADO_LEGIBLE,
  type Trabajo,
  encolarYEsperar,
  resumirPeticion,
  resumirResultado,
} from './comun';

export const LineaCorreo: React.FC<{
  c: any;
  estado?: string;
  onMarcar?: (id: string, estado: string) => void;
}> = ({ c, estado, onMarcar }) => (
  <div className={'pnl-correo' + (estado ? ' resuelto' : '')}>
    <span className={`pnl-clase ${c.clase ?? ''}`}>{CLASES_CORREO[c.clase] ?? c.clase ?? '?'}</span>
    {` ${c.remitente ?? '?'} — `}
    <span className="pnl-asunto">{c.asunto ?? '(sin asunto)'}</span>
    {c.motivo && <div className="pnl-motivo">{c.motivo}</div>}
    {onMarcar && c.id && (
      <div className="pnl-acciones-correo">
        {estado ? (
          <>
            <span className="pnl-motivo">{estado === 'atendido' ? 'Atendido' : 'Descartado'}</span>
            <button onClick={() => onMarcar(c.id, 'pendiente')}>Reabrir</button>
          </>
        ) : (
          <>
            <button className="hecho" onClick={() => onMarcar(c.id, 'atendido')}>Hecho</button>
            <button onClick={() => onMarcar(c.id, 'descartado')}>Descartar</button>
          </>
        )}
      </div>
    )}
  </div>
);

/** Una barra con su número. Misma información que en el móvil, misma forma:
 *  dos dibujos distintos del mismo dato acaban discrepando. */
const Barra: React.FC<{ etiqueta: string; porcentaje: number; detalle?: string }> = ({
  etiqueta, porcentaje, detalle,
}) => (
  <div className="pnl-medida">
    <div className="pnl-medida-cabeza">
      <span className="pnl-mayus">{etiqueta}</span>
      <span>{Math.round(porcentaje)}%</span>
    </div>
    <div className="pnl-carril">
      <div
        className={'pnl-relleno' + (porcentaje >= 90 ? ' malo' : porcentaje >= 70 ? ' aviso' : '')}
        style={{ width: `${Math.min(100, Math.max(0, porcentaje))}%` }}
      />
    </div>
    {detalle && <div className="pnl-motivo">{detalle}</div>}
  </div>
);

/** Lo que un asistente debería saber sin que se lo preguntes. Va lo primero de
 *  la pestaña porque es lo único que cambia lo que haces ahora. */
/** La línea de lectura de arriba del todo: reloj, tiempo encendido y un punto
 *  que late.
 *
 *  No dice nada que no esté ya en las cifras de debajo, y aun así hace falta:
 *  es lo que convierte una pantalla de datos en un puesto encendido. El punto
 *  late porque un panel quieto y un panel colgado se ven igual. */
export const Lectura: React.FC<{ encendido: string; generado?: string; version?: string }> = ({
  encendido, generado, version,
}) => {
  const [reloj, setReloj] = useState(() => new Date());
  useEffect(() => {
    const t = setInterval(() => setReloj(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  const hora = reloj.toLocaleTimeString('es-ES', { hour12: false });
  // El desfase entre el reloj y el último vistazo al núcleo: si esto crece, la
  // pantalla dejó de refrescarse y el resto de números son de hace rato.
  const desde = generado ? Math.max(0, Math.round((reloj.getTime() - new Date(generado).getTime()) / 1000)) : null;

  // En desarrollo no hay marca que comparar: se construye en cada recarga.
  const desfasada = !EN_DESARROLLO && !!version && version !== CONSTRUCCION;

  return (
    <div className="pnl-lectura">
      <span className="pnl-latido" />
      <span>PERSEO // NÚCLEO ACTIVO</span>
      <span className="pnl-lectura-sep">·</span>
      <span>{hora}</span>
      <span className="pnl-lectura-sep">·</span>
      <span>EN PIE {encendido}</span>
      {desde !== null && (
        <>
          <span className="pnl-lectura-sep">·</span>
          <span>DATOS DE HACE {desde}s</span>
        </>
      )}
      {/* Dos marcas, no una: la que lleva esta app dentro y la que el núcleo
          tiene sellada. Si no coinciden, esta ventana es de una construcción
          anterior y hay que pasar `perseo actualizar`. Ver lib/version.ts. */}
      <span className="pnl-lectura-sep">·</span>
      <span className={desfasada ? 'pnl-lectura-aviso' : undefined}>
        VERSIÓN {EN_DESARROLLO ? 'DESARROLLO' : CONSTRUCCION}
        {desfasada && ` · EL NÚCLEO DICE ${version}`}
      </span>
    </div>
  );
};

/** La línea de una magnitud en el tiempo, dibujada a mano en SVG.
 *
 *  Un número dice si la CPU está alta **ahora**; la línea dice si lleva diez
 *  minutos así, que es la pregunta que uno se hace de verdad mirando esto. Sin
 *  ejes ni rejilla: el eje va de 0 a 100 siempre, así que dos líneas se comparan
 *  entre sí sin leer un solo número.
 */
const Linea: React.FC<{ puntos: number[]; etiqueta: string; valor: string }> = ({
  puntos,
  etiqueta,
  valor,
}) => {
  const ancho = 240;
  const alto = 34;
  // Con una sola muestra no hay línea que dibujar; se repite para que salga
  // una recta en vez de un hueco, que en una pantalla que se acaba de abrir
  // parece que la telemetría no va.
  const serie = puntos.length === 1 ? [puntos[0], puntos[0]] : puntos;
  const paso = serie.length > 1 ? ancho / (serie.length - 1) : ancho;
  const y = (v: number) => alto - (Math.max(0, Math.min(100, v)) / 100) * alto;
  const camino = serie.map((v, i) => `${i === 0 ? 'M' : 'L'}${(i * paso).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  const relleno = `${camino} L${ancho},${alto} L0,${alto} Z`;

  return (
    <div className="pnl-linea">
      <div className="pnl-linea-cabeza">
        <span className="pnl-mayus">{etiqueta}</span>
        <span className="pnl-linea-valor">{valor}</span>
      </div>
      <svg viewBox={`0 0 ${ancho} ${alto}`} preserveAspectRatio="none" aria-hidden>
        {/* La mitad de la escala, para tener contra qué leer la línea sin ejes. */}
        <line className="pnl-linea-mitad" x1="0" y1={alto / 2} x2={ancho} y2={alto / 2} />
        <path className="pnl-linea-area" d={relleno} />
        <path className="pnl-linea-trazo" d={camino} />
        {/* Dónde está *ahora*: sin esto, en una línea plana no se sabe cuál es
            el extremo vivo y cuál el viejo. */}
        <circle className="pnl-linea-punta" cx={ancho} cy={y(serie[serie.length - 1])} r="2" />
      </svg>
    </div>
  );
};

/** El día: lo que hay en la agenda y el correo que nadie ha resuelto.
 *
 *  Los dos datos ya estaban en el sistema —en el calendario y en el triaje— y
 *  había que ir a buscarlos a dos sitios. Aquí se leen de una mirada, que es
 *  para lo que sirve un tablero. */
export const ElDia: React.FC<{ p: any }> = ({ p }) => {
  const eventos: any[] = p.eventos?.length ? p.eventos : p.proximo_evento ? [p.proximo_evento] : [];
  const pendientes = Object.entries(p.correo ?? {}) as [string, number][];
  const total = pendientes.reduce((n, [, c]) => n + c, 0);

  const hora = (e: any) =>
    e?.momento
      ? new Date(e.momento).toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit' })
      : '--:--';

  return (
    <div className="pnl-tarjeta pnl-hud">
      <div className="pnl-cabeza">El día</div>
      {eventos.length ? (
        eventos.map((e, i) => (
          <div className="pnl-cita" key={e.id ?? i}>
            <span className="pnl-cita-hora">{hora(e)}</span>
            <span className="pnl-cita-titulo">{e.titulo ?? '(sin título)'}</span>
          </div>
        ))
      ) : (
        <div className="pnl-detalle">Nada en la agenda de las próximas 24 h.</div>
      )}
      <div className="pnl-detalle pnl-separado">
        {total
          ? `${total} correo${total === 1 ? '' : 's'} sin resolver` +
            (p.correo?.requiere_accion ? ` · ${p.correo.requiere_accion} requieren acción` : '')
          : 'Correo al día'}
      </div>
    </div>
  );
};

export const Presencia: React.FC<{ p: any }> = ({ p }) => {
  return (
    <div className="pnl-tarjeta pnl-hud">
      <div className="pnl-cabeza">Ahora mismo</div>
      <div className="pnl-detalle">
        {p.haciendo ? `Trabajando: #${p.haciendo.id} · ${p.haciendo.agente}` : 'Sin nada entre manos'}
      </div>
      {!!p.esperando_un_si && (
        <div className="pnl-detalle">{p.esperando_un_si} esperando un sí</div>
      )}
      {/* El correo y la agenda se cuentan en «El día», justo debajo: repetirlos
          aquí era la misma frase dos veces en la misma pantalla. */}
    </div>
  );
};

/** La máquina donde vive el núcleo — lo único de esta pantalla que no habla de
 *  Perseo. Si un día el núcleo se muda a la Raspberry, describe la Raspberry. */
export const Maquina: React.FC<{ m: any }> = ({ m }) => {
  const historial: any[] = Array.isArray(m.historial) ? m.historial : [];
  return (
  <div className="pnl-tarjeta pnl-hud">
    <div className="pnl-cabeza">
      Máquina
      {m.bateria && (
        <span className="pnl-ficha">
          {m.bateria.porcentaje}%{m.bateria.enchufado ? ' · enchufada' : ''}
        </span>
      )}
    </div>
    <div className="pnl-medidas">
      <Barra etiqueta="CPU" porcentaje={m.cpu ?? 0} detalle={`${m.nucleos ?? '?'} hilos`} />
      <Barra etiqueta="RAM" porcentaje={m.memoria?.porcentaje ?? 0} detalle={m.memoria?.legible} />
      <Barra etiqueta="Disco" porcentaje={m.disco?.porcentaje ?? 0} detalle={m.disco?.legible} />
    </div>
    {historial.length > 0 && (
      <div className="pnl-lineas">
        <Linea
          etiqueta="CPU"
          valor={`${Math.round(m.cpu ?? 0)}%`}
          puntos={historial.map(h => h.cpu)}
        />
        <Linea
          etiqueta="RAM"
          valor={`${Math.round(m.memoria?.porcentaje ?? 0)}%`}
          puntos={historial.map(h => h.memoria)}
        />
      </div>
    )}
    {m.red?.legible && <div className="pnl-motivo">Red · {m.red.legible}</div>}
  </div>
  );
};

/** Una nota del vault. El contenido se pide solo al desplegarla: una búsqueda
 *  devuelve diez, y traerlas enteras para leer una es tirar el trabajo. */
export const NotaVault: React.FC<{ n: any }> = ({ n }) => {
  const [contenido, setContenido] = useState<string | null>(null);

  const abrir = async (e: React.SyntheticEvent<HTMLDetailsElement>) => {
    if (!e.currentTarget.open || contenido !== null || !n.ruta) return;
    setContenido('Leyendo…');
    try {
      const r = await encolarYEsperar('memoria', { accion: 'leer', ruta: n.ruta });
      setContenido(r?.contenido ?? '(vacía)');
    } catch (err: any) {
      setContenido('No se pudo leer: ' + err);
    }
  };

  return (
    <details className="pnl-tarjeta pnl-nota" onToggle={abrir}>
      <summary>{n.titulo || n.ruta || '(sin título)'}</summary>
      {n.ruta && <div className="pnl-ruta">{n.ruta}</div>}
      {n.extracto && <div className="pnl-extracto">{n.extracto}</div>}
      {contenido !== null && <div className="pnl-contenido">{contenido}</div>}
    </details>
  );
};

export const TarjetaTrabajo: React.FC<{
  t: Trabajo;
  onResponder: (id: number, d: string) => void;
  /** Lo que la pestaña de agentes cuelga debajo: la bitácora del encargo. La
   *  cola no la enseña —ahí se mira el ciclo de vida, no el paso a paso—. */
  extra?: React.ReactNode;
}> = ({ t, onResponder, extra }) => {
  const notas: any[] = Array.isArray(t.resultado?.notas) ? t.resultado.notas : [];
  const clasificados: any[] = Array.isArray(t.resultado?.clasificados) ? t.resultado.clasificados : [];

  return (
    <div className="pnl-tarjeta">
      <div className="pnl-cabeza">
        <span className={`pnl-etiqueta ${t.estado}`}>{ESTADO_LEGIBLE[t.estado] ?? t.estado}</span>
        {` #${t.id} · ${t.agente} · ${t.origen}`}
      </div>
      <div className="pnl-cuerpo">{resumirPeticion(t)}</div>

      {/* Lo que está haciendo AHORA. Un encargo de código tarda minutos y sin
          esto la tarjeta dice «en curso» y nada más durante todo ese rato. */}
      {t.progreso && <div className="pnl-progreso">{t.progreso}</div>}

      {(t.resultado || t.error) && (
        <div className="pnl-resultado">
          {t.error ? (
            `Error: ${t.error}`
          ) : notas.length ? (
            <>
              <div>{t.resultado.titular ?? 'Sin resultados'}</div>
              {notas.slice(0, 3).map((n, i) => <NotaVault key={i} n={n} />)}
              {notas.length > 3 && (
                <div className="pnl-motivo">y {notas.length - 3} más — búscalas en Memoria</div>
              )}
            </>
          ) : clasificados.length ? (
            <>
              <div>{t.resultado.titular ?? 'Nada que destacar'}</div>
              {clasificados
                .filter(c => c.clase !== 'ignorar')
                .map((c, i) => <LineaCorreo key={i} c={c} />)}
            </>
          ) : typeof t.resultado?.contenido === 'string' ? (
            // Leer una nota devuelve la nota entera: doce mil caracteres de
            // Markdown para decir que se leyó un fichero.
            `Leída ${t.resultado.ruta ?? ''} — ${t.resultado.contenido.length} caracteres`
          ) : (
            resumirResultado(t.resultado)
          )}
        </div>
      )}

      {t.estado === 'esperando' && t.confirmacion && (
        <div className="pnl-pregunta">
          <div>{t.confirmacion.resumen ?? '¿Confirmas?'}</div>
          {t.confirmacion.detalle && <div className="pnl-detalle">{t.confirmacion.detalle}</div>}
        </div>
      )}

      {ESTADOS_ABIERTOS.has(t.estado) && (
        <div className="pnl-acciones">
          {t.estado === 'esperando' && (
            <>
              <button className="pnl-pildora aprobar" onClick={() => onResponder(t.id, 'aprobar')}>
                Aprobar
              </button>
              <button className="pnl-pildora peligro" onClick={() => onResponder(t.id, 'rechazar')}>
                Rechazar
              </button>
            </>
          )}
          <button className="pnl-pildora peligro" onClick={() => onResponder(t.id, 'cancelar')}>
            Cancelar
          </button>
        </div>
      )}

      {extra}
    </div>
  );
};

/** Un paso de la bitácora de un encargo. */
type Paso = {
  tipo: string;
  titulo: string;
  detalle: string;
  agente: string;
  ok: boolean;
  momento: string;
};

type Actividad = {
  id: number;
  vivo: boolean;
  estado?: string;
  pasos: Paso[];
  agentes: { id: string; titulo: string; pasos: number; fallos: number }[];
};

const NOMBRES_DE_PASO: Record<string, string> = {
  herramienta: 'hace',
  resultado: 'sale',
  dice: 'dice',
  piensa: 'piensa',
  subagente: 'subagente',
  fin: 'fin',
  error: 'error',
};

/** La bitácora de un encargo: qué hizo el agente principal y qué hizo cada
 *  subagente, paso a paso y con el detalle a mano.
 *
 *  Existe por el 2026-08-25: dos encargos seguidos dijeron HECHO —«16 vueltas»—
 *  sin haber abierto la app que se les pidió, y no había forma de saber qué
 *  habían hecho durante esas dieciséis vueltas sin abrir el registro del
 *  núcleo desde otro ordenador. Ahora se abre aquí, y se puede entrar dentro de
 *  cada subagente. */
export const Bitacora: React.FC<{ id: number; vivo: boolean }> = ({ id, vivo }) => {
  const [actividad, setActividad] = useState<Actividad | null>(null);
  const [mirado, setMirado] = useState<string>('todo');
  const [fallo, setFallo] = useState('');

  const cargar = useCallback(async () => {
    try {
      setActividad(await invoke<Actividad>('panel_actividad', { id }));
      setFallo('');
    } catch (e: any) {
      setFallo(String(e));
    }
  }, [id]);

  useEffect(() => {
    cargar();
    // Un encargo terminado ya no cambia: sondearlo sería repintar encima de lo
    // que estás leyendo cada tres segundos, sin nada nuevo que enseñar.
    if (!vivo) return;
    const t = setInterval(cargar, 3000);
    return () => clearInterval(t);
  }, [cargar, vivo]);

  if (fallo) return <div className="pnl-motivo">No se pudo leer la actividad: {fallo}</div>;
  if (!actividad) return <div className="pnl-motivo">Leyendo la actividad…</div>;

  const visibles = actividad.pasos.filter(p => mirado === 'todo' || p.agente === mirado);

  return (
    <div className="pnl-actividad">
      {actividad.agentes.length > 1 && (
        <div className="pnl-filtros">
          <button
            type="button"
            className="pnl-ficha"
            aria-pressed={mirado === 'todo'}
            onClick={() => setMirado('todo')}
          >
            todo ({actividad.pasos.length})
          </button>
          {actividad.agentes.map(a => (
            <button
              key={a.id}
              type="button"
              className="pnl-ficha"
              aria-pressed={mirado === a.id}
              onClick={() => setMirado(a.id)}
              title={a.id}
            >
              {a.titulo} ({a.pasos}){a.fallos ? ' ⚠' : ''}
            </button>
          ))}
        </div>
      )}

      <div className="pnl-pasos">
        {visibles.length === 0 && (
          <div className="pnl-motivo">
            Sin pasos apuntados. Los motores de consola no cuentan nada hasta el final.
          </div>
        )}
        {visibles.map((p, i) => (
          <div key={i} className={`pnl-paso${p.ok ? '' : ' mal'}`}>
            <span className="pnl-hora">{(p.momento || '').slice(11, 19)}</span>
            <span className="pnl-tipo-paso">{NOMBRES_DE_PASO[p.tipo] ?? p.tipo}</span>
            <div className="pnl-que">
              <div>{p.titulo || '(sin título)'}</div>
              {p.detalle && p.detalle !== p.titulo && (
                <details>
                  <summary>detalle</summary>
                  <pre>{p.detalle}</pre>
                </details>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
