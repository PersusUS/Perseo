/**
 * El panel de Perseo dentro de la ventana de la app.
 *
 * Las mismas cuatro pestañas que la interfaz del móvil —cola, correo, memoria y
 * estado— pero en React y hablando con el núcleo **a través de Rust**
 * (`src-tauri/src/panel.rs`), no por cookie.
 *
 * Por qué existe esta segunda implementación, que es una duplicación de verdad y
 * conviene tener escrito antes de que alguien la "arregle": hospedar la
 * interfaz del núcleo aquí dentro no funciona. En una ventana aparte, la CSP de
 * la app bloquea su `<script>` en línea y sale en blanco; en un `<iframe>`, la
 * cookie de sesión es `SameSite=Strict` y el navegador no la manda desde un
 * contexto embebido, así que no hay forma de autenticarse. Ninguna de las dos se
 * arregla con código nuestro.
 *
 * Lo que se gana además de la ventana única: **aquí no se pega ningún token**.
 * Lo lee Rust del disco, como para las herramientas de voz.
 *
 * El precio, y hay que pagarlo a conciencia: lo que se cambie en
 * `perseo_core/interfaz/index.html` hay que traerlo aquí. Son dos pantallas con
 * el mismo trabajo. La del móvil manda: es la que se usa a diario.
 */

import { invoke } from '@tauri-apps/api/core';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';

type Pestana = 'chat' | 'cola' | 'correo' | 'memoria' | 'estado';

type Trabajo = {
  id: number;
  estado: string;
  agente: string;
  origen: string;
  peticion?: any;
  resultado?: any;
  error?: string | null;
  confirmacion?: { resumen?: string; detalle?: string } | null;
};

type Pieza = { id: string; nombre: string; estado: string; detalle: string; arreglo: string };

type Estado = {
  encendido_segundos: number;
  /** Cuándo lo reunió el núcleo, en ISO. Sirve para saber si esta pantalla se
   *  quedó congelada: la diferencia con el reloj se enseña en la lectura. */
  generado?: string;
  piezas: Pieza[];
  trabajos: Record<string, number>;
  agentes: string[];
  disparadores: { nombre: string; activo: boolean; intervalo: number }[];
  cuota: { dia: string; nota: string; servicios: { modelo: string; usadas: number; tope: number | null }[] };
  /** La máquina donde vive el núcleo. Sin `psutil` llega `disponible: false`. */
  maquina?: any;
  /** Qué se está haciendo, qué correo espera y qué toca en la agenda. */
  presencia?: any;
};

const ESTADOS_ABIERTOS = new Set(['pendiente', 'en_curso', 'esperando']);

/** Cada cuánto se repregunta mientras el panel está delante.
 *  Se sondea en vez de escuchar el flujo SSE: el flujo se autentica por cookie y
 *  aquí no hay cookie — es justo la razón de que este panel exista. */
const REFRESCO = 4000;
const REFRESCO_ESTADO = 20000;

const CLASES_CORREO: Record<string, string> = {
  requiere_accion: 'acción',
  interesante: 'interesante',
  no_seguro: 'sin decidir',
  ignorar: 'ignorar',
};

const ORDEN_CAJONES = ['requiere_accion', 'no_seguro', 'interesante', 'ignorar'];

const FILTROS: Record<string, (t: Trabajo) => boolean> = {
  todo: () => true,
  abiertos: t => ESTADOS_ABIERTOS.has(t.estado),
  esperando: t => t.estado === 'esperando',
  mios: t => t.origen !== 'disparador',
  solos: t => t.origen === 'disparador',
  fallidos: t => t.estado === 'fallido',
};

const NOMBRES_FILTRO: Record<string, string> = {
  todo: 'todo',
  abiertos: 'abiertos',
  esperando: 'esperan un sí',
  mios: 'los pedí yo',
  solos: 'salieron solos',
  fallidos: 'fallidos',
};

function duracion(segundos: number): string {
  const d = Math.floor(segundos / 86400);
  const h = Math.floor((segundos % 86400) / 3600);
  const m = Math.floor((segundos % 3600) / 60);
  if (d) return `${d} d ${h} h`;
  if (h) return `${h} h ${m} min`;
  return `${m} min`;
}

function resumirPeticion(t: Trabajo): string {
  // Un trabajo de correo trae el lote entero dentro. Volcarlo llena la pantalla
  // del JSON de veinte correos antes de llegar al resultado.
  const mensajes = t.peticion?.mensajes;
  if (Array.isArray(mensajes)) {
    return `${mensajes.length} correo${mensajes.length === 1 ? '' : 's'} del buzón`;
  }
  return t.peticion?.texto ?? t.peticion?.accion ?? JSON.stringify(t.peticion ?? {});
}

/** Qué pasó con un trabajo, en una línea y en castellano.
 *
 *  El último recurso era `JSON.stringify(resultado)`, y se veía: guardar una
 *  conversación dejaba `{"accion":"conversacion","mensajes":4,"ruta":…,
 *  "titular":null}` en la cola. Un panel que enseña JSON es un panel que se deja
 *  de leer. */
function resumirResultado(resultado: any): string {
  if (resultado == null) return '';
  if (typeof resultado === 'string') return resultado;
  if (resultado.titular) return String(resultado.titular);
  if (resultado.texto) return String(resultado.texto);
  if (resultado.ruta) {
    const cuantos = typeof resultado.mensajes === 'number'
      ? `${resultado.mensajes} mensaje${resultado.mensajes === 1 ? '' : 's'} · `
      : '';
    return `${cuantos}guardado en ${resultado.ruta}`;
  }
  // Lo que no se sepa resumir se enseña como pares, no como JSON: sigue siendo
  // feo, pero se lee.
  return Object.entries(resultado)
    .filter(([, v]) => v !== null && v !== undefined && v !== '')
    .map(([k, v]) => `${k}: ${typeof v === 'object' ? JSON.stringify(v) : v}`)
    .join(' · ');
}

/** Encola un trabajo y espera su resultado sondeando. */
async function encolarYEsperar(agente: string, peticion: any, segundos = 20): Promise<any> {
  const trabajo = await invoke<Trabajo>('panel_encolar', { agente, peticion });
  const limite = Date.now() + segundos * 1000;
  while (Date.now() < limite) {
    await new Promise(r => setTimeout(r, 400));
    const actual = await invoke<Trabajo>('panel_trabajo', { id: trabajo.id });
    if (actual.estado === 'hecho') return actual.resultado;
    if (['fallido', 'cancelado', 'rechazado'].includes(actual.estado)) {
      throw new Error(actual.error || `El trabajo quedó ${actual.estado}`);
    }
  }
  throw new Error('Sigue en marcha; míralo en la cola.');
}

/** Una línea de correo triado, y qué se ha hecho con él.
 *
 *  Las acciones solo salen en la pestaña de Correo (`onMarcar`): en la cola,
 *  una línea de correo es el resultado de un trabajo —lo que pasó— y ahí no se
 *  decide nada. Un correo resuelto no se esconde, se apaga: esconderlo quitaría
 *  la única forma de ver que el triaje se ha comido algo. */
const LineaCorreo: React.FC<{
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
const Lectura: React.FC<{ encendido: string; generado?: string }> = ({ encendido, generado }) => {
  const [reloj, setReloj] = useState(() => new Date());
  useEffect(() => {
    const t = setInterval(() => setReloj(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  const hora = reloj.toLocaleTimeString('es-ES', { hour12: false });
  // El desfase entre el reloj y el último vistazo al núcleo: si esto crece, la
  // pantalla dejó de refrescarse y el resto de números son de hace rato.
  const desde = generado ? Math.max(0, Math.round((reloj.getTime() - new Date(generado).getTime()) / 1000)) : null;

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
const ElDia: React.FC<{ p: any }> = ({ p }) => {
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

const Presencia: React.FC<{ p: any }> = ({ p }) => {
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
const Maquina: React.FC<{ m: any }> = ({ m }) => {
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
const NotaVault: React.FC<{ n: any }> = ({ n }) => {
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

const TarjetaTrabajo: React.FC<{ t: Trabajo; onResponder: (id: number, d: string) => void }> = ({
  t,
  onResponder,
}) => {
  const notas: any[] = Array.isArray(t.resultado?.notas) ? t.resultado.notas : [];
  const clasificados: any[] = Array.isArray(t.resultado?.clasificados) ? t.resultado.clasificados : [];

  return (
    <div className="pnl-tarjeta">
      <div className="pnl-cabeza">
        <span className={`pnl-etiqueta ${t.estado}`}>{t.estado}</span>
        {` #${t.id} · ${t.agente} · ${t.origen}`}
      </div>
      <div className="pnl-cuerpo">{resumirPeticion(t)}</div>

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
            // Markdown para decir que se leyó un fichero. Ver H-36.
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
    </div>
  );
};

export const Panel: React.FC<{ onCerrar: () => void }> = ({ onCerrar }) => {
  const [pestana, setPestana] = useState<Pestana>('chat');
  // El chat escrito, que es la otra mitad de hablar. Va por `/mensaje`, la
  // misma puerta que el móvil: el router decide si contesta o encola, y lo que
  // encola aparece en la cola de al lado.
  const [dialogo, setDialogo] = useState<{ mio: boolean; texto: string }[]>([]);
  const [escrito, setEscrito] = useState('');
  const [trabajos, setTrabajos] = useState<Trabajo[]>([]);
  const [estado, setEstado] = useState<Estado | null>(null);
  const [fallo, setFallo] = useState<string>('');
  const [filtro, setFiltro] = useState<string>('todo');

  const [modoMemoria, setModoMemoria] = useState<'buscar' | 'anotar'>('buscar');
  const [consulta, setConsulta] = useState('');
  const [tituloNota, setTituloNota] = useState('');
  const [textoNota, setTextoNota] = useState('');
  const [notas, setNotas] = useState<any[] | null>(null);
  const [marcados, setMarcados] = useState<Record<string, string>>({});
  const [avisoMemoria, setAvisoMemoria] = useState('');
  const trabajando = useRef(false);

  const cargarTrabajos = useCallback(async () => {
    try {
      const datos = await invoke<{ trabajos: Trabajo[] }>('panel_trabajos', { limite: 50 });
      setTrabajos(datos.trabajos);
      setFallo('');
    } catch (e: any) {
      setFallo(String(e));
    }
  }, []);

  const cargarEstado = useCallback(async () => {
    try {
      setEstado(await invoke<Estado>('panel_estado'));
      setFallo('');
    } catch (e: any) {
      setFallo(String(e));
    }
  }, []);

  /** Qué se ha hecho con cada correo. Va aparte de los trabajos porque el
   *  triaje dice de qué va un correo y esto dice qué has hecho tú con él: lo
   *  primero lo decide un modelo, lo segundo no lo decide nadie más. */
  const cargarMarcados = useCallback(async () => {
    try {
      const datos = await invoke<{ marcados: Record<string, string> }>('panel_correos');
      setMarcados(datos.marcados ?? {});
    } catch {
      // Perder las marcas no es perder los correos: se pintan pendientes.
      setMarcados({});
    }
  }, []);

  // Los proyectos vivían aquí, en una tarjeta de la pestaña Estado, y desde el
  // 2026-08-21 viven en la pantalla de la llamada (T-4 y T-5): un lanzador
  // deslizable en `components/Proyectos.tsx`. El panel es para mirar lo que
  // pasa; lanzar cosas se hace donde se está mirando.

  useEffect(() => {
    cargarTrabajos();
    cargarEstado();
    cargarMarcados();
    const a = setInterval(cargarTrabajos, REFRESCO);
    const b = setInterval(cargarEstado, REFRESCO_ESTADO);
    const c = setInterval(cargarMarcados, REFRESCO);
    return () => { clearInterval(a); clearInterval(b); clearInterval(c); };
  }, [cargarTrabajos, cargarEstado, cargarMarcados]);

  const marcarCorreo = async (id: string, estado: string) => {
    // Optimista y luego se confirma: el sondeo tarda cuatro segundos y un botón
    // que no responde hasta entonces se pulsa dos veces.
    setMarcados(previos => {
      const siguiente = { ...previos };
      if (estado === 'pendiente') delete siguiente[id];
      else siguiente[id] = estado;
      return siguiente;
    });
    try {
      await invoke('panel_marcar_correo', { id, estado });
    } catch (e: any) {
      setFallo(String(e));
    }
    cargarMarcados();
  };

  const responder = async (id: number, decision: string) => {
    try {
      await invoke('panel_responder', { id, decision });
    } catch (e: any) {
      // Contestar desde dos sitios a la vez es normal: el núcleo resuelve el
      // empate y el segundo se lleva un 409. Recargar enseña lo que quedó.
      setFallo(String(e));
    }
    cargarTrabajos();
  };

  const buscar = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!consulta.trim() || trabajando.current) return;
    trabajando.current = true;
    setAvisoMemoria(`Buscando «${consulta.trim()}»…`);
    setNotas(null);
    try {
      const r = await encolarYEsperar('memoria', { accion: 'buscar', texto: consulta.trim() });
      setNotas(r?.notas ?? []);
      setAvisoMemoria(
        (r?.notas ?? []).length
          ? `${r.notas.length} nota(s)`
          : 'Ninguna nota. Con el plugin de Obsidian la búsqueda es literal: prueba con tildes.'
      );
    } catch (err: any) {
      setAvisoMemoria('No se pudo buscar: ' + err);
    } finally {
      trabajando.current = false;
    }
  };

  const anotar = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!tituloNota.trim() || !textoNota.trim()) {
      setAvisoMemoria('Hacen falta un título y un texto.');
      return;
    }
    if (trabajando.current) return;
    trabajando.current = true;
    setAvisoMemoria('Anotando…');
    try {
      const r = await encolarYEsperar('memoria', {
        accion: 'anotar',
        titulo: tituloNota.trim(),
        texto: textoNota.trim(),
      });
      // Solo se vacía si salió bien: perder lo que acabas de escribir porque el
      // núcleo no contestó sería la peor forma de estrenar esto.
      setTituloNota('');
      setTextoNota('');
      setAvisoMemoria(r?.titular ?? 'Anotado.');
    } catch (err: any) {
      setAvisoMemoria('No se pudo anotar: ' + err);
    } finally {
      trabajando.current = false;
    }
  };

  const cajones = useMemo(() => {
    const mapa = new Map<string, any[]>(ORDEN_CAJONES.map(c => [c, []]));
    for (const t of trabajos) {
      for (const c of t.resultado?.clasificados ?? []) {
        (mapa.get(c.clase) ?? mapa.get('no_seguro'))!.push(c);
      }
    }
    return mapa;
  }, [trabajos]);

  const abiertos = trabajos.filter(t => ESTADOS_ABIERTOS.has(t.estado)).length;
  const malas = estado?.piezas.filter(p => p.estado === 'malo').length ?? 0;
  const visibles = trabajos.filter(FILTROS[filtro]);
  const confianza = estado?.piezas.find(p => p.id === 'confianza');
  const confiando = confianza?.estado === 'aviso';

  return (
    <div className="pnl">
      <header className="pnl-cabecera">
        <h2>Panel</h2>
        <button className="pnl-pildora" onClick={onCerrar}>Volver a la llamada</button>
      </header>

      <nav className="pnl-pestanas">
        {(['chat', 'cola', 'correo', 'memoria', 'estado'] as Pestana[]).map(p => (
          <button
            key={p}
            aria-selected={pestana === p}
            onClick={() => setPestana(p)}
          >
            {p}
            {p === 'cola' && abiertos > 0 && <span className="pnl-cuenta"> ({abiertos})</span>}
            {p === 'estado' && malas > 0 && <span className="pnl-cuenta"> ({malas})</span>}
          </button>
        ))}
      </nav>

      {fallo && <div className="pnl-nota">{fallo}</div>}

      <div className="pnl-cuerpo-scroll">
        {pestana === 'chat' && (
          <>
            {dialogo.length === 0 && (
              <div className="pnl-nota">
                Escríbele. Lo que necesite trabajo aparecerá en la cola; lo trivial lo
                contesta aquí mismo.
              </div>
            )}
            {dialogo.map((m, i) => (
              <div key={i} className={`pnl-burbuja ${m.mio ? 'mia' : 'suya'}`}>{m.texto}</div>
            ))}
            <form
              className="pnl-form"
              onSubmit={async e => {
                e.preventDefault();
                const texto = escrito.trim();
                if (!texto || trabajando.current) return;
                trabajando.current = true;
                setEscrito('');
                setDialogo(d => [...d, { mio: true, texto }]);
                try {
                  const r: any = await invoke('panel_mensaje', { texto });
                  setDialogo(d => [...d, {
                    mio: false,
                    // Si encola no se inventa una respuesta: se dice que hay
                    // trabajo y la cola es donde se sigue.
                    texto: r.destino === 'responder'
                      ? (r.respuesta || '(sin respuesta)')
                      : `Encargado — trabajo #${r.trabajo.id} (${r.trabajo.agente})`,
                  }]);
                  cargarTrabajos();
                } catch (err: any) {
                  setDialogo(d => [...d, { mio: false, texto: 'No se pudo enviar: ' + err }]);
                } finally {
                  trabajando.current = false;
                }
              }}
            >
              <input
                value={escrito}
                onChange={e => setEscrito(e.target.value)}
                placeholder="Escribe a Perseo…"
              />
              <button type="submit">Enviar</button>
            </form>
          </>
        )}

        {pestana === 'cola' && (
          <>
            <div className="pnl-filtros">
              {Object.keys(FILTROS).map(clave => (
                <button
                  key={clave}
                  className="pnl-filtro"
                  aria-pressed={filtro === clave}
                  onClick={() => setFiltro(clave)}
                >
                  {NOMBRES_FILTRO[clave]} {trabajos.filter(FILTROS[clave]).length}
                </button>
              ))}
            </div>
            {visibles.length === 0 ? (
              <div className="pnl-nota">
                {filtro === 'todo' ? 'No hay nada en la cola.' : 'Nada con ese filtro.'}
              </div>
            ) : (
              visibles.map(t => <TarjetaTrabajo key={t.id} t={t} onResponder={responder} />)
            )}
          </>
        )}

        {pestana === 'correo' && (
          <>
            {/* La cifra grande es lo que queda por hacer, no lo que llegó: un
                contador que nunca baja deja de mirarse a la semana. */}
            <div className="pnl-cifras">
              {ORDEN_CAJONES.map(c => {
                const todos = cajones.get(c)!;
                const pendientes = todos.filter(x => !marcados[x.id]).length;
                return (
                  <div className="pnl-cifra" key={c}>
                    <b>{pendientes}</b>
                    <span className="pnl-mayus">{CLASES_CORREO[c]}</span>
                    {pendientes !== todos.length && (
                      <div className="pnl-motivo">de {todos.length}</div>
                    )}
                  </div>
                );
              })}
            </div>
            {ORDEN_CAJONES.every(c => cajones.get(c)!.length === 0) && (
              <div className="pnl-nota">
                Todavía no hay ningún correo triado. Sale solo cuando el disparador mira el buzón.
              </div>
            )}
            {ORDEN_CAJONES.filter(c => cajones.get(c)!.length).map(c => (
              <div className="pnl-tarjeta" key={c}>
                <div className="pnl-cabeza">
                  <span className={`pnl-clase ${c}`}>{CLASES_CORREO[c]}</span>
                  {` ${cajones.get(c)!.length}`}
                </div>
                {(() => {
                  // Lo resuelto al fondo de su cajón, sin desaparecer.
                  const lineas = [...cajones.get(c)!]
                    .sort((a, b) => (marcados[a.id] ? 1 : 0) - (marcados[b.id] ? 1 : 0))
                    .map((x, i) => (
                      <LineaCorreo key={i} c={x} estado={marcados[x.id]} onMarcar={marcarCorreo} />
                    ));
                  return c === 'ignorar' ? (
                    <details>
                      <summary>ver los ignorados</summary>
                      {lineas}
                    </details>
                  ) : (
                    lineas
                  );
                })()}
              </div>
            ))}
          </>
        )}

        {pestana === 'memoria' && (
          <>
            <div className="pnl-filtros">
              <button className="pnl-filtro" aria-pressed={modoMemoria === 'buscar'}
                onClick={() => { setModoMemoria('buscar'); setAvisoMemoria(''); setNotas(null); }}>
                buscar
              </button>
              <button className="pnl-filtro" aria-pressed={modoMemoria === 'anotar'}
                onClick={() => { setModoMemoria('anotar'); setAvisoMemoria(''); setNotas(null); }}>
                anotar
              </button>
            </div>

            {modoMemoria === 'buscar' ? (
              <form className="pnl-form" onSubmit={buscar}>
                <input
                  value={consulta}
                  onChange={e => setConsulta(e.target.value)}
                  placeholder="Buscar en el vault…"
                />
                <button type="submit">Buscar</button>
              </form>
            ) : (
              <form className="pnl-form pnl-form-alta" onSubmit={anotar}>
                <input
                  value={tituloNota}
                  onChange={e => setTituloNota(e.target.value)}
                  placeholder="Título"
                />
                <textarea
                  value={textoNota}
                  onChange={e => setTextoNota(e.target.value)}
                  placeholder="Lo que quieras recordar…"
                  rows={4}
                />
                <button type="submit">Anotar</button>
              </form>
            )}

            {avisoMemoria && <div className="pnl-nota">{avisoMemoria}</div>}
            {notas?.map((n, i) => <NotaVault key={i} n={n} />)}
          </>
        )}

        {pestana === 'estado' && estado && (
          <>
            <Lectura encendido={duracion(estado.encendido_segundos)} generado={estado.generado} />
            <div className="pnl-tarjeta pnl-hud">
              <div className="pnl-cifras">
                <div className="pnl-cifra">
                  <b>{duracion(estado.encendido_segundos)}</b>
                  <span className="pnl-mayus">encendido</span>
                </div>
                <div className="pnl-cifra">
                  <b>{abiertos}</b><span className="pnl-mayus">en cola</span>
                </div>
                <div className="pnl-cifra">
                  <b>{estado.trabajos.esperando ?? 0}</b>
                  <span className="pnl-mayus">esperan un sí</span>
                </div>
                <div className="pnl-cifra">
                  <b>{estado.agentes.length}</b><span className="pnl-mayus">agentes</span>
                </div>
              </div>
            </div>

            <Presencia p={estado.presencia ?? {}} />
            <ElDia p={estado.presencia ?? {}} />
            {estado.maquina?.disponible && <Maquina m={estado.maquina} />}

            <div className="pnl-tarjeta pnl-piezas">
              {estado.piezas.map(p => (
                <div className="pnl-pieza" key={p.id}>
                  <span className={`pnl-punto ${p.estado}`} />
                  <div>
                    <div className="pnl-nombre">{p.nombre}</div>
                    <div className="pnl-detalle">{p.detalle}</div>
                    {p.arreglo && <div className="pnl-arreglo">{p.arreglo}</div>}
                  </div>
                </div>
              ))}
            </div>

            <div className="pnl-tarjeta">
              <div className="pnl-cabeza">Cuota de hoy · {estado.cuota.dia}</div>
              {estado.cuota.servicios.length === 0 && (
                <div className="pnl-detalle">Nadie ha llamado a ningún modelo de fuera.</div>
              )}
              {estado.cuota.servicios.map(s => {
                const parte = s.tope ? Math.min(100, (s.usadas / s.tope) * 100) : 0;
                return (
                  <div key={s.modelo} style={{ marginTop: 9 }}>
                    <div className="pnl-nombre">{s.modelo}</div>
                    <div className="pnl-detalle">
                      {s.tope ? `${s.usadas} de ${s.tope}` : `${s.usadas} (sin tope conocido)`}
                    </div>
                    {s.tope && (
                      <div className="pnl-barra">
                        <span className={parte >= 90 ? 'lleno' : ''} style={{ width: `${parte}%` }} />
                      </div>
                    )}
                  </div>
                );
              })}
              <div className="pnl-arreglo">{estado.cuota.nota}</div>
            </div>

            <div className="pnl-tarjeta">
              <div className="pnl-cabeza">Disparadores</div>
              <div className="pnl-fichas">
                {estado.disparadores.map(d => (
                  <span key={d.nombre} className={`pnl-ficha ${d.activo ? '' : 'apagada'}`}>
                    {d.activo ? `${d.nombre} · cada ${Math.round(d.intervalo / 60)} min` : `${d.nombre} · apagado`}
                  </span>
                ))}
              </div>
            </div>

            <div className="pnl-acciones">
              <button
                className={`pnl-pildora ${confiando ? 'peligro' : ''}`}
                onClick={async () => {
                  await invoke('panel_confianza', { minutos: confiando ? null : 60 });
                  cargarEstado();
                }}
              >
                {confiando ? 'Apagar el modo confianza' : 'Confiar durante 60 min'}
              </button>
              <button className="pnl-pildora" onClick={cargarEstado}>Refrescar</button>
            </div>
          </>
        )}
      </div>
    </div>
  );
};
