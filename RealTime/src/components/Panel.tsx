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
 * `perseo_core/caras/interfaz/index.html` hay que traerlo aquí. Son dos pantallas con
 * el mismo trabajo. La del móvil manda: es la que se usa a diario.
 */

import { invoke } from '@tauri-apps/api/core';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { AgentesTab } from './panel/AgentesTab';
import { ChatTab } from './panel/ChatTab';
import {
  CLASES_CORREO,
  ESTADOS_ABIERTOS,
  FILTROS,
  NOMBRES_FILTRO,
  NOMBRES_PESTANA,
  ORDEN_CAJONES,
  REFRESCO,
  REFRESCO_ESTADO,
  type Estado,
  type Pestana,
  type Trabajo,
  duracion,
  encolarYEsperar,
} from './panel/comun';
import {
  ElDia,
  Lectura,
  LineaCorreo,
  Maquina,
  NotaVault,
  Presencia,
  TarjetaTrabajo,
} from './panel/piezas';

export const Panel: React.FC<{ onCerrar: () => void }> = ({ onCerrar }) => {
  const [pestana, setPestana] = useState<Pestana>('chat');
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
  // Ojo: `aviso` es también el color de «apagadas», así que el modo
  // confianza solo cuenta cuando el sistema llega a preguntar alguna vez.
  const confirmaciones = estado?.confirmaciones !== false;
  const confiando = confirmaciones && confianza?.estado === 'aviso';

  return (
    <div className="pnl">
      <header className="pnl-cabecera">
        <h2>Panel</h2>
        <button className="pnl-pildora" onClick={onCerrar}>Volver a la llamada</button>
      </header>

      <nav className="pnl-pestanas">
        {(['chat', 'agentes', 'cola', 'correo', 'memoria', 'estado'] as Pestana[]).map(p => (
          <button
            key={p}
            aria-selected={pestana === p}
            onClick={() => setPestana(p)}
          >
            {NOMBRES_PESTANA[p]}
            {p === 'cola' && abiertos > 0 && <span className="pnl-cuenta"> ({abiertos})</span>}
            {p === 'estado' && malas > 0 && <span className="pnl-cuenta"> ({malas})</span>}
          </button>
        ))}
      </nav>

      {fallo && <div className="pnl-nota">{fallo}</div>}

      <div className={'pnl-cuerpo-scroll' + (pestana === 'chat' ? ' pnl-cuerpo-chat' : '')}>
        {pestana === 'chat' && <ChatTab />}

        {pestana === 'agentes' && <AgentesTab onEncargado={cargarTrabajos} />}

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
            <Lectura
              encendido={duracion(estado.encendido_segundos)}
              generado={estado.generado}
              version={estado.version?.marca}
            />
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
              {/* Con las confirmaciones apagadas (ADR 0005) este botón no
                  cambiaría nada: encender la confianza solo baja lo
                  irreversible a reversible, y ya no se para nada. Un botón que
                  no hace lo que dice es peor que no tenerlo. La pieza
                  «Confirmaciones» de arriba explica por qué no está. */}
              {confirmaciones && (
                <button
                  className={`pnl-pildora ${confiando ? 'peligro' : ''}`}
                  onClick={async () => {
                    await invoke('panel_confianza', { minutos: confiando ? null : 60 });
                    cargarEstado();
                  }}
                >
                  {confiando ? 'Apagar el modo confianza' : 'Confiar durante 60 min'}
                </button>
              )}
              <button className="pnl-pildora" onClick={cargarEstado}>Refrescar</button>
            </div>
          </>
        )}
      </div>
    </div>
  );
};
