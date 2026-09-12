/**
 * La pestaña del chat escrito.
 *
 * Salió de `Panel.tsx` el 2026-09-12: una pestaña, un fichero.
 */

import { invoke } from '@tauri-apps/api/core';
import React, { useCallback, useEffect, useRef, useState } from 'react';

import { type Mensaje, type Sesion } from './comun';

/** El chat escrito, la otra mitad de hablar.
 *
 *  Ya no va por `/mensaje` (el router local contestaba sin herramientas ni
 *  memoria): ahora cada turno es un trabajo para el agente `chat`, que piensa
 *  con Gemini y usa las MISMAS herramientas que la voz —agenda, buzón triado,
 *  memoria, web, PC, MCP, subagentes—. La vista solo encola y sondea: las caras
 *  no piensan. */
export const ChatTab: React.FC = () => {
  const [sesiones, setSesiones] = useState<Sesion[]>([]);
  const [sesion, setSesion] = useState<number | null>(null);
  const [mensajes, setMensajes] = useState<Mensaje[]>([]);
  const [turno, setTurno] = useState<string>('libre');
  const [escrito, setEscrito] = useState('');
  const [aviso, setAviso] = useState('');
  /** El carril por el que scrollea la conversación y si estábamos abajo.
   *  Empujar a abajo en cada trozo arrastraba a quien había subido a releer:
   *  solo se sigue al final cuando ya se estaba cerca o al enviar. */
  const hiloRef = useRef<HTMLDivElement | null>(null);
  const pegadoAbajoRef = useRef(true);

  const cargarSesiones = useCallback(async (preferir?: number) => {
    try {
      const datos = await invoke<{ sesiones: Sesion[] }>('chat_sesiones');
      setSesiones(datos.sesiones);
      setSesion(actual => {
        if (actual != null && datos.sesiones.some(s => s.id === actual)) return actual;
        if (preferir != null && datos.sesiones.some(s => s.id === preferir)) return preferir;
        return datos.sesiones[0]?.id ?? null;
      });
    } catch (e: any) {
      setAviso(String(e));
    }
  }, []);

  /** Sondea mientras hay un turno en marcha. El texto de Perseo crece en la
   *  base; aquí se ve crecer en pantalla. Cuando el semáforo vuelve a «libre»,
   *  el sondeo se para solo. */
  useEffect(() => {
    if (sesion == null) return;
    let vivo = true;
    let temporizador: number | undefined;

    const mirar = async () => {
      try {
        const datos = await invoke<Sesion & { mensajes: Mensaje[] }>('chat_sesion', { id: sesion });
        if (!vivo) return;
        setMensajes(datos.mensajes);
        setTurno(datos.turno);
        setAviso('');
        if (datos.turno !== 'ocupado') return;
      } catch (e: any) {
        if (vivo) setAviso(String(e));
      }
      temporizador = window.setTimeout(mirar, 700);
    };

    mirar();
    return () => { vivo = false; if (temporizador) clearTimeout(temporizador); };
  }, [sesion, turno]);

  // La última burbuja a la vista, pero solo si ya se miraba abajo: llegar un
  // trozo nuevo no autoriza a secuestrar el scroll de quien subió a releer.
  useEffect(() => {
    const hilo = hiloRef.current;
    if (!hilo || !pegadoAbajoRef.current) return;
    hilo.scrollTop = hilo.scrollHeight;
  }, [mensajes, aviso]);

  // Cambiar de conversación empieza abajo: la posición de scroll era de la
  // otra charla y no significa nada aquí.
  useEffect(() => {
    pegadoAbajoRef.current = true;
    const hilo = hiloRef.current;
    if (hilo) hilo.scrollTop = hilo.scrollHeight;
  }, [sesion]);

  const enviar = async (e: React.FormEvent) => {
    e.preventDefault();
    const texto = escrito.trim();
    if (!texto || sesion == null || turno === 'ocupado') return;
    setEscrito('');
    // Lo que acabas de mandar se ve siempre, aunque estuvieras leyendo arriba.
    pegadoAbajoRef.current = true;
    try {
      await invoke('chat_hablar', { id: sesion, texto });
      const datos = await invoke<Sesion & { mensajes: Mensaje[] }>('chat_sesion', { id: sesion });
      setMensajes(datos.mensajes);
      setTurno(datos.turno);
      setAviso('');
    } catch (err: any) {
      setAviso(String(err));
    }
  };

  const nueva = async () => {
    try {
      const nueva_sesion = await invoke<Sesion>('chat_crear');
      await cargarSesiones(nueva_sesion.id);
    } catch (err: any) {
      setAviso(String(err));
    }
  };

  const borrar = async (id: number) => {
    try {
      await invoke('chat_borrar', { id });
      await cargarSesiones();
    } catch (err: any) {
      setAviso(String(err));
    }
  };

  useEffect(() => { cargarSesiones(); }, [cargarSesiones]);

  const ocupado = turno === 'ocupado';
  const pensando = ocupado && (!mensajes.length || mensajes[mensajes.length - 1].estado === 'escribiendo');

  return (
    <div className="pnl-chat">
      <aside className="pnl-chat-sesiones">
        <button className="pnl-pildora pnl-chat-nueva" onClick={nueva}>Nueva conversación</button>
        <div className="pnl-chat-lista">
          {sesiones.map(s => (
            <div key={s.id} className={'pnl-chat-sesion' + (s.id === sesion ? ' activa' : '')}>
              <button onClick={() => setSesion(s.id)} title={s.titulo}>
                {s.titulo || 'Conversación'}
              </button>
              {s.id === sesion && !ocupado && (
                <span className="pnl-chat-borrar" title="Borrar" onClick={() => borrar(s.id)}>×</span>
              )}
            </div>
          ))}
          {sesiones.length === 0 && <div className="pnl-nota">Ninguna conversación todavía.</div>}
        </div>
      </aside>

      <div
        className="pnl-chat-hilo"
        ref={hiloRef}
        onScroll={() => {
          const hilo = hiloRef.current;
          if (!hilo) return;
          pegadoAbajoRef.current = hilo.scrollHeight - hilo.scrollTop - hilo.clientHeight < 120;
        }}
      >
        {mensajes.length === 0 && (
          <div className="pnl-nota">
            Escríbele. Tiene las mismas manos que en la llamada: agenda, buzón
            triado, memoria, web, PC, MCP y subagentes. Lo que haga aparece en la cola.
          </div>
        )}
        {mensajes.map(m => (
          <div key={m.id} className={`pnl-turno ${m.rol === 'usuario' ? 'mio' : 'suyo'}`}>
            <div className="pnl-turno-cabeza">
              <span>{m.rol === 'usuario' ? 'Señor Persus' : 'Perseo'}</span>
              <span className="pnl-turno-hora">
                {new Date(m.momento).toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit' })}
              </span>
            </div>
            <div className={'pnl-burbuja ' + (m.rol === 'usuario' ? 'mia' : 'suya')}>
              {m.texto}
              {m.estado === 'escribiendo' && m.texto && <span className="pnl-cursor" />}
            </div>
            {m.herramientas?.length > 0 && (
              <div className="pnl-herramientas">
                {m.herramientas.map((h, i) => <span key={i} className="pnl-ficha">{h}</span>)}
              </div>
            )}
          </div>
        ))}
        {pensando && (
          <div className="pnl-burbuja suya pensando"><span /><span /><span /></div>
        )}
        {aviso && <div className="pnl-nota">{aviso}</div>}
        <form className="pnl-form pnl-chat-form" onSubmit={enviar}>
          <input
            value={escrito}
            onChange={e => setEscrito(e.target.value)}
            placeholder={ocupado ? 'Perseo está escribiendo…' : 'Escribe a Perseo…'}
            disabled={ocupado}
          />
          <button type="submit" disabled={ocupado}>Enviar</button>
        </form>
      </div>
    </div>
  );
};
