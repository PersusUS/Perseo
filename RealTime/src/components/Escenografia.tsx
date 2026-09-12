/**
 * La escenografía de la pantalla de la llamada (T-11, 2026-08-22).
 * Aquí vive todo lo que decora el modo live y nada de lo que lo hace funcionar:
 * la llamada, la cara, la transcripción y los controles siguen en `App.tsx`.
 * Este fichero solo pinta el puesto de mando alrededor.
 * Hay tres aspectos y se eligen en Ajustes (`aspectoLive`). Los tres respetan
 * el límite que se fijó en T-7 y que no conviene deshacer: **negro, monocromo,
 * la ola al fondo y nada de neones**. Lo que da el aire de sala de control es la
 * geometría y el movimiento, no el color; el color sigue reservado para los
 * puntos de estado.
 * Y una licencia que es a propósito: las lecturas dicen «PERSEO V2» donde por
 * dentro hay otra cosa. Es decoración, no un dato de diagnóstico — lo que sí
 * hace falta para depurar (latencia, kHz, tiempo en pie) es de verdad.
 */
import React, { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import type { AspectoLive } from '../lib/datos/config';
import { CONSTRUCCION, EN_DESARROLLO } from '../lib/datos/version';

/**
 * En qué punto de la llamada estamos. Manda el dibujo, no el texto.
 * `espera` es la fase de «pulsar para hablar» con el botón suelto: hay llamada
 * abierta, pero el micrófono está cerrado. Antes esto se dibujaba como
 * `escuchando`, y la pantalla se quedaba clavada en «Escuchando» mientras
 * nadie hablaba — decía justo lo contrario de lo que pasaba (2026-09-09).
 */
export type Fase = 'reposo' | 'conectando' | 'espera' | 'escuchando' | 'hablando';

interface Props {
  aspecto: AspectoLive;
  fase: Fase;
  /** El mismo texto que se enseña bajo el nombre, para no inventar otro. */
  estadoTexto: string;
  /** Segundos que lleva abierta esta llamada. 0 si no hay ninguna. */
  sesion: number;
}

/** La marca de esta construcción, tal y como se enseña. Sale en los tres
 *  aspectos a propósito: es lo que dice de un vistazo si lo que está abierto es
 *  lo último que se construyó. Ver lib/version.ts. */
const SELLO = EN_DESARROLLO ? 'DESARROLLO' : CONSTRUCCION;

/** Cada cuánto se le pregunta al núcleo por la máquina, en `mando`. Es el mismo
 *  ritmo que usa el panel: la telemetría se muestrea cada 8 s como poco. */
const REFRESCO_ESTADO = 20000;

const KANJI: Record<Fase, string> = {
  reposo: '待機',      // en espera
  conectando: '接続',  // conectando
  espera: '無音', // micrófono cerrado, esperando la pulsación
  escuchando: '傾聴',  // escuchando
  hablando: '応答',    // respondiendo
};

/** Lo que dice la cartela grande de «cartel», por fase. Corto a propósito: la
 * cartela va en una sola línea y sin partir. */
const CARTELA: Record<Fase, string> = {
  reposo: 'En reposo',
  conectando: 'Conectando',
  espera: 'Pulse para hablar',
  escuchando: 'Escuchando',
  hablando: 'Hablando',
};

/** hh:mm:ss a partir de segundos. */
export function comoReloj(segundos: number): string {
  const s = Math.max(0, Math.floor(segundos));
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${pad(Math.floor(s / 3600))}:${pad(Math.floor(s / 60) % 60)}:${pad(s % 60)}`;
}

/** El reloj de pared. Vive aquí y no en el padre para no repintar la llamada
 *  entera una vez por segundo. */
const Reloj: React.FC = () => {
  const [hora, setHora] = useState(() => new Date());
  useEffect(() => {
    const t = setInterval(() => setHora(new Date()), 1000);
    return () => clearInterval(t);
  }, []);
  return <span>{hora.toLocaleTimeString('es-ES', { hour12: false })}</span>;
};

/** Una serie dibujada a mano, igual que las del panel: un lienzo de 240×34 que
 *  se escala, con el trazo sin engordar y un punto en el extremo vivo. */
const Traza: React.FC<{ puntos: number[] }> = ({ puntos }) => {
  const utiles = puntos.filter(p => typeof p === 'number' && !Number.isNaN(p));
  if (utiles.length < 2) return null;

  const px = (i: number) => (i / (utiles.length - 1)) * 240;
  const py = (v: number) => 34 - (Math.min(100, Math.max(0, v)) / 100) * 32;
  const d = utiles.map((v, i) => `${i ? 'L' : 'M'}${px(i).toFixed(1)} ${py(v).toFixed(1)}`).join(' ');

  return (
    <svg viewBox="0 0 240 34" preserveAspectRatio="none" className="esc-traza">
      <path className="esc-traza-area" d={`${d} L240 34 L0 34 Z`} />
      <path className="esc-traza-linea" d={d} />
      <circle className="esc-traza-punta" cx={240} cy={py(utiles[utiles.length - 1])} r={2.2} />
    </svg>
  );
};

/** Un bloque con las esquinas marcadas. Dos pseudoelementos, ningún nodo de
 *  más: el mismo truco que las tarjetas del panel. */
const Bloque: React.FC<{ titulo: string; ficha?: React.ReactNode; children?: React.ReactNode }> = ({
  titulo, ficha, children,
}) => (
  <div className="esc-bloque">
    <div className="esc-bloque-cabeza">
      <span>{titulo}</span>
      {ficha != null && <span>{ficha}</span>}
    </div>
    {children}
  </div>
);

/** Las barras de voz: se mueven con lo que dice Perseo y se caen solas cuando
 *  calla, así que también sirven para ver que el audio llega.
 *  La altura la calcula el CSS con `--vol` —que ya viene suavizada— y un factor
 *  por barra: así se mueven sin que React repinte nada. */
const BarrasVoz: React.FC = () => (
  <div className="esc-voz">
    {Array.from({ length: 26 }, (_, i) => {
      // Más alto en el centro: una fila plana parece un ecualizador roto.
      const centro = 1 - Math.abs(i - 13) / 13;
      return <i key={i} style={{ ['--peso' as any]: (0.35 + centro).toFixed(2) }} />;
    })}
  </div>
);

/** La columna de instrumentos del aspecto `mando`.
 *  Lee lo mismo que el panel (`panel_estado`) y no abre ninguna vía nueva. Si
 *  el núcleo no contesta —o `psutil` no está— se queda con lo que había y no
 *  enseña un error en mitad de una llamada: esto informa, no manda. */
const Instrumentos: React.FC = () => {
  const [estado, setEstado] = useState<any>(null);

  useEffect(() => {
    let vivo = true;
    const preguntar = async () => {
      try {
        const datos = await invoke<any>('panel_estado');
        if (vivo) setEstado(datos);
      } catch {
        // Callado a propósito: ver el comentario de arriba.
      }
    };
    preguntar();
    const t = setInterval(preguntar, REFRESCO_ESTADO);
    return () => { vivo = false; clearInterval(t); };
  }, []);

  const maquina = estado?.maquina ?? {};
  const historial: any[] = Array.isArray(maquina.historial) ? maquina.historial : [];
  const presencia = estado?.presencia ?? {};
  // Los mismos nombres que usa el panel: `eventos` con `momento` y `titulo`, y
  // `correo` como un recuento por clase. Ver `estado.presencia()`.
  const eventos: any[] = presencia.eventos?.length
    ? presencia.eventos
    : presencia.proximo_evento ? [presencia.proximo_evento] : [];
  const proxima = eventos[0];
  const correos = (Object.values(presencia.correo ?? {}) as number[]).reduce((n, c) => n + c, 0);
  const trabajos = estado?.trabajos ?? {};
  const abiertos = (trabajos.pendiente ?? 0) + (trabajos.en_curso ?? 0) + (trabajos.esperando ?? 0);

  const hora = (e: any) => {
    if (!e?.momento) return '--:--';
    const d = new Date(e.momento);
    return Number.isNaN(d.getTime())
      ? '--:--'
      : d.toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit', hour12: false });
  };

  return (
    <div className="esc-instrumentos">
      <Bloque titulo="CPU" ficha={`${Math.round(maquina.cpu ?? 0)}%`}>
        <Traza puntos={historial.map(h => h.cpu)} />
      </Bloque>

      <Bloque titulo="Memoria" ficha={`${Math.round(maquina.memoria?.porcentaje ?? 0)}%`}>
        <Traza puntos={historial.map(h => h.memoria)} />
      </Bloque>

      <Bloque titulo="Cola" ficha={`${abiertos} abierto${abiertos === 1 ? '' : 's'}`}>
        <div className="esc-cifra">
          {trabajos.esperando ?? 0}
          <span> esperando un sí</span>
        </div>
      </Bloque>

      {/* La franja marca dónde acaba lo que informa y empieza lo que te toca a
          ti. Rayas, no color: el color es de los puntos de estado. */}
      <div className="esc-franja" />

      <Bloque titulo="El día" ficha={proxima ? hora(proxima) : '—'}>
        <div className="esc-nota">
          {proxima ? (proxima.titulo ?? '(sin título)') : 'Nada en las próximas 24 h.'}
        </div>
        <div className="esc-nota esc-nota-tenue">
          {correos ? `${correos} correo${correos === 1 ? '' : 's'} sin resolver` : 'Correo al día'}
        </div>
      </Bloque>
    </div>
  );
};

export const Escenografia: React.FC<Props> = ({ aspecto, fase, estadoTexto, sesion }) => {
  const enLlamada = fase !== 'reposo';

  return (
    <>
      {/* Fondo común a los tres: rejilla casi invisible y un barrido que cruza
          muy despacio. Es lo que quita de encima el aire de página quieta. */}
      <div className="esc-rejilla" />
      <div className="esc-barrido" />

      {aspecto === 'mira' && (
        <>
          <div className="esc-mira tl" />
          <div className="esc-mira tr" />
          <div className="esc-mira bl" />
          <div className="esc-mira br" />
          <div className="esc-regla" />

          <div className="esc-kana">音声接続 · {KANJI[fase]}</div>

          <div className="esc-lectura tl">
            SESIÓN <b>{comoReloj(sesion)}</b>
            <br />
            ENLACE <b>PERSEO V2</b>
            <br />
            VERSIÓN <b>{SELLO}</b>
          </div>
          <div className="esc-lectura tr">
            ENTRADA <b>16 kHz</b>
            <br />
            SALIDA <b>24 kHz</b>
          </div>
          <div className="esc-lectura bl">
            <span className="esc-latido" /> {enLlamada ? 'ENLACE ABIERTO' : 'EN ESPERA'}
          </div>
        </>
      )}

      {aspecto === 'mando' && (
        <>
          <div className="esc-cinta">
            <span className="esc-latido" />
            <span>PERSEO V2</span>
            <span className="esc-sep">·</span>
            <span>{SELLO}</span>
            <span className="esc-sep">·</span>
            <Reloj />
            <span className="esc-sep">·</span>
            <span>SESIÓN {comoReloj(sesion)}</span>
            <span className="esc-sep">·</span>
            <span className="esc-cinta-estado">{estadoTexto || 'SIN LLAMADA'}</span>
            <span className="esc-cinta-der">
              <span>{enLlamada ? 'ENLACE ABIERTO' : 'ENLACE CERRADO'}</span>
              <span className="esc-sep">·</span>
              <span>16 / 24 kHz</span>
            </span>
          </div>

          <Instrumentos />
          <BarrasVoz />
        </>
      )}

      {aspecto === 'cartel' && (
        <>
          <div className="esc-marco" />
          <div className="esc-kanji">{KANJI[fase]}</div>

          <div className="esc-cartel">
            <div className="esc-cartel-etiqueta">
              <span className="esc-latido" /> PERSEO V2 · ENLACE DE VOZ
            </div>
            {/* La cartela cambia de texto al cambiar de fase, y entra animada:
                la `key` es lo que hace que React la vuelva a montar. */}
            <div className={`esc-cartela ${fase === 'espera' ? 'esc-cartela-larga' : ''}`} key={fase}>
              {CARTELA[fase]}
            </div>
            <div className="esc-cartel-regla" />
            <div className="esc-cartel-datos">
              <div>Sesión<b>{comoReloj(sesion)}</b></div>
              <div>Enlace<b>{enLlamada ? 'Abierto' : 'Cerrado'}</b></div>
              <div>Audio<b>16 / 24 kHz</b></div>
              <div>Versión<b>{SELLO}</b></div>
            </div>
          </div>
        </>
      )}
    </>
  );
};
