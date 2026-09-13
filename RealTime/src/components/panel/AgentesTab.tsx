/**
 * La pestaña de encargos de código.
 *
 * Salió de `Panel.tsx` el 2026-09-12: una pestaña, un fichero.
 */

import * as nucleo from '../../lib/datos/panel';
import React, { useCallback, useEffect, useState } from 'react';

import { ESTADOS_ABIERTOS, REFRESCO, type Trabajo } from './comun';
import { Bitacora, TarjetaTrabajo } from './piezas';

/** Los encargos de código, lanzados desde el panel.
 *
 *  Un encargo se escribe COMO SE HABLA: «En Armario, añade un README con
 *  opencode». El núcleo entiende el proyecto y el motor desde el propio
 *  texto — la vista no pregunta nada, solo manda el encargo. Las caras no
 *  piensan: el que lee y decide es `dev.py`.
 *
 *  El señor Persus tachó los desplegables el 2026-08-24 —*«no me gusta cómo
 *  se ve el elegir opciones»*— y pidió que el dónde se dijera en la entrada.
 *  Antes de eso había dos selects que nadie sabía rellenar. */
const EJEMPLOS_ENCARGO = [
  'En CVScraper: ejecuta los tests, arregla los que fallen y cuenta qué pasaba.',
  'Añade un README con qué es este proyecto y cómo arrancarlo.',
  'Arranca la app de Armario y déjala escuchando para poder usarla desde el móvil.',
];

/** Con qué SISTEMA trabaja un encargo. Lo que ya no se elige es la carpeta.
 *
 *  **opencode va primero y es lo que sale puesto**: es el que no gasta
 *  suscripción, y el señor Persus lo dijo con todas las letras el 2026-08-26
 *  —«usar Claude es secundario, quiero la opción gratuita»—. Claude sigue ahí
 *  para el encargo que lo merezca, un escalón por debajo. */
const SISTEMAS_AGENTE: [string, string][] = [
  ['opencode', 'opencode · gratis'],
  ['sdk', 'Claude · SDK'],
  ['claude', 'Claude · consola'],
  ['', 'El configurado por defecto'],
];

/** Los modelos GRATIS de opencode Zen que **contestan**, de más rápido a menos.
 *
 *  Salen de `opencode models opencode` y de probarlos uno a uno el 2026-08-28.
 *  El orden no es capricho: los dos que encabezaban esta lista —los nemotron—
 *  no devolvían una sola línea en cien segundos, y un modelo que no contesta no
 *  da error: se cuelga hasta el tope de 900 s. Desde esta pantalla eso se veía
 *  como un encargo que no termina nunca, que es justo lo que pasaba.
 *
 *  Es la MISMA lista que `perseo_core/agentes/dev.py`, `perseo_core/caras/interfaz/index.html`
 *  y `commands/subagentes_mcp.py`. Si cambia una, cambian todas: se vuelven a
 *  sacar del mismo comando y se vuelven a probar. */
const MODELOS_OPENCODE: [string, string][] = [
  ['opencode/big-pickle', 'big-pickle · 200k'],
  ['opencode/hy3-free', 'hy3 · 190k'],
  ['opencode/muse-spark-1.2-contributor-free', 'muse-spark · 1M'],
  ['opencode/ling-3.0-flash-fin-free', 'ling-3.0-flash'],
  ['opencode/mimo-v2.5-free', 'mimo-v2.5 · 200k (lento)'],
  // No contestaban el 2026-08-28: cien segundos sin una línea. Al final, para
  // que nadie los coja sin pedirlos. Ver `perseo_core/agentes/dev.py`.
  ['opencode/nemotron-3-ultra-free', 'nemotron-3-ultra · 1M (no contestaba)'],
  ['opencode/nemotron-3.5-lightning-free', 'nemotron-3.5-lightning (no contestaba)'],
  ['', 'El que tenga configurado opencode'],
];

const MODELOS_CLAUDE: [string, string][] = [
  ['', 'Modelo por defecto'],
  ['opus', 'opus'],
  ['sonnet', 'sonnet'],
  ['haiku', 'haiku'],
];

/** Y con qué modelo. Cada sistema tiene los suyos. */
const MODELOS_AGENTE: Record<string, [string, string][]> = {
  opencode: MODELOS_OPENCODE,
  sdk: MODELOS_CLAUDE,
  claude: MODELOS_CLAUDE,
  '': MODELOS_CLAUDE,
};

export const AgentesTab: React.FC<{ onEncargado: () => void }> = ({ onEncargado }) => {
  const [tarea, setTarea] = useState('');
  const [aviso, setAviso] = useState('');
  const [encargos, setEncargos] = useState<Trabajo[]>([]);
  // Se arranca en opencode y en su primer modelo gratis: lo que no cuesta
  // suscripción es lo que debe salir puesto, no lo que hay que ir a buscar.
  const [motor, setMotor] = useState(SISTEMAS_AGENTE[0][0]);
  const [modelo, setModelo] = useState(MODELOS_OPENCODE[0][0]);
  /** Qué bitácoras están abiertas. Fuera del render de cada tarjeta: la lista
   *  se recarga sola y cerrar lo que estás leyendo sería inservible. */
  const [abiertos, setAbiertos] = useState<Set<number>>(new Set());

  const cargar = useCallback(async () => {
    try {
      const datos = { trabajos: await nucleo.trabajos(50) };
      setEncargos(datos.trabajos.filter(t => t.agente === 'dev'));
    } catch (e: any) {
      setAviso(String(e));
    }
  }, []);

  useEffect(() => {
    cargar();
    const t = setInterval(cargar, REFRESCO);
    return () => clearInterval(t);
  }, [cargar]);

  const lanzar = async (e: React.FormEvent) => {
    e.preventDefault();
    const texto = tarea.trim();
    if (!texto) { setAviso('Escribe primero qué tiene que hacer.'); return; }
    // Sin `directorio`: el núcleo trabaja desde la carpeta del usuario y el
    // agente entra en el proyecto que haga falta. Elegir la raíz era el
    // impuesto de cada encargo, y mandaba una URL cuando el proyecto era un
    // servicio (2026-08-26).
    const peticion: Record<string, string> = { texto };
    if (motor) peticion.motor = motor;
    if (modelo) peticion.modelo = modelo;
    try {
      const trabajo = await nucleo.encolar('dev', peticion);
      setTarea('');
      setAviso(`Encargo #${trabajo.id} en marcha. Puede tardar minutos: trabaja solo.`);
      onEncargado();
    } catch (err: any) {
      setAviso('No se pudo lanzar: ' + err);
    }
  };

  // Tres montones, leídos como los lee una persona: lo que va, lo que espera
  // tu decisión y lo que ya terminó. La cola cruda está en su pestaña; aquí
  // solo importa el ciclo de vida de UN encargo.
  const enMarcha = encargos.filter(t => t.estado === 'pendiente' || t.estado === 'en_curso');
  const esperando = encargos.filter(t => t.estado === 'esperando');
  const terminados = encargos.filter(t => !ESTADOS_ABIERTOS.has(t.estado));

  const tarjeta = (t: Trabajo) => (
    <TarjetaTrabajo
      key={t.id}
      t={t}
      onResponder={(id, d) => {
        nucleo.responder(id, d).then(cargar).catch(() => {});
      }}
      extra={
        <div className="pnl-acciones">
          <button
            type="button"
            className="pnl-pildora"
            onClick={() => setAbiertos(previo => {
              const copia = new Set(previo);
              if (copia.has(t.id)) copia.delete(t.id); else copia.add(t.id);
              return copia;
            })}
          >
            {abiertos.has(t.id) ? 'Ocultar actividad' : 'Ver actividad'}
          </button>
          {abiertos.has(t.id) && <Bitacora id={t.id} vivo={ESTADOS_ABIERTOS.has(t.estado)} />}
        </div>
      }
    />
  );

  return (
    <>
      <div className="pnl-tarjeta pnl-agentes-form">
        <div className="pnl-cabeza">Encargos de código</div>
        <div className="pnl-detalle">
          Escribe el encargo como se habla: el proyecto va en el texto
          («en CVScraper…»). El agente trabaja desde tu carpeta de usuario y
          puede entrar en cualquier proyecto: no hay que elegir raíz. Con qué
          trabaja se elige abajo.
        </div>
      </div>

      <form className="pnl-tarjeta pnl-agentes-form" onSubmit={lanzar}>
        <div className="pnl-opciones-agente">
          {/* Al cambiar de sistema, el modelo pasa a ser el PRIMERO del nuevo y
              no vacío: con opencode, vacío significa «el que tenga configurado»,
              que puede ser de pago. */}
          <select
            value={motor}
            onChange={e => {
              const elegido = e.target.value;
              setMotor(elegido);
              setModelo((MODELOS_AGENTE[elegido] ?? MODELOS_CLAUDE)[0][0]);
            }}
          >
            {SISTEMAS_AGENTE.map(([valor, nombre]) => (
              <option key={valor} value={valor}>{nombre}</option>
            ))}
          </select>
          <select value={modelo} onChange={e => setModelo(e.target.value)}>
            {(MODELOS_AGENTE[motor] ?? MODELOS_AGENTE['']).map(([valor, nombre]) => (
              <option key={valor} value={valor}>{nombre}</option>
            ))}
          </select>
        </div>
        <textarea
          value={tarea}
          onChange={e => setTarea(e.target.value)}
          rows={4}
          placeholder="P. ej.: «En Armario, añade un README con cómo arrancarlo»."
        />
        {!tarea && (
          <div className="pnl-ejemplos">
            {EJEMPLOS_ENCARGO.map(ejemplo => (
              <button
                key={ejemplo}
                type="button"
                className="pnl-ficha pnl-ejemplo"
                onClick={() => setTarea(ejemplo)}
              >
                {ejemplo}
              </button>
            ))}
          </div>
        )}

        <div className="pnl-acciones">
          <button type="submit" className="pnl-pildora aprobar">Lanzar encargo</button>
          {aviso && <span className="pnl-detalle">{aviso}</span>}
        </div>
      </form>

      {encargos.length === 0 ? (
        <div className="pnl-nota">
          Ningún encargo todavía. Lanza el primero con un ejemplo de arriba.
        </div>
      ) : (
        <>
          {(enMarcha.length > 0 || esperando.length > 0) && (
            <div className="pnl-seccion">En marcha ({enMarcha.length + esperando.length})</div>
          )}
          {enMarcha.map(tarjeta)}
          {esperando.map(tarjeta)}

          {terminados.length > 0 && (
            <div className="pnl-seccion">Terminados ({terminados.length})</div>
          )}
          {terminados.map(tarjeta)}
        </>
      )}
    </>
  );
};
