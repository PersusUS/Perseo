/**
 * Los ajustes de la aplicación de voz.
 *
 * Rehechos el 2026-08-22 por encargo del señor Persus —*«los ajustes en sí no
 * están bien»*— con tres cambios de fondo:
 *
 *  1. **La clave ya no se toca desde aquí.** Era el único sitio de Perseo donde
 *     un secreto se escribía, se enseñaba y se guardaba en un almacén aparte, y
 *     bastaba con abrir una ventana para tenerlo delante. Ahora la clave se pone
 *     donde ya vivía la del núcleo —`<datos>/gemini.txt`, o la variable
 *     `GEMINI_API_KEY`— y esta pantalla solo dice si hay una puesta. El orden en
 *     el que se busca lo decide Rust (`commands.rs::obtener_api_key`).
 *  2. **El aspecto se elige viéndolo.** Tres fichas con lo que cambia cada una,
 *     y el cambio se aplica al pulsar, no al guardar: si hay que guardar para
 *     saber cómo queda, no es elegir, es apostar. Cancelar devuelve el que
 *     había.
 *  3. **El aire es el del puesto de mando**: esquinas marcadas, versalitas
 *     espaciadas y nada redondeado. Antes era un formulario gris en medio de una
 *     pantalla negra.
 */
import React, { useState } from 'react';
import { defaultConfig, guardarAjuste, SYSTEM_PROMPT_POR_DEFECTO, type AspectoLive } from '../lib/config';

interface Props {
  onClose: () => void;
  /** Si hay una llamada en curso, el cambio de voz no se aplica hasta reconectar. */
  llamadaActiva?: boolean;
  /** Aplica el aspecto en caliente, para poder verlo mientras se elige. */
  onAspecto?: (aspecto: AspectoLive) => void;
}

/** Los tres aspectos del modo live. Ver components/Escenografia.tsx. */
const ASPECTOS: { valor: AspectoLive; nombre: string; que: string }[] = [
  { valor: 'mira', nombre: 'Mira', que: 'Perseo en el centro, anillos girando y lecturas en las esquinas.' },
  { valor: 'mando', nombre: 'Puesto de mando', que: 'Tres columnas: instrumentos, Perseo y la bitácora de la llamada.' },
  { valor: 'cartel', nombre: 'Cartel', que: 'Descentrada, con el estado en letras grandes y el kanji al fondo.' },
];

const VOCES: { valor: string; nombre: string; como: string }[] = [
  { valor: 'Charon', nombre: 'Charon', como: 'Muy grave' },
  { valor: 'Orus', nombre: 'Orus', como: 'Grave' },
  { valor: 'Fenrir', nombre: 'Fenrir', como: 'Media' },
  { valor: 'Puck', nombre: 'Puck', como: 'Clara' },
];

export const Settings: React.FC<Props> = ({ onClose, llamadaActiva = false, onAspecto }) => {
  const [voice, setVoice] = useState(defaultConfig.voiceName);
  const [prompt, setPrompt] = useState(defaultConfig.systemPrompt);
  const [guardarHistorial, setGuardarHistorial] = useState(defaultConfig.saveHistoryEnabled);
  const [pantallaAuto, setPantallaAuto] = useState(defaultConfig.pantallaAuto);
  const [aspecto, setAspecto] = useState<AspectoLive>(defaultConfig.aspectoLive);
  const [guardando, setGuardando] = useState(false);
  const [error, setError] = useState('');

  // El que había al abrir, para devolverlo si se cancela: el aspecto se aplica
  // al pulsar la ficha, así que sin esto «Cancelar» dejaría el cambio hecho.
  const [aspectoOriginal] = useState<AspectoLive>(defaultConfig.aspectoLive);

  const voiceCambiada = voice !== defaultConfig.voiceName;
  const hayClave = !!defaultConfig.geminiApiKey;

  const elegirAspecto = (valor: AspectoLive) => {
    setAspecto(valor);
    onAspecto?.(valor);
  };

  const cerrarSinGuardar = () => {
    if (aspecto !== aspectoOriginal) onAspecto?.(aspectoOriginal);
    onClose();
  };

  // Todo se persiste en el almacén local que gestiona Rust: antes esto solo
  // mutaba un objeto en memoria y se perdía al cerrar la app. Ver H-08.
  const handleSave = async () => {
    setGuardando(true);
    setError('');
    try {
      await guardarAjuste('voiceName', voice);
      await guardarAjuste('systemPrompt', prompt);
      await guardarAjuste('saveHistoryEnabled', guardarHistorial);
      await guardarAjuste('pantallaAuto', pantallaAuto);
      await guardarAjuste('aspectoLive', aspecto);
      onClose();
    } catch (e) {
      setError(`No se pudo guardar: ${e}`);
      setGuardando(false);
    }
  };

  return (
    <div className="ajustes-fondo" onClick={cerrarSinGuardar}>
      <div className="ajustes" onClick={e => e.stopPropagation()}>
        <header className="ajustes-cabeza">
          <h3>Ajustes</h3>
          <button className="ajustes-cerrar" onClick={cerrarSinGuardar} title="Cerrar">✕</button>
        </header>

        <div className="ajustes-cuerpo">
          <section className="ajustes-bloque">
            <div className="ajustes-titulo">Aspecto del modo live</div>
            <div className="ajustes-fichas">
              {ASPECTOS.map(a => (
                <button
                  key={a.valor}
                  className={`ajustes-ficha ${aspecto === a.valor ? 'elegida' : ''}`}
                  onClick={() => elegirAspecto(a.valor)}
                >
                  <span className="ajustes-ficha-nombre">{a.nombre}</span>
                  <span className="ajustes-ficha-que">{a.que}</span>
                </button>
              ))}
            </div>
            <p className="ajustes-nota">Se aplica al pulsar. Cancelar devuelve el que había.</p>
          </section>

          <section className="ajustes-bloque">
            <div className="ajustes-titulo">Voz</div>
            <div className="ajustes-fichas">
              {VOCES.map(v => (
                <button
                  key={v.valor}
                  className={`ajustes-ficha estrecha ${voice === v.valor ? 'elegida' : ''}`}
                  onClick={() => setVoice(v.valor)}
                >
                  <span className="ajustes-ficha-nombre">{v.nombre}</span>
                  <span className="ajustes-ficha-que">{v.como}</span>
                </button>
              ))}
            </div>
            {llamadaActiva && voiceCambiada && (
              <p className="ajustes-nota">La voz se aplicará al volver a llamar.</p>
            )}
          </section>

          <section className="ajustes-bloque">
            <div className="ajustes-titulo">Al colgar</div>
            <label className="ajustes-interruptor">
              <input
                type="checkbox"
                checked={guardarHistorial}
                onChange={e => setGuardarHistorial(e.target.checked)}
              />
              <span>Guardar la conversación en el vault</span>
            </label>
          </section>

          <section className="ajustes-bloque">
            <div className="ajustes-titulo">Durante la llamada</div>
            <label className="ajustes-interruptor">
              <input
                type="checkbox"
                checked={pantallaAuto}
                onChange={e => setPantallaAuto(e.target.checked)}
              />
              <span>Compartir mi pantalla con Perseo al conectar</span>
            </label>
            <p className="ajustes-nota">
              Apagado, Perseo no ve nada hasta que se lo pidas: te preguntará y, con tu
              sí, empezará a mirar por su cuenta.
            </p>
          </section>

          <section className="ajustes-bloque">
            <div className="ajustes-titulo">
              Instrucciones del sistema
              <button
                className="ajustes-restaurar"
                onClick={() => setPrompt(SYSTEM_PROMPT_POR_DEFECTO)}
                disabled={prompt === SYSTEM_PROMPT_POR_DEFECTO}
              >
                Restaurar
              </button>
            </div>
            <textarea value={prompt} onChange={e => setPrompt(e.target.value)} rows={10} />
          </section>

          <section className="ajustes-bloque">
            <div className="ajustes-titulo">Clave del modelo</div>
            <p className="ajustes-nota">
              {hayClave
                ? 'Hay una clave puesta. No se edita desde aquí a propósito.'
                : 'No hay clave: sin ella no se puede llamar.'}
              {' '}Se lee de <code>GEMINI_API_KEY</code> o del fichero{' '}
              <code>perseo_core/datos/gemini.txt</code>, que es el mismo que usa el núcleo.
            </p>
          </section>
        </div>

        {error && <p className="ajustes-error">{error}</p>}

        <footer className="ajustes-pie">
          <button onClick={cerrarSinGuardar} disabled={guardando}>Cancelar</button>
          <button className="principal" onClick={handleSave} disabled={guardando}>
            {guardando ? 'Guardando…' : 'Guardar'}
          </button>
        </footer>
      </div>
    </div>
  );
};
