import React, { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { defaultConfig, guardarAjuste, SYSTEM_PROMPT_POR_DEFECTO } from '../lib/config';

interface Props {
  onClose: () => void;
  /** Si hay una llamada en curso, el cambio de voz no se aplica hasta reconectar. */
  llamadaActiva?: boolean;
}

export const Settings: React.FC<Props> = ({ onClose, llamadaActiva = false }) => {
  const [apiKey, setApiKey] = useState(defaultConfig.geminiApiKey);
  const [voice, setVoice] = useState(defaultConfig.voiceName);
  const [prompt, setPrompt] = useState(defaultConfig.systemPrompt);
  const [guardarHistorial, setGuardarHistorial] = useState(defaultConfig.saveHistoryEnabled);
  const [guardando, setGuardando] = useState(false);
  const [error, setError] = useState('');

  const voiceCambiada = voice !== defaultConfig.voiceName;

  // Todo se persiste en el almacén local que gestiona Rust: antes esto solo
  // mutaba un objeto en memoria y se perdía al cerrar la app. Ver H-08.
  const handleSave = async () => {
    setGuardando(true);
    setError('');
    try {
      await invoke('guardar_api_key', { clave: apiKey });
      defaultConfig.geminiApiKey = apiKey;

      await guardarAjuste('voiceName', voice);
      await guardarAjuste('systemPrompt', prompt);
      await guardarAjuste('saveHistoryEnabled', guardarHistorial);
      onClose();
    } catch (e) {
      setError(`No se pudo guardar: ${e}`);
      setGuardando(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-panel" onClick={e => e.stopPropagation()}>
        <h3>Ajustes</h3>

        <div>
          <label>Clave API</label>
          <input
            type="password"
            value={apiKey}
            onChange={e => setApiKey(e.target.value)}
            placeholder="AIzaSy..."
          />
        </div>

        <div>
          <label>Voz</label>
          <select value={voice} onChange={e => setVoice(e.target.value)}>
            <option value="Charon">Charon — Muy grave</option>
            <option value="Orus">Orus — Grave</option>
            <option value="Fenrir">Fenrir — Media</option>
            <option value="Puck">Puck — Clara</option>
          </select>
          {llamadaActiva && voiceCambiada && (
            <p style={{ fontSize: '0.8em', opacity: 0.75, margin: '4px 0 0' }}>
              La voz se aplicará al volver a llamar.
            </p>
          )}
        </div>

        <div>
          <label>Instrucciones del sistema</label>
          <textarea
            value={prompt}
            onChange={e => setPrompt(e.target.value)}
            rows={8}
            style={{ width: '100%', fontFamily: 'inherit', fontSize: '0.85em', resize: 'vertical' }}
          />
          <button
            onClick={() => setPrompt(SYSTEM_PROMPT_POR_DEFECTO)}
            style={{ fontSize: '0.8em', marginTop: 4 }}
            disabled={prompt === SYSTEM_PROMPT_POR_DEFECTO}
          >
            Restaurar el original
          </button>
        </div>

        <div>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <input
              type="checkbox"
              checked={guardarHistorial}
              onChange={e => setGuardarHistorial(e.target.checked)}
            />
            Guardar las conversaciones en el vault al colgar
          </label>
        </div>

        {error && <p style={{ color: '#f85149', fontSize: '0.85em' }}>{error}</p>}

        <div className="modal-actions">
          <button onClick={onClose} disabled={guardando}>Cancelar</button>
          <button className="primary" onClick={handleSave} disabled={guardando}>
            {guardando ? 'Guardando…' : 'Guardar'}
          </button>
        </div>
      </div>
    </div>
  );
};
