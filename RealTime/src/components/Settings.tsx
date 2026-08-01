import React, { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { defaultConfig } from '../lib/config';

interface Props {
  onClose: () => void;
}

export const Settings: React.FC<Props> = ({ onClose }) => {
  const [apiKey, setApiKey] = useState(defaultConfig.geminiApiKey);
  const [voice, setVoice] = useState(defaultConfig.voiceName);
  const [guardando, setGuardando] = useState(false);
  const [error, setError] = useState('');

  // La clave se persiste en Rust, no en el bundle ni en localStorage. Ver H-17.
  const handleSave = async () => {
    setGuardando(true);
    setError('');
    try {
      await invoke('guardar_api_key', { clave: apiKey });
      defaultConfig.geminiApiKey = apiKey;
      defaultConfig.voiceName = voice;
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
