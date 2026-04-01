import React, { useState } from 'react';
import { defaultConfig } from '../lib/config';

interface Props {
  onClose: () => void;
}

export const Settings: React.FC<Props> = ({ onClose }) => {
  const [apiKey, setApiKey] = useState(defaultConfig.geminiApiKey);
  const [voice, setVoice] = useState(defaultConfig.voiceName);

  const handleSave = () => {
    defaultConfig.geminiApiKey = apiKey;
    defaultConfig.voiceName = voice;
    onClose();
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

        <div className="modal-actions">
          <button onClick={onClose}>Cancelar</button>
          <button className="primary" onClick={handleSave}>Guardar</button>
        </div>
      </div>
    </div>
  );
};
