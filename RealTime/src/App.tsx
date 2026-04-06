import { useEffect, useState, useRef } from 'react';
import { PerseoFace } from './components/PerseoFace';
import { Settings } from './components/Settings';
import { geminiClient } from './lib/gemini-live';
import { audioManager } from './lib/audio-manager';
import { audioPlayer } from './lib/audio-player';
import { cameraManager } from './lib/camera-manager';
import { screenManager } from './lib/screen-manager';
import { defaultConfig } from './lib/config';
import autoCallData from './autocall.json';

// ── SVG Icons (clean, white, stroke-only) ──

const IconMic = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="22"></line></svg>
);

const IconMicOff = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="2" y1="2" x2="22" y2="22"></line><path d="M18.89 13.23A7.12 7.12 0 0 0 19 12v-2"></path><path d="M5 10v2a7 7 0 0 0 12 5"></path><path d="M15 9.34V5a3 3 0 0 0-5.68-1.33"></path><path d="M9 9v3a3 3 0 0 0 5.12 1.67"></path><line x1="12" y1="19" x2="12" y2="22"></line></svg>
);

const IconCamera = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg>
);

const IconCameraOff = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="2" y1="2" x2="22" y2="22"></line><path d="M21 21H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h3m3-3h6l2 3h4a2 2 0 0 1 2 2v9.34M14.54 14.54a3 3 0 0 1-4.08-4.08"></path></svg>
);

const IconScreen = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect><line x1="8" y1="21" x2="16" y2="21"></line><line x1="12" y1="17" x2="12" y2="21"></line></svg>
);

const IconPhone = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"></path></svg>
);

const IconPhoneOff = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M10.68 13.31a16 16 0 0 0 3.41 2.6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7 2 2 0 0 1 1.72 2v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.42 19.42 0 0 1-3.33-2.67m-2.67-3.34a19.79 19.79 0 0 1-3.07-8.63A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91"></path><line x1="22" y1="2" x2="2" y2="22"></line></svg>
);

const IconSettings = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
);

// ── Types ──
type TranscriptMsg = { id: string; text: string; type: 'ai' | 'user' | 'system' };

function App() {
  const [connectionState, setConnectionState] = useState<'disconnected' | 'connecting' | 'connected' | 'error'>('disconnected');
  const [transcripts, setTranscripts] = useState<TranscriptMsg[]>([]);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [screenFrame, setScreenFrame] = useState<string | null>(null);
  const [isScreenSharing, setIsScreenSharing] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [volume, setVolume] = useState(0);
  const transcriptRef = useRef<HTMLDivElement>(null);
  const cameraVideoRef = useRef<HTMLVideoElement>(null);
  const fullConversationRef = useRef<TranscriptMsg[]>([]);

  // Función preparada para guardar el historial completo
  const saveConversationHistory = (history: TranscriptMsg[]) => {
    // Aquí a futuro implementaremos la base de datos o el guardado en local (IndexedDB, archivos, etc.)
    console.log('[History System] Guardando conversación anterior...', history);
    // localStorage.setItem('perseo-history-latest', JSON.stringify(history));
  };

  useEffect(() => {
    geminiClient.onConnectionStateChange = (state) => {
      setConnectionState(state as any);
      if (state === 'connected') {
        audioManager.start(); // Reactivar el micrófono al reconectar
        if (defaultConfig.cameraEnabled) cameraManager.start();
        if (defaultConfig.screenEnabled) {
          screenManager.start();
          setIsScreenSharing(true);
        }
        addTranscript('system', 'Conectado.');
      } else if (state === 'disconnected' || state === 'error') {
        audioManager.stop();
        cameraManager.stop();
        screenManager.stop();
        setCameraStream(null);
        setScreenFrame(null);
        setIsScreenSharing(false);
        setIsSpeaking(false);
        
        // Guardar conversación si está habilitado mediante config antes de limpiar
        if (defaultConfig.saveHistoryEnabled && fullConversationRef.current.length > 0) {
          saveConversationHistory(fullConversationRef.current);
          fullConversationRef.current = []; // Reiniciamos el registro de la sesión
        }
      }
    };

    geminiClient.onTranscriptChange = (text) => {
      addTranscript('ai', text);
    };

    geminiClient.onError = (msg) => addTranscript('system', msg);
    audioManager.onStreamReady = () => addTranscript('system', 'Micrófono activo.');
    audioManager.onError = (err) => addTranscript('system', err);
    cameraManager.onStreamReady = (stream) => setCameraStream(stream);
    screenManager.onFrameReady = (base64) => setScreenFrame(base64);

    audioPlayer.onVolumeChange = (v) => {
      setVolume(v);
      if (v > 0.05) {
        setIsSpeaking(true);
      } else {
        setIsSpeaking(false);
      }
    };

    // Asignamos la función para obtener el historial a pasar a Gemini en cada conexión/re-conexión
    geminiClient.getConversationHistory = () => {
      if (!defaultConfig.saveHistoryEnabled) return "";
      const pastMessages = fullConversationRef.current
        .filter(m => m.type === 'ai') // Idealmente podemos filtrar de ambos si la API transcribiese la voz del usuario
        .map(m => `Tú (Perseo) dijiste: "${m.text}"`)
        .join('\n');
      
      // Solo tomamos el final para que no ocupe un prompt excesivamente gigante.
      return pastMessages.slice(-2000);
    };

    return () => {
      geminiClient.disconnect();
      audioManager.stop();
      cameraManager.stop();
      screenManager.stop();
    };
  }, []);

  // Efecto adicional para la auto-llamada cuando se abre desde los aplausos
  useEffect(() => {
    if (autoCallData.autoCall && connectionState === 'disconnected') {
      const timer = setTimeout(() => {
        handleCall();
      }, 1500); // 1.5s de gracia tras cargar la UI
      return () => clearTimeout(timer);
    }
  }, []);

  useEffect(() => {
    if (transcriptRef.current) transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight;
  }, [transcripts]);

  useEffect(() => {
    if (cameraVideoRef.current && cameraStream) cameraVideoRef.current.srcObject = cameraStream;
  }, [cameraStream]);

  const addTranscript = (type: TranscriptMsg['type'], text: string) => {
    const newMessage: TranscriptMsg = { id: `${Date.now()}-${Math.random()}`, text, type };
    fullConversationRef.current.push(newMessage);
    setTranscripts(prev => [...prev.slice(-40), newMessage]);
  };

  const handleCall = () => {
    if (!defaultConfig.geminiApiKey) { addTranscript('system', 'API Key no configurada. Pulsa ⚙.'); return; }
    audioPlayer.initialize();
    audioManager.start();
    geminiClient.connect();
  };

  const handleHangup = () => {
    geminiClient.disconnect();
    audioManager.stop();
    audioPlayer.clearQueue();
    setIsSpeaking(false);
  };

  const toggleMute = () => {
    if (isMuted) { audioManager.start(); setIsMuted(false); }
    else { audioManager.stop(); setIsMuted(true); }
  };

  const toggleCamera = () => {
    if (cameraStream) { cameraManager.stop(); setCameraStream(null); defaultConfig.cameraEnabled = false; }
    else { defaultConfig.cameraEnabled = true; cameraManager.start(); }
  };

  const toggleScreen = () => {
    if (isScreenSharing) { screenManager.stop(); setIsScreenSharing(false); setScreenFrame(null); defaultConfig.screenEnabled = false; }
    else { setIsScreenSharing(true); defaultConfig.screenEnabled = true; screenManager.start(); }
  };

  const isActive = connectionState === 'connected' || connectionState === 'connecting';
  const isConnected = connectionState === 'connected';

  const statusText = connectionState === 'disconnected' ? ''
    : connectionState === 'connecting' ? 'Conectando...'
    : connectionState === 'error' ? 'Error de conexión'
    : isMuted ? 'Micrófono silenciado'
    : isSpeaking ? 'Perseo está hablando'
    : 'Perseo está escuchando';

  return (
    <div className="app-container">
      <div className="bg-art" />

      {/* Settings */}
      <button className="settings-btn" onClick={() => setShowSettings(true)}>
        <IconSettings />
      </button>

      {/* PIP Containers */}
      <div className="pip-container">
        {cameraStream && (
          <div className="camera-pip">
            <video ref={cameraVideoRef} autoPlay playsInline muted />
          </div>
        )}
        {screenFrame && (
          <div className="screen-pip">
            <img src={`data:image/jpeg;base64,${screenFrame}`} alt="Screen display" />
          </div>
        )}
      </div>

      {/* Center */}
      <div className="call-center">
        <div className="perseo-face-container">
          <PerseoFace
            isSpeaking={isSpeaking}
            isListening={isConnected && !isSpeaking}
            isConnecting={connectionState === 'connecting'}
            volume={volume}
          />
        </div>
        <div className="perseo-text-container">
          <span className="perseo-name">Perseo</span>
          <div className="perseo-status-wrapper">
            {statusText && <span className="perseo-status">{statusText}</span>}
          </div>
        </div>
      </div>

      {/* Transcript */}
      <div className="transcript-overlay" ref={transcriptRef}>
        {transcripts.map(msg => (
          <div key={msg.id} className={`transcript-line ${msg.type}`}>{msg.text}</div>
        ))}
      </div>

      {/* Controls */}
      <div className="controls-bar">
        {isActive && (
          <>
            <button className={`ctrl-btn ${isMuted ? 'muted' : ''}`} onClick={toggleMute} title={isMuted ? 'Activar micro' : 'Silenciar'}>
              {isMuted ? <IconMicOff /> : <IconMic />}
            </button>
            <button className={`ctrl-btn ${cameraStream ? 'active' : ''}`} onClick={toggleCamera} title="Cámara">
              {cameraStream ? <IconCamera /> : <IconCameraOff />}
            </button>
            <button className={`ctrl-btn ${isScreenSharing ? 'active' : ''}`} onClick={toggleScreen} title="Pantalla">
              <IconScreen />
            </button>
          </>
        )}

        {!isActive ? (
          <button className="ctrl-btn call-btn" onClick={handleCall} title="Llamar">
            <IconPhone />
          </button>
        ) : (
          <button className="ctrl-btn hangup-btn" onClick={handleHangup} title="Colgar">
            <IconPhoneOff />
          </button>
        )}
      </div>

      {showSettings && <Settings onClose={() => setShowSettings(false)} />}
    </div>
  );
}

export default App;
