import { useEffect, useState, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { PerseoFace } from './components/PerseoFace';
import { Settings } from './components/Settings';
import { Panel } from './components/Panel';
import { geminiClient } from './lib/gemini-live';
import { audioManager } from './lib/audio-manager';
import { audioPlayer } from './lib/audio-player';
import { cameraManager } from './lib/camera-manager';
import { screenManager } from './lib/screen-manager';
import { defaultConfig, cargarAjustesPersistidos } from './lib/config';

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

// Cuadrícula: el panel — cola, correo, memoria y estado — en esta misma
// ventana. Ver components/Panel.tsx y src-tauri/src/panel.rs.
const IconPanel = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="7" height="9" rx="1"></rect><rect x="14" y="3" width="7" height="5" rx="1"></rect><rect x="14" y="12" width="7" height="9" rx="1"></rect><rect x="3" y="16" width="7" height="5" rx="1"></rect></svg>
);

// ── Types ──
// `abierto` marca un mensaje que aún está recibiendo fragmentos de transcripción.
type TranscriptMsg = { id: string; text: string; type: 'ai' | 'user' | 'system'; abierto?: boolean };

const MAX_MENSAJES = 400;   // tope de memoria de una sesión
const MENSAJES_VISIBLES = 40;

function App() {
  const [connectionState, setConnectionState] = useState<'disconnected' | 'connecting' | 'connected' | 'error'>('disconnected');
  // El oyente de autollamada se registra una vez y vive todo el rato, así que
  // no puede leer `connectionState` del cierre: se le quedaría el valor de
  // cuando se montó. La referencia sí está siempre al día.
  const connectionStateRef = useRef(connectionState);
  connectionStateRef.current = connectionState;
  const [transcripts, setTranscripts] = useState<TranscriptMsg[]>([]);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [screenFrame, setScreenFrame] = useState<string | null>(null);
  const [isScreenSharing, setIsScreenSharing] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [volume, setVolume] = useState(0);
  const [apiKeyReady, setApiKeyReady] = useState(false);
  // El panel es una vista de esta misma ventana, no otra ventana: ver
  // src-tauri/src/panel.rs para por qué no puede ser la interfaz del núcleo.
  const [showPanel, setShowPanel] = useState(false);
  const transcriptRef = useRef<HTMLDivElement>(null);
  const cameraVideoRef = useRef<HTMLVideoElement>(null);

  // Espejo del estado para poder leerlo desde callbacks sin recrearlos.
  // Antes había dos almacenes en paralelo (un ref y un estado) y se
  // desincronizaban: el ref se vaciaba en 'disconnected', que es justo el evento
  // que emite handleReconnect antes de reconectar, así que el historial que se
  // inyectaba al reconectar siempre estaba vacío. Ver H-06.
  const conversacionRef = useRef<TranscriptMsg[]>([]);
  useEffect(() => { conversacionRef.current = transcripts; }, [transcripts]);

  /** Persiste la conversación como Markdown en el vault, donde el RAG la indexa
   *  solo. Antes esto era un console.log con la escritura comentada. Ver H-07. */
  const guardarConversacion = async (mensajes: TranscriptMsg[]) => {
    const utiles = mensajes.filter(m => m.type !== 'system' && m.text.trim());
    if (utiles.length === 0) return;

    try {
      await invoke('ejecutar_herramienta', {
        toolName: 'guardar_conversacion',
        argumentos: JSON.stringify({
          mensajes: utiles.map(m => ({ tipo: m.type, texto: m.text })),
        }),
      });
      console.log(`[Historial] Conversación guardada (${utiles.length} mensajes).`);
    } catch (e) {
      console.error('[Historial] No se pudo guardar la conversación:', e);
    }
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
        // La conversación NO se borra aquí: 'disconnected' también se emite en
        // cada reconexión automática, y borrarla era lo que dejaba sin efecto la
        // reinyección de contexto. Se guarda y se limpia al colgar. Ver H-06.
      }
    };

    geminiClient.onTranscript = (rol, delta, final) => {
      añadirFragmento(rol, delta, final);
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

    // Historial que se reinyecta en el prompt al reconectar. Ahora incluye
    // también lo que dijo el usuario, porque la transcripción de entrada ya
    // está activada (H-05); antes solo se recuperaba el monólogo de Perseo.
    geminiClient.getConversationHistory = () => {
      if (!defaultConfig.saveHistoryEnabled) return "";
      const texto = conversacionRef.current
        .filter(m => m.type !== 'system' && m.text.trim())
        .map(m => (m.type === 'ai' ? `Perseo: "${m.text}"` : `Señor Persus: "${m.text}"`))
        .join('\n');

      // Solo el final, para no inflar el prompt sin límite.
      return texto.slice(-2000);
    };

    return () => {
      geminiClient.disconnect();
      audioManager.stop();
      cameraManager.stop();
      screenManager.stop();
    };
  }, []);

  // «Panel» en el menú de la bandeja: la ventana ya la ha sacado Rust, aquí
  // solo hay que cambiar de vista.
  useEffect(() => {
    const dejarDeEscuchar = listen('abrir-panel', () => setShowPanel(true));
    return () => { dejarDeEscuchar.then(quitar => quitar()).catch(() => {}); };
  }, []);

  // Cargar la API Key desde Rust (almacén local o variable de entorno del
  // sistema). Ya no viaja dentro del bundle — ver H-17.
  useEffect(() => {
    (async () => {
      try {
        await cargarAjustesPersistidos();
        const clave = await invoke<string>('obtener_api_key');
        defaultConfig.geminiApiKey = clave;
        if (!clave) addTranscript('system', 'No hay API Key configurada. Pulsa ⚙ para añadirla.');
      } catch (e) {
        addTranscript('system', `No se pudieron cargar los ajustes: ${e}`);
      } finally {
        setApiKeyReady(true);
      }
    })();
  }, []);

  // Auto-llamada cuando la app se abre desde el detector de aplausos.
  // La señal se consume en tiempo de ejecución desde Rust (un fichero marcador
  // que se borra al leerlo), no con un JSON importado estáticamente: Vite
  // congelaba ese valor al compilar, así que en producción el disparo por
  // aplausos no funcionaba, y además nunca volvía a false. Ver H-09.
  useEffect(() => {
    if (!apiKeyReady || connectionState !== 'disconnected') return;

    let cancelado = false;
    invoke<boolean>('consumir_autollamada').then(pedida => {
      if (pedida && !cancelado) {
        setTimeout(() => { if (!cancelado) handleCall(); }, 1500);
      }
    }).catch(e => console.warn('[AutoLlamada] No se pudo comprobar la señal:', e));

    return () => { cancelado = true; };
  }, [apiKeyReady]);

  // Y la autollamada con la app ya abierta. Desde que Perseo vive en la bandeja
  // el arranque no vuelve a ocurrir, así que el efecto de arriba —que solo mira
  // el marcador al montarse— dejaría la palabra clave sin efecto. Rust vigila
  // el fichero y avisa por evento; aquí solo se atiende.
  useEffect(() => {
    if (!apiKeyReady) return;

    let cancelado = false;
    const dejarDeEscuchar = listen('perseo://autollamada', () => {
      if (cancelado) return;
      // Si ya está en llamada no se hace nada: la palabra clave sirve para
      // empezar una conversación, no para cortar la que hay.
      if (connectionStateRef.current !== 'disconnected') return;
      setTimeout(() => { if (!cancelado) handleCall(); }, 1500);
    });

    return () => {
      cancelado = true;
      dejarDeEscuchar.then(quitar => quitar()).catch(() => {});
    };
  }, [apiKeyReady]);

  useEffect(() => {
    if (transcriptRef.current) transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight;
  }, [transcripts]);

  useEffect(() => {
    if (cameraVideoRef.current && cameraStream) cameraVideoRef.current.srcObject = cameraStream;
  }, [cameraStream]);

  const addTranscript = (type: TranscriptMsg['type'], text: string) => {
    setTranscripts(prev =>
      [...prev, { id: `${Date.now()}-${Math.random()}`, text, type }].slice(-MAX_MENSAJES)
    );
  };

  /** Acumula un fragmento de transcripción sobre el mensaje abierto del mismo
   *  hablante, o abre uno nuevo. Las transcripciones llegan troceadas, así que
   *  una línea por fragmento produciría un muro ilegible. */
  const añadirFragmento = (rol: 'ai' | 'user', delta: string, final: boolean) => {
    setTranscripts(prev => {
      if (final) {
        return prev.map(m => (m.type === rol && m.abierto ? { ...m, abierto: false } : m));
      }
      if (!delta) return prev;

      const ultimo = prev[prev.length - 1];
      if (ultimo && ultimo.type === rol && ultimo.abierto) {
        return [...prev.slice(0, -1), { ...ultimo, text: ultimo.text + delta }];
      }
      return [
        ...prev,
        { id: `${Date.now()}-${Math.random()}`, text: delta, type: rol, abierto: true },
      ].slice(-MAX_MENSAJES);
    });
  };

  const handleCall = () => {
    if (!defaultConfig.geminiApiKey) { addTranscript('system', 'API Key no configurada. Pulsa ⚙.'); return; }
    audioPlayer.initialize();
    audioManager.start();
    geminiClient.connect();
  };

  const handleHangup = async () => {
    geminiClient.disconnect();
    audioManager.stop();
    audioPlayer.clearQueue();
    setIsSpeaking(false);

    // Colgar sí cierra la conversación de verdad: aquí es donde se guarda y se
    // limpia, no en cada 'disconnected'. Ver H-06 y H-07.
    if (defaultConfig.saveHistoryEnabled) {
      await guardarConversacion(conversacionRef.current);
    }
    setTranscripts([]);
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

      {/* El panel tapa la llamada, no la corta: la sesión de voz sigue viva
          detrás, así que volver es instantáneo y no se pierde la conversación. */}
      {showPanel && <Panel onCerrar={() => setShowPanel(false)} />}

      {/* Panel y ajustes */}
      <button
        className="settings-btn panel-btn"
        title="Panel: cola, correo y estado"
        onClick={() => setShowPanel(true)}
      >
        <IconPanel />
      </button>
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
        {transcripts.slice(-MENSAJES_VISIBLES).map(msg => (
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

      {showSettings && (
        <Settings onClose={() => setShowSettings(false)} llamadaActiva={isActive} />
      )}
    </div>
  );
}

export default App;
