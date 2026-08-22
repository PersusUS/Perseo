import { useEffect, useState, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { PerseoFace } from './components/PerseoFace';
import { Settings } from './components/Settings';
import { Panel } from './components/Panel';
import { Proyectos } from './components/Proyectos';
import { Escenografia, comoReloj, type Fase } from './components/Escenografia';
import { Marco } from './components/Marco';
import {
  IconCamera, IconCameraOff, IconMic, IconMicOff, IconPhone, IconPhoneOff, IconScreen,
} from './components/Iconos';
import { geminiClient } from './lib/gemini-live';
import { audioManager } from './lib/audio-manager';
import { audioPlayer } from './lib/audio-player';
import { cameraManager } from './lib/camera-manager';
import { screenManager } from './lib/screen-manager';
import { defaultConfig, cargarAjustesPersistidos, type AspectoLive } from './lib/config';

// ── Types ──
// `abierto` marca un mensaje que aún está recibiendo fragmentos de transcripción.
// `hora` se sella al abrir el mensaje, no al cerrarlo: es cuando se dijo. Solo
// se enseña en el aspecto «mando», donde la transcripción es una bitácora.
type TranscriptMsg = {
  id: string;
  text: string;
  type: 'ai' | 'user' | 'system';
  abierto?: boolean;
  hora: string;
};

const ahoraCorta = () =>
  new Date().toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit', hour12: false });

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
  // El volumen NO es estado de React. Llega a 60 por segundo desde el
  // reproductor (`audio-player.ts` lo mide con un rAF), y con `useState` eso
  // repintaba la aplicación entera sesenta veces por segundo y hacía que la
  // cara diera saltos. Ahora se guarda en una referencia, se suaviza en el
  // propio rAF y sale por una variable de CSS, que el navegador aplica sin
  // pasar por React. Lo único que sigue siendo estado es «habla o no habla»,
  // que cambia dos veces por frase y no sesenta veces por segundo.
  const volumenCrudo = useRef(0);
  const ultimaMedida = useRef(0);
  const contenedorRef = useRef<HTMLDivElement>(null);
  const [apiKeyReady, setApiKeyReady] = useState(false);
  // El panel es una vista de esta misma ventana, no otra ventana: ver
  // src-tauri/src/panel.rs para por qué no puede ser la interfaz del núcleo.
  const [showPanel, setShowPanel] = useState(false);
  const [showProyectos, setShowProyectos] = useState(false);
  // El aspecto del modo live (T-11). Se elige en Ajustes y se guarda con el
  // resto: aquí hace falta como estado —y no leyendo `defaultConfig`— porque
  // cambiarlo tiene que repintar la pantalla sin cerrar la aplicación.
  const [aspecto, setAspecto] = useState<AspectoLive>(defaultConfig.aspectoLive);
  // Segundos que lleva abierta esta llamada. Cuenta desde que el enlace queda
  // conectado, no desde que se pulsa: entre las dos cosas hay una negociación
  // que a veces falla.
  const [sesion, setSesion] = useState(0);
  const sesionDesde = useRef<number | null>(null);
  // Lo que Perseo ha pedido hacer y está parado esperando un sí. Ver H-51: la
  // pregunta vivía solo en el panel, que en mitad de una llamada nadie mira.
  const [pendientes, setPendientes] = useState<{ id: number; pregunta: string }[]>([]);
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
        if (sesionDesde.current === null) sesionDesde.current = Date.now();
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

    geminiClient.onAprobacionPendiente = (id, pregunta) => {
      setPendientes(prev => (prev.some(p => p.id === id) ? prev : [...prev, { id, pregunta }]));
    };
    audioManager.onStreamReady = () => addTranscript('system', 'Micrófono activo.');
    audioManager.onError = (err) => addTranscript('system', err);
    cameraManager.onStreamReady = (stream) => setCameraStream(stream);
    screenManager.onFrameReady = (base64) => setScreenFrame(base64);

    audioPlayer.onVolumeChange = (v) => {
      volumenCrudo.current = v;
      ultimaMedida.current = Date.now();
      // Dos umbrales y no uno: con un único corte en 0,05 el estado parpadeaba
      // en cada pausa entre palabras.
      setIsSpeaking(previo => (previo ? v > 0.02 : v > 0.08));
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
        setAspecto(defaultConfig.aspectoLive);
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

  // El suavizado de la voz. La medida cruda es la energía de cada trozo de
  // audio: sube y baja con cada sílaba, y aplicada tal cual a una escala hacía
  // que la cara vibrase. Se persigue el valor con una constante distinta al
  // subir que al bajar —rápido al empezar a hablar, lento al callar—, que es
  // como se comporta un vúmetro y por qué se lee bien.
  useEffect(() => {
    let suave = 0;
    let cuadro = 0;
    const pintar = () => {
      // Cuando Perseo termina de hablar, el reproductor deja de medir: sin esto
      // la última medida se quedaría clavada y la cara, hinchada.
      const callado = Date.now() - ultimaMedida.current > 200;
      const objetivo = callado ? 0 : volumenCrudo.current;
      suave += (objetivo - suave) * (objetivo > suave ? 0.22 : 0.06);
      contenedorRef.current?.style.setProperty('--vol', suave.toFixed(3));
      cuadro = requestAnimationFrame(pintar);
    };
    cuadro = requestAnimationFrame(pintar);
    return () => cancelAnimationFrame(cuadro);
  }, []);

  // El cronómetro de la llamada. Cuenta desde una marca de tiempo y no sumando
  // segundos, porque 'disconnected' se emite también en cada reconexión
  // automática (H-06) y sumando se pondría a cero a media conversación. La
  // marca solo se borra al colgar, que es cuando la llamada acaba de verdad.
  useEffect(() => {
    const t = setInterval(() => {
      const desde = sesionDesde.current;
      setSesion(desde ? Math.floor((Date.now() - desde) / 1000) : 0);
    }, 1000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    if (cameraVideoRef.current && cameraStream) cameraVideoRef.current.srcObject = cameraStream;
  }, [cameraStream]);

  const addTranscript = (type: TranscriptMsg['type'], text: string) => {
    setTranscripts(prev =>
      [...prev, { id: `${Date.now()}-${Math.random()}`, text, type, hora: ahoraCorta() }].slice(-MAX_MENSAJES)
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
        { id: `${Date.now()}-${Math.random()}`, text: delta, type: rol, abierto: true, hora: ahoraCorta() },
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
    sesionDesde.current = null;
    setSesion(0);
    // Colgar cierra también las preguntas sin contestar: siguen vivas en la
    // cola, y ahí es donde tiene sentido mirarlas cuando ya no hay llamada.
    setPendientes([]);

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

  /** Contesta a un trabajo parado sin salir de la llamada. El sí lo sigue dando
   *  una persona, que es lo que pide la política; lo único que cambia es que la
   *  pregunta aparece donde estás mirando. */
  const responderPendiente = async (id: number, decision: 'aprobar' | 'rechazar') => {
    setPendientes(prev => prev.filter(p => p.id !== id));
    try {
      await invoke('panel_responder', { id, decision });
      addTranscript('system', decision === 'aprobar' ? `Aprobado el #${id}.` : `Rechazado el #${id}.`);
    } catch (e) {
      addTranscript('system', `No se pudo responder al #${id}: ${e}`);
    }
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

  // La fase manda el dibujo; `statusText` manda el texto. Son dos cosas: hay
  // estados que se cuentan distinto (silenciado) y se pintan igual.
  const fase: Fase = !isActive ? 'reposo'
    : connectionState === 'connecting' ? 'conectando'
    : isSpeaking ? 'hablando'
    : 'escuchando';

  const statusText = connectionState === 'disconnected' ? ''
    : connectionState === 'connecting' ? 'Conectando...'
    : connectionState === 'error' ? 'Error de conexión'
    : isMuted ? 'Micrófono silenciado'
    : isSpeaking ? 'Perseo está hablando'
    : 'Perseo está escuchando';

  return (
    // El volumen viaja como variable de CSS (`--vol`, la escribe el rAF de
    // arriba) para que lo lean el anillo, las barras y la cara sin props ni
    // repintados.
    <div ref={contenedorRef} className={`app-container aspecto-${aspecto}`}>
      <div className="bg-art" />

      <Escenografia
        aspecto={aspecto}
        fase={fase}
        estadoTexto={statusText}
        sesion={sesion}
      />

      {/* El panel tapa la llamada, no la corta: la sesión de voz sigue viva
          detrás, así que volver es instantáneo y no se pierde la conversación. */}
      {showPanel && <Panel onCerrar={() => setShowPanel(false)} />}

      {/* El riel de arriba —panel, ajustes y los botones de la ventana— y la
          pestaña de proyectos. Ver components/Marco.tsx. */}
      <Marco
        onPanel={() => setShowPanel(true)}
        onAjustes={() => setShowSettings(true)}
        onProyectos={() => setShowProyectos(v => !v)}
        proyectosAbiertos={showProyectos}
      />

      {/* Lo que Perseo ve: la cámara en vivo y el JPEG que le llega del modelo.
          Son lecturas del borde; cada aspecto las coloca donde no estorban. */}
      <div className="pip-container">
        {cameraStream && (
          <div className="camera-pip">
            <video ref={cameraVideoRef} autoPlay playsInline muted />
            <span className="pip-etiqueta">Cámara</span>
          </div>
        )}
        {screenFrame && (
          <div className="screen-pip">
            <img src={`data:image/jpeg;base64,${screenFrame}`} alt="Pantalla compartida" />
            <span className="pip-etiqueta">Pantalla</span>
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
            aspecto={aspecto}
          />
        </div>
        <div className="perseo-text-container">
          <span className="perseo-name">Perseo</span>
          <div className="perseo-status-wrapper">
            {statusText && <span className="perseo-status">{statusText}</span>}
            {sesion > 0 && <span className="perseo-sesion">{comoReloj(sesion)}</span>}
          </div>
        </div>
      </div>

      {/* Lo que Perseo ha pedido hacer y espera un sí. Sale aquí y no solo en el
          panel: en mitad de una llamada el panel no se está mirando, así que la
          acción se quedaba parada y parecía que la herramienta no iba (H-51). */}
      {!!pendientes.length && (
        <div className="pendientes">
          {pendientes.map(p => (
            <div key={p.id} className="pendiente">
              <div className="pendiente-texto">{p.pregunta}</div>
              <div className="pendiente-acciones">
                <button className="aprobar" onClick={() => responderPendiente(p.id, 'aprobar')}>
                  Aprobar
                </button>
                <button onClick={() => responderPendiente(p.id, 'rechazar')}>Rechazar</button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Transcript */}
      <div className="transcript-overlay" ref={transcriptRef}>
        {transcripts.slice(-MENSAJES_VISIBLES).map(msg => (
          <div key={msg.id} className={`transcript-line ${msg.type}`}>
            {/* Quién habla y a qué hora. Solo se ve en «mando», donde la
                transcripción deja de ser un susurro y pasa a ser bitácora; en
                los otros dos aspectos el CSS lo esconde. */}
            <span className="transcript-cab">
              <b>{msg.type === 'ai' ? 'Perseo' : msg.type === 'user' ? 'Señor Persus' : 'Sistema'}</b>
              <span>{msg.hora}</span>
            </span>
            {msg.text}
          </div>
        ))}
      </div>

      {/* La tira de proyectos. Se despliega sobre la barra de controles y no
          toca la llamada: abrir un proyecto no corta la sesión de voz. */}
      <Proyectos abierto={showProyectos} onCerrar={() => setShowProyectos(false)} />

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
        <Settings
          onClose={() => setShowSettings(false)}
          llamadaActiva={isActive}
          onAspecto={setAspecto}
        />
      )}
    </div>
  );
}

export default App;
