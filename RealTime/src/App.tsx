import { useEffect, useState, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { PerseoFace } from './components/PerseoFace';
import { Settings } from './components/Settings';
import { Panel } from './components/Panel';
import { Habitos } from './components/Habitos';
import { Proyectos } from './components/Proyectos';
import { Escenografia, comoReloj, type Fase } from './components/Escenografia';
import { Marco } from './components/Marco';
import {
  IconCamera, IconCameraOff, IconMic, IconMicOff, IconPhone, IconPhoneOff,
} from './components/Iconos';
import { geminiClient } from './lib/gemini-live';
import { sonar, callar } from './lib/timbre';
import { audioManager } from './lib/audio-manager';
import { audioPlayer } from './lib/audio-player';
import { cameraManager } from './lib/camera-manager';
import { screenManager } from './lib/screen-manager';
import { vigilante, type CaraDetectada } from './lib/identidad';
import {
  defaultConfig,
  cargarAjustesPersistidos,
  type AspectoLive,
  type EstiloHabitos,
} from './lib/config';

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

/** Un aviso que nadie atendió: qué dijo y cuándo se apuntó. */
type LlamadaPendiente = { id: number; texto: string; cuando: string };

const CLAVE_PENDIENTES_LLAMADAS = 'perseo-llamadas-pendientes';
/** Cuánto suena la campana antes de rendirse y apuntar la llamada. */
const PLAZO_AVISO_MS = 45_000;

function App() {
  const [connectionState, setConnectionState] = useState<'disconnected' | 'connecting' | 'connected' | 'error'>('disconnected');
  // El oyente de autollamada se registra una vez y vive todo el rato, así que
  // no puede leer `connectionState` del cierre: se le quedaría el valor de
  // cuando se montó. La referencia sí está siempre al día.
  const connectionStateRef = useRef(connectionState);
  connectionStateRef.current = connectionState;
  // Instante del arranque de la interfaz. Los primeros segundos son «apertura»:
  // si el vigilante de Rust gana la carrera al marcador que el detector dejó
  // justo antes de lanzar la app, el evento llegaría aquí con motivo vacío y
  // sería una llamada al abrirse disfrazada de evento — justo lo que ya no se
  // quiere. Dentro de esa ventana un marcador vacío no entra en llamada; uno
  // con motivo (subagente) suena igual siempre.
  const instanteArranque = useRef(Date.now());
  const [transcripts, setTranscripts] = useState<TranscriptMsg[]>([]);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
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
  const [showHabitos, setShowHabitos] = useState(false);
  // El aire de la pantalla de hábitos vive aquí y no dentro de ella para que
  // Ajustes pueda cambiarlo en caliente, igual que hace con el aspecto del
  // modo live. Ver components/Settings.tsx.
  const [estiloHabitos, setEstiloHabitos] = useState<EstiloHabitos>(defaultConfig.estiloHabitos);
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
  /** Aviso de subagente terminado, esperando que se acepte o se deje para luego. */
  const [avisoEntrante, setAvisoEntrante] = useState<string | null>(null);
  // Reconocimiento de personas (biometría local): quién habla ahora y qué
  // caras hay en el último fotograma analizado. Ambas cosas las decide el
  // núcleo; aquí solo se pintan. Ver lib/identidad.ts.
  const [hablante, setHablante] = useState<string | null>(null);
  const [caras, setCaras] = useState<CaraDetectada[]>([]);
  const transcriptRef = useRef<HTMLDivElement>(null);
  const cameraVideoRef = useRef<HTMLVideoElement>(null);

  // Anti-martilleo del aviso de identidad al modelo: sin esto, cada pausa de
  // 4,5 s del watchdog del vigilante limpiaría el hablante y al volver a hablar
  // reinyectaría «ahora habla Persus» una y otra vez. Mismo texto dentro de un
  // minuto no se repite; texto distinto (voz/cara) pasa siempre la primera vez.
  // Y solo con sesión viva: si llega un resultado cuando el socket ya cayó, se
  // descarta — encolarlo era que saliera horas después en otra llamada.
  const conectadoRef = useRef(false);
  const ultimaIdentidad = useRef<{ texto: string; cuando: number }>({ texto: '', cuando: 0 });
  const carasVistas = useRef('');
  /** Las etiquetas provisionales van marcadas como tales: sin esto el modelo
   *  recibía «Desconocido 1» y podía saludar así a la persona. */
  const etiquetaParaElModelo = (nombre: string): string =>
    nombre.startsWith('Desconocido')
      ? `${nombre} (etiqueta provisional, aún no sabemos su nombre)`
      : nombre;
  const avisarIdentidad = (texto: string) => {
    if (!conectadoRef.current) return;
    const ahora = Date.now();
    if (
      ultimaIdentidad.current.texto === texto &&
      ahora - ultimaIdentidad.current.cuando < 60_000
    ) {
      return;
    }
    ultimaIdentidad.current = { texto, cuando: ahora };
    geminiClient.informarIdentidad(texto);
  };

  // Espejo del estado para poder leerlo desde callbacks sin recrearlos.
  // Antes había dos almacenes en paralelo (un ref y un estado) y se
  // desincronizaban: el ref se vaciaba en 'disconnected', que es justo el evento
  // que emite handleReconnect antes de reconectar, así que el historial que se
  // inyectaba al reconectar siempre estaba vacío. Ver H-06.
  const conversacionRef = useRef<TranscriptMsg[]>([]);
  // Dónde empieza el episodio en curso: la transcripción anterior a esa marca
  // no viaja al prompt ni en reconexión. Lo pide el arreglo del 2026-08-24 —
  // Perseo arrastraba avisos de llamadas previas («el s5 terminó») a cada
  // llamada nueva.
  const inicioEpisodio = useRef(0);
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
        // Reconocimiento de personas: solo si está encendido en Ajustes. Se
        // rearma en cada reconexión porque 'disconnected' lo apaga siempre.
        if (defaultConfig.identidadActivada) vigilante.activar();
        // La vista de la pantalla ya no se pide: si el ajuste no dice lo
        // contrario, Perseo la ve desde el primer segundo de la llamada. Es lo
        // que hace de esto un agente — mirar sin que le den las cosas.
        if (defaultConfig.pantallaAuto) {
          screenManager.start();
        } else if (defaultConfig.screenEnabled) {
          screenManager.start();
        }
        // Confianza automática en llamada (N-3): si hay enlace de voz hay una
        // persona delante, y lo irreversible deja de pedir un sí que ya está
        // oyendo. El techo de 60 min es red de seguridad por si la app muere
        // sin pasar por el colgado; cada reconexión lo rearma.
        invoke('panel_confianza', { minutos: 60 }).catch(e =>
          console.warn('[Confianza] No se pudo activar:', e)
        );
        addTranscript('system', 'Conectado.');
        conectadoRef.current = true;
      } else if (state === 'disconnected' || state === 'error') {
        audioManager.stop();
        cameraManager.stop();
        screenManager.stop();
        vigilante.desactivar();
        conectadoRef.current = false;
        setHablante(null);
        setCaras([]);
        // Nueva llamada, presentación nueva: sin esto, si la misma cara sigue
        // delante al reconectar nadie se lo diría otra vez al modelo.
        carasVistas.current = '';
        ultimaIdentidad.current = { texto: '', cuando: 0 };
        setCameraStream(null);
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
    // Resuelta por voz: la tarjeta sale de la pantalla. Los botones siguen para
    // cuando la confirmación no llega hablada —sin micro, o sin llamada—.
    geminiClient.onAprobacionResuelta = (id) => {
      setPendientes(prev => prev.filter(p => p.id !== id));
    };
    // La herramienta `ver_pantalla`: enciende o apaga la captura aquí, que es
    // donde vive. La frase que devuelve es la que Perseo cuenta por voz.
    geminiClient.onVerPantalla = (activar) => {
      if (activar) { screenManager.start(); return 'Empiezo a ver su pantalla.'; }
      screenManager.stop();
      return 'Dejo de mirar la pantalla.';
    };
    audioManager.onStreamReady = () => addTranscript('system', 'Micrófono activo.');
    audioManager.onError = (err) => addTranscript('system', err);
    cameraManager.onStreamReady = (stream) => setCameraStream(stream);
    // Biometría: la app transporta, el núcleo reconoce, aquí se pinta. La
    // etiqueta de voz entra también en la bitácora: es el registro de quién
    // dijo qué cuando lo lea alguien dentro de un rato.
    audioManager.onTrozoPCM = (trozo) => vigilante.consumirAudio(trozo);
    cameraManager.onFotograma = (base64) => vigilante.consumirFotograma(base64);
    vigilante.onHablante = (nombre) => {
      setHablante(nombre);
      if (nombre) {
        addTranscript('system', `Habla ${nombre}.`);
        avisarIdentidad(`[IDENTIDAD] Ahora habla ${etiquetaParaElModelo(nombre)}.`);
      }
    };
    vigilante.onCaras = (lista) => {
      setCaras(lista);
      // Solo cuando cambia QUIÉN está delante, no cada fotograma: el modelo
      // no necesita el mismo aviso cada cuatro segundos. Las caras sin nombre
      // se descartan del aviso: sin el filtro saldría un literal «null».
      const nombres = lista
        .map(c => c.nombre)
        .filter((n): n is string => !!n)
        .map(etiquetaParaElModelo)
        .sort()
        .join(', ');
      if (nombres) {
        if (nombres !== carasVistas.current) {
          carasVistas.current = nombres;
          avisarIdentidad(`[IDENTIDAD] Delante de la cámara: ${nombres}.`);
        }
      } else {
        carasVistas.current = '';
      }
    };
  
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
    // Solo el EPISODIO actual: una llamada nueva no hereda la transcripción
    // de las anteriores, o Perseo llegaría creyendo que sigue en la de antes
    // — el mismo porqué de matar el testigo al colgar.
    geminiClient.getConversationHistory = () => {
      if (!defaultConfig.saveHistoryEnabled) return "";
      const texto = conversacionRef.current
        .slice(inicioEpisodio.current)
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
      vigilante.desactivar();
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

  /**
   * Qué hacer con un marcador de autollamada.
   *
   * Vacío —palabra clave o aplauso—: entrar en llamada sin más, que es para lo
   * que el detector existe. Con motivo —un subagente terminó—: TIMBRE y
   * decisión del señor Persus; si prefiere no atender, queda pendiente y se
   * cuenta la próxima vez que hable con Perseo.
   */
  const atenderMarcador = (motivo: string, lanzarLlamada: () => void) => {
    if (!motivo) {
      setTimeout(lanzarLlamada, 1500);
      return;
    }
    setAvisoEntrante(prev => (prev ? `${prev}\n${motivo}` : motivo));
  };

  // Señal pendiente cuando la app se abre. La señal se consume desde Rust (un
  // fichero marcador que se borra al leerlo), no con un JSON importado
  // estáticamente: Vite congelaba ese valor al compilar, así que en producción
  // el disparo por aplausos no funcionaba, y además nunca volvía a false.
  // Ver H-09.
  //
  // Al abrirse ya NO hay llamada sola (encargo del señor Persus, 2026-08-24):
  // el marcador vacío de la palabra clave o del aplauso se consume y basta —
  // la ventana ya está al frente y la llamada empieza cuando él pulse. Solo
  // un motivo escrito (un subagente terminó) suena el timbre, que no es una
  // entrada automática: la decisión sigue siendo suya.
  useEffect(() => {
    if (!apiKeyReady || connectionState !== 'disconnected') return;

    let cancelado = false;
    invoke<string>('consumir_autollamada').then(motivo => {
      if (cancelado || !motivo) return;
      atenderMarcador(motivo, handleCall);
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
    const dejarDeEscuchar = listen<string>('perseo://autollamada', evento => {
      if (cancelado) return;
      // Si ya está en llamada no se hace nada: la palabra clave sirve para
      // empezar una conversación, no para cortar la que hay.
      if (connectionStateRef.current !== 'disconnected') return;
      // Recién abierto tampoco: ver `instanteArranque`.
      if (!evento.payload && Date.now() - instanteArranque.current < 4000) return;
      atenderMarcador(evento.payload || '', () => { if (!cancelado) handleCall(); });
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
    setTranscripts(prev => {
      // Antiduplicación: los reintentos y los avisos de sistema repiten la
      // misma frase varias veces seguidas, y una bitácora que se hace eco a sí
      // misma es ruido, no información.
      const ultimo = prev[prev.length - 1];
      if (ultimo && ultimo.type === type && ultimo.text === text) return prev;
      return [...prev, { id: `${Date.now()}-${Math.random()}`, text, type, hora: ahoraCorta() }].slice(-MAX_MENSAJES);
    });
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

  // Llamadas pendientes: avisos que nadie atendió a tiempo o que se dejaron
  // para después. Viven en localStorage para sobrevivir a reinicios de la app,
  // y desde la lista se pueden atender (llamada con su motivo) o quitar.
  const [llamadasPendientes, setLlamadasPendientes] = useState<LlamadaPendiente[]>(() => {
    try {
      const guardadas = JSON.parse(localStorage.getItem(CLAVE_PENDIENTES_LLAMADAS) || '[]');
      return Array.isArray(guardadas) ? guardadas : [];
    } catch {
      return [];
    }
  });
  useEffect(() => {
    localStorage.setItem(CLAVE_PENDIENTES_LLAMADAS, JSON.stringify(llamadasPendientes));
  }, [llamadasPendientes]);

  const apuntarPendiente = (texto: string) => {
    setLlamadasPendientes(prev => [
      ...prev,
      { id: Date.now(), texto, cuando: ahoraCorta() },
    ]);
  };

  // El timbre suena mientras haya un aviso de subagente sin resolver. Sin
  // respuesta en PLAZO_AVISO_MS el aviso NO se pierde: cae a llamadas
  // pendientes y la campana calla — nada de timbre eterno.
  useEffect(() => {
    if (!avisoEntrante) return;
    sonar();
    const plazo = window.setTimeout(() => {
      apuntarPendiente(avisoEntrante);
      addTranscript('system', 'Sin respuesta: la llamada queda en pendientes.');
      setAvisoEntrante(null);
    }, PLAZO_AVISO_MS);
    return () => {
      callar();
      window.clearTimeout(plazo);
    };
  }, [avisoEntrante]);

  const atenderAviso = () => {
    if (!avisoEntrante) return;
    geminiClient.contextoPendiente = avisoEntrante;
    setAvisoEntrante(null);
    handleCall();
  };

  const dejarAvisoParaDespues = () => {
    if (!avisoEntrante) return;
    apuntarPendiente(avisoEntrante);
    addTranscript('system', 'Queda en llamadas pendientes.');
    setAvisoEntrante(null);
  };

  /** Atiende una pendiente de la lista: llamada nueva con su motivo. */
  const atenderPendiente = (id: number) => {
    const pendiente = llamadasPendientes.find(p => p.id === id);
    if (!pendiente || connectionStateRef.current !== 'disconnected') return;
    setLlamadasPendientes(lista => lista.filter(p => p.id !== id));
    geminiClient.contextoPendiente = pendiente.texto;
    handleCall();
  };

  const quitarPendiente = (id: number) => {
    setLlamadasPendientes(lista => lista.filter(p => p.id !== id));
  };

  const handleCall = () => {
    if (!defaultConfig.geminiApiKey) { addTranscript('system', 'API Key no configurada. Pulsa ⚙.'); return; }
    // Quién llama, para que el modelo no lo invente: con motivo (un subagente
    // terminó) es saliente y el motivo ya lo dice; sin él, entrante — ha
    // llamado él. Ver lib/aviso-llamada.ts.
    geminiClient.iniciarLlamada();
    // Empieza un EPISODIO: lo que se hable aquí es lo único que se reinyecta
    // si la red corta a mitad — la transcripción de llamadas anteriores no.
    inicioEpisodio.current = conversacionRef.current.length;
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
    // Y se apaga la confianza que encendió la llamada (N-3): sin persona
    // delante, lo irreversible vuelve a preguntar.
    try {
      await invoke('panel_confianza', {});
    } catch (e) {
      console.warn('[Confianza] No se pudo apagar:', e);
    }

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


  const isActive = connectionState === 'connected' || connectionState === 'connecting';
  const isConnected = connectionState === 'connected';

  // La fase manda el dibujo. El texto largo de un error vive solo en la
  // transcripción (lo pone `onError`); bajo la cara y en la cinta manda una
  // etiqueta corta, porque una frase que crece rompe la composición.
  const fase: Fase = !isActive ? 'reposo'
    : connectionState === 'connecting' ? 'conectando'
    : isSpeaking ? 'hablando'
    : 'escuchando';

  // La etiqueta corta es lo que se pinta bajo el nombre y en la cinta de
  // «mando»: cuatro palabras como mucho. El detalle de un error —que puede
  // ser una frase larga— vive solo en la transcripción, donde hay sitio para
  // leerlo; en la pantalla manda la geometría, y una línea que crece y rompe
  // la composición al cambiar de texto es justo lo que no.
  const etiquetaEstado = connectionState === 'disconnected' ? ''
    : connectionState === 'connecting' ? 'Conectando…'
    : connectionState === 'error' ? 'Error de enlace'
    : isMuted ? 'Silenciado'
    : isSpeaking ? 'Hablando'
    : isConnected ? 'Escuchando'
    : '';

  return (
    // El volumen viaja como variable de CSS (`--vol`, la escribe el rAF de
    // arriba) para que lo lean el anillo, las barras y la cara sin props ni
    // repintados.
    <div ref={contenedorRef} className={`app-container aspecto-${aspecto}`}>
      <div className="bg-art" />

      <Escenografia
        aspecto={aspecto}
        fase={fase}
        estadoTexto={etiquetaEstado}
        sesion={sesion}
      />

      {/* El panel tapa la llamada, no la corta: la sesión de voz sigue viva
          detrás, así que volver es instantáneo y no se pierde la conversación. */}
      {showPanel && <Panel onCerrar={() => setShowPanel(false)} />}

      {/* Los hábitos, con la misma regla que el panel: tapan la llamada, no la
          cortan. Ver components/Habitos.tsx. */}
      {showHabitos && <Habitos onCerrar={() => setShowHabitos(false)} estilo={estiloHabitos} />}

      {/* La llamada entrante de un subagente: timbre y decisión del señor
          Persus. Nada de entrar solos — él acepta o lo deja para después, y
          si no contesta a tiempo cae sola a llamadas pendientes. */}
      {avisoEntrante && (
        <div className="aviso-capa">
          <div className="aviso-caja">
            <div className="aviso-cabecera">
              <span className="aviso-punto" />
              <span className="aviso-etiqueta">PERSEO LLAMA</span>
            </div>
            <p className="aviso-texto">{avisoEntrante}</p>
            <div className="aviso-botones">
              <button className="aviso-btn atender" onClick={atenderAviso}>ATENDER</button>
              <button className="aviso-btn despues" onClick={dejarAvisoParaDespues}>DESPUÉS</button>
            </div>
          </div>
        </div>
      )}

      {/* Llamadas pendientes: avisos rechazados o sin respuesta. Cada uno se
          atiende cuando él quiera (llamada con su motivo) o se tira. */}
      {!avisoEntrante && llamadasPendientes.length > 0 && (
        <div className="pendientes-caja">
          <div className="pendientes-titulo">
            Llamadas pendientes ({llamadasPendientes.length})
          </div>
          <ul className="pendientes-lista">
            {llamadasPendientes.map(p => (
              <li key={p.id} className="pendientes-fila">
                <span className="pendientes-hora">{p.cuando}</span>
                <span className="pendientes-texto">{p.texto}</span>
                <span className="pendientes-acciones">
                  <button onClick={() => atenderPendiente(p.id)} disabled={connectionState !== 'disconnected'}>
                    Atender
                  </button>
                  <button onClick={() => quitarPendiente(p.id)}>Quitar</button>
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* El riel de arriba —panel, ajustes y los botones de la ventana— y la
          pestaña de proyectos. Ver components/Marco.tsx. */}
      <Marco
        onPanel={() => setShowPanel(true)}
        onHabitos={() => setShowHabitos(true)}
        onAjustes={() => setShowSettings(true)}
        // La pestaña vuelve a estar viva (encargo del señor Persus, 2026-08-24):
        // pulsar una ficha arranca los servidores del proyecto — modo
        // `servicio` en perseo_core/proyectos.py — y la pestaña del navegador
        // se abre sola cuando el puerto contesta.
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
            {/* Etiquetas de reconocimiento: la caja la da YuNet sobre 640x480,
                que es exactamente lo que captura camera-manager, así que los
                porcentajes caen donde toca sin medir el vídeo real. */}
            {caras.map((cara, i) => (
              <div
                key={`${cara.nombre}-${i}`}
                className={`cara-marco ${cara.aprendiendo ? 'cara-aprendiendo' : ''}`}
                style={{
                  left: `${(cara.caja[0] / 640) * 100}%`,
                  top: `${(cara.caja[1] / 480) * 100}%`,
                  width: `${(cara.caja[2] / 640) * 100}%`,
                  height: `${(cara.caja[3] / 480) * 100}%`,
                }}
              >
                <span>{cara.nombre ?? '…'}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Quién habla ahora mismo, según su voz. El núcleo lo decide; este chip
          se apaga solo a los pocos segundos de silencio. */}
      {hablante && isActive && (
        <div className="hablante-chip">
          <span className="hablante-punto" />
          {hablante}
        </div>
      )}

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
            {etiquetaEstado && <span className="perseo-status">{etiquetaEstado}</span>}
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
      <Proyectos abierto={showProyectos} onCerrar={() => setShowProyectos(false)} aspecto={aspecto} />

      {/* Controls. La vista de pantalla ya no tiene botón: es automática al
          conectar (pantallaAuto) y un interruptor para algo que siempre está
          encendido solo ocupaba sitio. La cámara sigue siendo manual. */}
      <div className="controls-bar">
        {isActive && (
          <>
            <button className={`ctrl-btn ${isMuted ? 'muted' : ''}`} onClick={toggleMute} title={isMuted ? 'Activar micro' : 'Silenciar'}>
              {isMuted ? <IconMicOff /> : <IconMic />}
            </button>
            <button className={`ctrl-btn ${cameraStream ? 'active' : ''}`} onClick={toggleCamera} title="Cámara">
              {cameraStream ? <IconCamera /> : <IconCameraOff />}
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
          onEstiloHabitos={setEstiloHabitos}
        />
      )}
    </div>
  );
}

export default App;
