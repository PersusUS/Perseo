/**
 * La ventana de Perseo: quién enciende qué, y en qué orden.
 * Es el orquestador de la cara de voz. No habla con Gemini —eso es
 * `lib/gemini-live.ts`—, no dibuja la escenografía —`components/Escenografia.tsx`—
 * y no decide nada de lo que se hace: reparte estado y escucha eventos.
 * Lo que sostiene desde aquí:
 * - **El estado de la llamada** (conectando, conectado, error), el silencio del
 *   micrófono, la transcripción de los dos lados y quién está hablando.
 * - **Las cuatro pantallas que se abren encima**: ajustes, panel, hábitos y el
 *   riel de proyectos. Cada una es un componente propio; aquí solo vive el
 *   interruptor.
 * - **Lo que llega de fuera sin pedirlo**: los eventos de Tauri (`listen`) para
 *   abrir el panel desde la bandeja, las llamadas entrantes y las
 *   confirmaciones pendientes que el núcleo deja esperando un sí.
 * La clave de la API nunca pasa por aquí: la lee Rust del almacén cifrado y
 * este fichero solo se entera de si está lista (`apiKeyReady`).
 */

import { useEffect, useState, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { PerseoFace } from './components/PerseoFace';
import { Settings } from './components/Settings';
import { Panel } from './components/Panel';
import { Habitos } from './components/Habitos';
import { Tareas } from './components/Tareas';
import {
  EVENTO_CAMBIO as TAREAS_CAMBIO,
  aplicarOrden as aplicarOrdenTarea,
  foto as fotoTareas,
  leer as leerTareas,
  resumen as resumenTareas,
} from './lib/tareas';
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
import { iniciarDiagnostico, pararDiagnostico } from './lib/diagnostico';
import { cameraManager } from './lib/camera-manager';
import { screenManager } from './lib/screen-manager';
import { vigilante, type CaraDetectada } from './lib/identidad';
import { avisoCaras, avisoHablante, esElSenor, sinAvisoDeIdentidad } from './lib/quien-hay';
import {
  defaultConfig,
  cargarAjustesPersistidos,
  type AspectoLive,
  type EstiloHabitos,
  type ModoMicro,
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

/** Cada cuánto se le pregunta al núcleo si hay algo pendiente que hacer con el
 *  tablero de tareas.
 *  El corcho vive en el `localStorage` de esta ventana, así que el chat escrito
 *  y los agentes —que son Python— no pueden clavar una nota: dejan la orden en
 *  el núcleo y esto la recoge. Ocho segundos porque la conversación escrita
 *  suele pasar con la app delante y esperar medio minuto a ver aparecer la nota
 *  que acabas de pedir se siente roto; y es una petición a localhost, no a
 *  internet. Ver `perseo_core/tareas.py`. */
const ESPERA_ORDENES_TAREAS = 8000;

/** Cuánto se espera, sin que nadie toque nada, antes de mandarle al núcleo la
 *  copia del tablero. Arrastrar una nota cambia el estado varias veces seguidas
 *  y cada cambio sería un POST; con la espera, un arrastre entero manda uno. */
const ESPERA_ESPEJO_TAREAS = 2000;
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
  // Manos libres o pulsar para hablar (ModoMicro en lib/config.ts). Vive aquí
  // y no dentro de Ajustes porque manda sobre la barra de controles: en
  // «pulsar» el botón de silenciar deja el sitio al de hablar.
  const [modoMicro, setModoMicro] = useState<ModoMicro>(defaultConfig.modoMicro);
  // El modo con el que se ABRIÓ la sesión en curso. El ajuste puede cambiar a
  // mitad de llamada y esa llamada sigue siendo la que era: el servidor solo
  // acepta señales de turno si se le pidió con la detección apagada, así que
  // sin esto la barra espaciadora hablaría con una sesión que no escucha.
  const modoMicroRef = useRef<ModoMicro>(defaultConfig.modoMicro);
  // Si el botón de hablar está pulsado AHORA MISMO. Se duplica en una
  // referencia porque los atajos de teclado se registran una sola vez y
  // leerían el valor del cierre — el de cuando se montaron.
  const [pulsando, setPulsando] = useState(false);
  const pulsandoRef = useRef(false);
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
  const [showTareas, setShowTareas] = useState(false);
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
  // Lo que Perseo ha pedido hacer y está parado esperando un sí. Vive aquí
  // y no solo en el panel, que en mitad de una llamada nadie mira.
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
  // Los últimos avisos mandados, para reconocerlos si el modelo los lee en voz
  // alta y quitarlos de la transcripción. Tres bastan: un aviso viejo ya no
  // puede estar saliendo por la boca de Perseo.
  const avisosRecientes = useRef<string[]>([]);
  const carasVistas = useRef('');
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
    avisosRecientes.current = [texto, ...avisosRecientes.current].slice(0, 3);
    geminiClient.informarIdentidad(texto);
  };

  // Espejo del estado para poder leerlo desde callbacks sin recrearlos.
  // Antes había dos almacenes en paralelo (un ref y un estado) y se
  // desincronizaban: el ref se vaciaba en 'disconnected', que es justo el evento
  // que emite handleReconnect antes de reconectar, así que el historial que se
  // inyectaba al reconectar siempre estaba vacío.
  const conversacionRef = useRef<TranscriptMsg[]>([]);
  // Dónde empieza el episodio en curso: la transcripción anterior a esa marca
  // no viaja al prompt ni en reconexión. Lo pide el arreglo del 2026-08-24 —
  // Perseo arrastraba avisos de llamadas previas («el s5 terminó») a cada
  // llamada nueva.
  const inicioEpisodio = useRef(0);
  useEffect(() => { conversacionRef.current = transcripts; }, [transcripts]);

  /** Persiste la conversación como Markdown en el vault, donde el RAG la indexa
   * solo. Antes esto era un console.log con la escritura comentada. */
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
        armarMicro();
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
        // Un turno abierto que se queda a medias no se cierra solo: sin esto,
        // el botón de hablar seguiría encendido sobre una llamada muerta.
        pulsandoRef.current = false;
        setPulsando(false);
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
        // reinyección de contexto. Se guarda y se limpia al colgar.
      }
    };

    geminiClient.onTranscript = (rol, delta, final) => {
      añadirFragmento(rol, delta, final);
    };

    geminiClient.onError = (msg) => addTranscript('system', msg);
    // Perseo acaba de preguntarle su nombre a quien tenía delante y lo ha
    // guardado. La pantalla no puede seguir enseñando «Desconocido 2» después
    // de que el propio interesado haya dicho cómo se llama.
    geminiClient.onPersonaNombrada = (etiqueta, nombre) => {
      addTranscript(
        'system',
        esElSenor(nombre, defaultConfig.perfilPersus)
          ? `«${etiqueta}» era usted: perfil guardado como ${nombre}.`
          : `«${etiqueta}» ya tiene nombre: ${nombre}.`,
      );
      setHablante(previo => (previo === etiqueta ? nombre : previo));
      setCaras(previas =>
        previas.map(c => (c.nombre === etiqueta ? { ...c, nombre } : c)),
      );
      // El aviso de caras solo sale cuando cambia quién está delante: sin
      // limpiar la huella, el nombre nuevo no llegaría al modelo hasta que
      // alguien entrara o saliera del encuadre.
      carasVistas.current = '';
    };

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
        // El aviso dice quién habla Y qué trato le toca. Un nombre a secas
        // dejaba al modelo llamando «señor Persus» a cualquiera que pasara por
        // delante de la cámara. Ver lib/quien-hay.ts.
        avisarIdentidad(avisoHablante(nombre, defaultConfig.perfilPersus));
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
        .sort();
      const clave = nombres.join(', ');
      if (clave) {
        if (clave !== carasVistas.current) {
          carasVistas.current = clave;
          const aviso = avisoCaras(nombres, defaultConfig.perfilPersus);
          if (aviso) avisarIdentidad(aviso);
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
    // está activada; antes solo se recuperaba el monólogo de Perseo.
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
  // sistema). Ya no viaja dentro del bundle
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
  // automática y sumando se pondría a cero a media conversación. La
  // marca solo se borra al colgar, que es cuando la llamada acaba de verdad.
  useEffect(() => {
    const t = setInterval(() => {
      const desde = sesionDesde.current;
      setSesion(desde ? Math.floor((Date.now() - desde) / 1000) : 0);
    }, 1000);
    return () => clearInterval(t);
  }, []);

  // Lo que Perseo pidió hacer con el tablero desde fuera de esta ventana.
  // Va en App y no dentro de la pantalla de tareas a propósito: una nota que se
  // pide por el chat escrito tiene que clavarse aunque el corcho esté cerrado.
  // Si está abierto se entera solo, porque `aplicarOrden` avisa (ver
  // EVENTO_CAMBIO en lib/tareas.ts).
  // Un fallo aquí no se enseña: el núcleo apagado es un estado normal de esta
  // app, y las órdenes esperan en su cola hasta la próxima vuelta.
  useEffect(() => {
    const recoger = async () => {
      try {
        const recogido = await invoke<{ ordenes?: unknown[] }>('tareas_recoger');
        for (const orden of recogido?.ordenes ?? []) aplicarOrdenTarea(orden);
      } catch {
        // Núcleo apagado o sin token todavía: se reintenta a la vuelta siguiente.
      }
    };
    recoger();
    const t = setInterval(recoger, ESPERA_ORDENES_TAREAS);
    return () => clearInterval(t);
  }, []);

  // Y la copia del tablero para el núcleo, que sale de aquí y de ningún otro
  // sitio.
  // Estuvo dentro de la pantalla de tareas hasta el 2026-09-03. Dejó de valer
  // en cuanto Perseo pudo escribir: con el corcho cerrado esa pantalla no
  // existe, así que una nota clavada por voz o por el chat se guardaba en el
  // almacén y el núcleo seguía sirviendo el tablero de antes — Perseo no veía
  // la nota que él mismo acababa de poner. Aquí se oye a los tres escritores
  // por igual, porque los tres avisan (ver `avisar()` en lib/tareas.ts).
  // Se manda también al arrancar: si la última vez se cerró la app antes de que
  // saliera la copia, esta es la ocasión de ponerla al día.
  // Si falla, se calla: el núcleo apagado es un estado normal de esta app.
  useEffect(() => {
    let espera: ReturnType<typeof setTimeout>;
    const espejar = () => {
      clearTimeout(espera);
      espera = setTimeout(() => {
        const datos = leerTareas();
        invoke('tareas_espejo', { texto: resumenTareas(datos), foto: fotoTareas(datos) })
          .catch(e => console.debug('[Tareas] El núcleo no recogió la copia:', e));
      }, ESPERA_ESPEJO_TAREAS);
    };
    espejar();
    window.addEventListener(TAREAS_CAMBIO, espejar);
    return () => {
      clearTimeout(espera);
      window.removeEventListener(TAREAS_CAMBIO, espejar);
    };
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

      // Los avisos de identidad son información del sistema; si el modelo los
      // lee en voz alta, al menos no se quedan escritos en la pantalla ni en la
      // bitácora que se guarda en el vault. Se limpia el texto entero y no el
      // trozo: la marca llega partida entre fragmentos. Ver lib/quien-hay.ts.
      const ultimo = prev[prev.length - 1];
      if (ultimo && ultimo.type === rol && ultimo.abierto) {
        const texto = sinAvisoDeIdentidad(ultimo.text + delta, avisosRecientes.current);
        return [...prev.slice(0, -1), { ...ultimo, text: texto }];
      }
      return [
        ...prev,
        {
          id: `${Date.now()}-${Math.random()}`,
          text: sinAvisoDeIdentidad(delta, avisosRecientes.current),
          type: rol,
          abierto: true,
          hora: ahoraCorta(),
        },
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
    // El cuaderno de la llamada, para poder mirar después por qué se oyó como
    // se oyó. Ver lib/diagnostico.ts.
    iniciarDiagnostico(`micrófono ${defaultConfig.modoMicro}, pantalla ${defaultConfig.pantallaAuto ? 'automática' : 'apagada'}`);
    audioPlayer.initialize();
    audioManager.start();
    armarMicro();
    geminiClient.connect();
  };

  /**
   * Deja el micrófono como pide el modo: de par en par en manos libres, mudo
   * hasta que se pulse en «pulsar para hablar». Se llama al llamar y en cada
   * reconexión, porque `audioManager.stop()` devuelve el paso abierto.
   */
  const armarMicro = () => {
    const modo = defaultConfig.modoMicro;
    modoMicroRef.current = modo;
    setModoMicro(modo);
    audioManager.transmitir(modo !== 'pulsar');
    pulsandoRef.current = false;
    setPulsando(false);
  };

  /** Se aprieta el botón de hablar: se avisa al modelo y se abre el paso. */
  const empezarAHablar = () => {
    if (modoMicroRef.current !== 'pulsar') return;
    if (!conectadoRef.current || pulsandoRef.current) return;
    pulsandoRef.current = true;
    setPulsando(true);
    // Y que ningún botón se quede con el foco mientras se habla: pulsar
    // «Llamar» se lo deja al de colgar, y la barra espaciadora lo activaría
    // por su cuenta. Cinturón además del `preventDefault` de las dos teclas.
    const enfocado = document.activeElement as HTMLElement | null;
    if (enfocado && enfocado.tagName === 'BUTTON') enfocado.blur();
    // El aviso primero y el audio después: con la detección automática
    // apagada, un trozo que llegue antes del `activityStart` se tira.
    geminiClient.abrirTurno();
    audioManager.transmitir(true);
  };

  /** Se suelta: se cierra el paso y se le dice al modelo que conteste. */
  const dejarDeHablar = () => {
    if (!pulsandoRef.current) return;
    pulsandoRef.current = false;
    setPulsando(false);
    audioManager.transmitir(false);
    geminiClient.cerrarTurno();
  };

  const handleHangup = async () => {
    pararDiagnostico(`${audioPlayer.diagnostico().vecesSeca} veces seca la cola`);
    dejarDeHablar();
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
    // limpia, no en cada 'disconnected'. Ver y.
    if (defaultConfig.saveHistoryEnabled) {
      await guardarConversacion(conversacionRef.current);
    }
    setTranscripts([]);
  };

  // La barra espaciadora hace lo mismo que el botón, que es como se usa esto
  // de verdad: mirando la pantalla y sin buscar el ratón. Se registra una sola
  // vez y solo lee referencias, así que no se le queda ningún valor viejo.
  useEffect(() => {
    const escribiendo = (destino: EventTarget | null) => {
      const el = destino as HTMLElement | null;
      if (!el || !el.tagName) return false;
      return el.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName);
    };
    const abajo = (e: KeyboardEvent) => {
      if (e.code !== 'Space' || e.repeat || escribiendo(e.target)) return;
      // Sin esto la barra además pulsa el botón que tenga el foco, y el
      // teclado colgaba la llamada mientras se hablaba.
      e.preventDefault();
      empezarAHablar();
    };
    const arriba = (e: KeyboardEvent) => {
      if (e.code !== 'Space') return;
      // Y aquí otra vez, que es donde de verdad importa: un botón enfocado se
      // activa con la barra en el KEYUP, no en el keydown, así que cancelar
      // solo el keydown dejaba el click vivo — se hablaba, se soltaba y el
      // teclado pulsaba «Colgar». Ver también el desenfoque de `empezarAHablar`.
      if (!escribiendo(e.target)) e.preventDefault();
      dejarDeHablar();
    };
    // Cambiar de ventana con la barra apretada dejaba el turno abierto para
    // siempre: la tecla se suelta donde ya no lo oye nadie.
    const fuera = () => dejarDeHablar();
    window.addEventListener('keydown', abajo);
    window.addEventListener('keyup', arriba);
    window.addEventListener('blur', fuera);
    return () => {
      window.removeEventListener('keydown', abajo);
      window.removeEventListener('keyup', arriba);
      window.removeEventListener('blur', fuera);
    };
  }, []);

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
  // «Escuchando» solo cuando el micrófono está abierto de verdad: pulsando
  // para hablar y con el botón suelto, el paso está cerrado y la pantalla se
  // quedaba clavada en «Escuchando» mientras no oía nada (2026-09-09).
  const microAbierto = modoMicro !== 'pulsar' || pulsando;
  const fase: Fase = !isActive ? 'reposo'
    : connectionState === 'connecting' ? 'conectando'
    : isSpeaking ? 'hablando'
    : microAbierto ? 'escuchando'
    : 'espera';

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
    // Pulsando, «Escuchando» sería mentira la mayor parte del tiempo: el
    // micrófono está cerrado hasta que se aprieta, y la pantalla lo dice.
    : isConnected && modoMicro === 'pulsar' ? (pulsando ? 'Le escucho' : 'Pulse para hablar')
    : isConnected ? 'Escuchando'
    : '';

  // La cara: «escuchando» es un estado del micrófono, no de la llamada. Con el
  // botón suelto no late, se queda quieta — que es lo que está haciendo.
  const caraEscuchando = isConnected && !isSpeaking && microAbierto;

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

      {/* El corcho de tareas, con la misma regla: tapa la llamada, no la
          corta. Ver components/Tareas.tsx. */}
      {showTareas && <Tareas onCerrar={() => setShowTareas(false)} />}

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
        onTareas={() => setShowTareas(true)}
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
            isListening={caraEscuchando}
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
          acción se quedaba parada y parecía que la herramienta no iba. */}
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
            {/* En «pulsar para hablar» el botón de silenciar no pinta nada: el
                micrófono ya está cerrado de serie. Su sitio lo ocupa el de
                hablar, que se mantiene apretado —o la barra espaciadora—. */}
            {modoMicro === 'pulsar' ? (
              <button
                className={`ctrl-btn hablar ${pulsando ? 'pulsando' : ''}`}
                onPointerDown={empezarAHablar}
                onPointerUp={dejarDeHablar}
                onPointerLeave={dejarDeHablar}
                onPointerCancel={dejarDeHablar}
                // El botón no es un botón de pulsar y soltar del teclado: la
                // barra ya la escucha la ventana entera, y dejarle su
                // comportamiento de siempre abría dos turnos por pulsación.
                onKeyDown={e => e.preventDefault()}
                title="Mantén pulsado para hablar (o la barra espaciadora)"
              >
                {pulsando ? <IconMic /> : <IconMicOff />}
              </button>
            ) : (
              <button className={`ctrl-btn ${isMuted ? 'muted' : ''}`} onClick={toggleMute} title={isMuted ? 'Activar micro' : 'Silenciar'}>
                {isMuted ? <IconMicOff /> : <IconMic />}
              </button>
            )}
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
          // Fuera de llamada el cambio se ve al momento; dentro no se toca la
          // sesión viva — la abrió el modo anterior y así se queda.
          onModoMicro={(modo) => { if (!isActive) { modoMicroRef.current = modo; setModoMicro(modo); } }}
        />
      )}
    </div>
  );
}

export default App;
