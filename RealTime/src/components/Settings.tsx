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
import React, { useEffect, useState } from 'react';
import {
  defaultConfig,
  guardarAjuste,
  SYSTEM_PROMPT_POR_DEFECTO,
  type AspectoLive,
  type EstiloHabitos,
  type ModoMicro,
} from '../lib/config';
import {
  borrarPerfil,
  capturarCara,
  crearPerfil,
  estadoBiometria,
  grabarMuestra,
  renombrarPerfil,
  type EstadoBiometria,
} from '../lib/identidad';
import { esElSenor } from '../lib/quien-hay';

interface Props {
  onClose: () => void;
  /** Si hay una llamada en curso, el cambio de voz no se aplica hasta reconectar. */
  llamadaActiva?: boolean;
  /** Aplica el aspecto en caliente, para poder verlo mientras se elige. */
  onAspecto?: (aspecto: AspectoLive) => void;
  /** Ídem para el aire de la pantalla de hábitos. Misma regla que el aspecto:
   *  si hay que guardar para saber cómo queda, no es elegir, es apostar. */
  onEstiloHabitos?: (estilo: EstiloHabitos) => void;
  /** Cambia la barra de controles al elegir el modo de micrófono. Con una
   *  llamada en curso la ventana lo ignora: la sesión viva sigue como nació. */
  onModoMicro?: (modo: ModoMicro) => void;
}

/** Los tres aspectos del modo live. Ver components/Escenografia.tsx. */
const ASPECTOS: { valor: AspectoLive; nombre: string; que: string }[] = [
  { valor: 'mira', nombre: 'Mira', que: 'Perseo en el centro, anillos girando y lecturas en las esquinas.' },
  { valor: 'mando', nombre: 'Puesto de mando', que: 'Tres columnas: instrumentos, Perseo y la bitácora de la llamada.' },
  { valor: 'cartel', nombre: 'Cartel', que: 'Descentrada, con el estado en letras grandes y el kanji al fondo.' },
];

/** Los dos aires de la pantalla de hábitos. Ver components/Habitos.tsx. */
const ESTILOS_HABITOS: { valor: EstiloHabitos; nombre: string; que: string }[] = [
  { valor: 'perseo', nombre: 'Perseo', que: 'Negro, versalitas y monoespaciada, como el panel y la llamada.' },
  { valor: 'plantilla', nombre: 'Plantilla', que: 'El beige y el taupe de la hoja de la que salió la pantalla.' },
];

/** Los dos modos de micrófono. Ver ModoMicro en lib/config.ts. */
const MODOS_MICRO: { valor: ModoMicro; nombre: string; que: string }[] = [
  {
    valor: 'manos-libres',
    nombre: 'Manos libres',
    que: 'Perseo escucha siempre y contesta cuando detecta que has terminado.',
  },
  {
    valor: 'pulsar',
    nombre: 'Pulsar para hablar',
    que: 'El micrófono va cerrado; se abre mientras mantienes el botón o la barra espaciadora.',
  },
];

const VOCES: { valor: string; nombre: string; como: string }[] = [
  { valor: 'Charon', nombre: 'Charon', como: 'Muy grave' },
  { valor: 'Orus', nombre: 'Orus', como: 'Grave' },
  { valor: 'Fenrir', nombre: 'Fenrir', como: 'Media' },
  { valor: 'Puck', nombre: 'Puck', como: 'Clara' },
];

export const Settings: React.FC<Props> = ({
  onClose, llamadaActiva = false, onAspecto, onEstiloHabitos, onModoMicro,
}) => {
  const [voice, setVoice] = useState(defaultConfig.voiceName);
  const [prompt, setPrompt] = useState(defaultConfig.systemPrompt);
  const [guardarHistorial, setGuardarHistorial] = useState(defaultConfig.saveHistoryEnabled);
  const [pantallaAuto, setPantallaAuto] = useState(defaultConfig.pantallaAuto);
  const [identidad, setIdentidad] = useState(defaultConfig.identidadActivada);
  // Cuál de los perfiles es él. Sin esto, Perseo trata de «señor Persus» a
  // cualquiera que reconozca, incluida una visita. Ver lib/quien-hay.ts.
  const [perfilPersus, setPerfilPersus] = useState(defaultConfig.perfilPersus);
  const [aspecto, setAspecto] = useState<AspectoLive>(defaultConfig.aspectoLive);
  const [estiloHabitos, setEstiloHabitos] = useState<EstiloHabitos>(defaultConfig.estiloHabitos);
  const [modoMicro, setModoMicro] = useState<ModoMicro>(defaultConfig.modoMicro);
  const [guardando, setGuardando] = useState(false);
  const [error, setError] = useState('');

  // Biometría: estado de perfiles y motores, traído del núcleo al abrir.
  const [bio, setBio] = useState<EstadoBiometria | null>(null);
  const [grabando, setGrabando] = useState(false);
  // La muestra grabada espera su nombre; el renombrado en curso espera texto;
  // el borrado pide un segundo clic. Todo EN LÍNEA y no con window.prompt:
  // los diálogos nativos no están garantizados dentro del WebView de Tauri.
  const [vozGrabada, setVozGrabada] = useState<string | null>(null);
  const [nombreVoz, setNombreVoz] = useState('');
  // El alta tarda (el motor carga la primera vez): sin este estado, cada clic
  // impaciente reenviaba la misma muestra y el núcleo la reforzaba otra vez.
  const [guardandoVoz, setGuardandoVoz] = useState(false);
  // Cara: mismo patrón que la voz — fotograma capturado espera su nombre.
  const [caraGrabada, setCaraGrabada] = useState<string | null>(null);
  const [nombreCara, setNombreCara] = useState('');
  const [capturandoCara, setCapturandoCara] = useState(false);
  const [guardandoCara, setGuardandoCara] = useState(false);
  const [renombrarDe, setRenombrarDe] = useState<string | null>(null);
  const [nombreNuevo, setNombreNuevo] = useState('');
  const [borrarConfirmando, setBorrarConfirmando] = useState<string | null>(null);

  useEffect(() => {
    let vivo = true;
    estadoBiometria()
      .then((estado) => { if (vivo) setBio(estado); })
      .catch((e) => console.warn('[Ajustes] Biometría sin respuesta:', e));
    return () => { vivo = false; };
  }, []);

  const refrescarBiometria = () => {
    estadoBiometria().then(setBio).catch(() => {});
  };

  /** Enseñar la voz: graba seis segundos y deja la muestra esperando nombre. */
  const ensenarVoz = async () => {
    setError('');
    setGrabando(true);
    try {
      const audio = await grabarMuestra(6);
      setVozGrabada(audio);
      setNombreVoz('');
    } catch (e) {
      setError(`No se pudo grabar: ${e}`);
    } finally {
      setGrabando(false);
    }
  };

  const guardarVoz = async () => {
    if (!vozGrabada || !nombreVoz.trim() || guardandoVoz) return;
    setGuardandoVoz(true);
    try {
      const resultado = await crearPerfil(nombreVoz.trim(), vozGrabada);
      if (resultado.error) { setError(resultado.error); return; }
      setVozGrabada(null);
      setNombreVoz('');
      refrescarBiometria();
    } finally {
      setGuardandoVoz(false);
    }
  };

  /** Enseñar la cara: abre la cámara, dispara un fotograma y espera nombre. */
  const ensenarCara = async () => {
    setError('');
    setCapturandoCara(true);
    try {
      const imagen = await capturarCara();
      setCaraGrabada(imagen);
      setNombreCara('');
    } catch (e) {
      setError(`No se pudo abrir la cámara: ${e}`);
    } finally {
      setCapturandoCara(false);
    }
  };

  const guardarCara = async () => {
    if (!caraGrabada || !nombreCara.trim() || guardandoCara) return;
    setGuardandoCara(true);
    try {
      const resultado = await crearPerfil(nombreCara.trim(), undefined, caraGrabada);
      if (resultado.error) { setError(resultado.error); return; }
      setCaraGrabada(null);
      setNombreCara('');
      refrescarBiometria();
    } finally {
      setGuardandoCara(false);
    }
  };

  const confirmarRenombrar = async () => {
    if (!renombrarDe || !nombreNuevo.trim()) return;
    try {
      const resultado = await renombrarPerfil(renombrarDe, nombreNuevo.trim());
      if (resultado.error) { setError(resultado.error); return; }
      setRenombrarDe(null);
      setNombreNuevo('');
      refrescarBiometria();
    } catch (e) {
      setError(`No se pudo renombrar a «${renombrarDe}»: ${e}`);
    }
  };

  // El núcleo contesta 400 o 404 con un `Err` de Rust, y eso aquí llega como
  // excepción, no como `{ error }`. Sin el try el botón parecía muerto: se
  // pulsaba «¿Seguro?» y no pasaba nada ni se decía por qué.
  const borrarUno = async (nombre: string) => {
    try {
      const resultado = await borrarPerfil(nombre);
      if (resultado.error) { setError(resultado.error); return; }
      setBorrarConfirmando(null);
      refrescarBiometria();
    } catch (e) {
      setError(`No se pudo borrar a «${nombre}»: ${e}`);
    }
  };

  // El que había al abrir, para devolverlo si se cancela: el aspecto se aplica
  // al pulsar la ficha, así que sin esto «Cancelar» dejaría el cambio hecho.
  const [aspectoOriginal] = useState<AspectoLive>(defaultConfig.aspectoLive);
  const [estiloHabitosOriginal] = useState<EstiloHabitos>(defaultConfig.estiloHabitos);
  const [modoMicroOriginal] = useState<ModoMicro>(defaultConfig.modoMicro);

  const voiceCambiada = voice !== defaultConfig.voiceName;
  const hayClave = !!defaultConfig.geminiApiKey;

  const elegirAspecto = (valor: AspectoLive) => {
    setAspecto(valor);
    onAspecto?.(valor);
  };

  const elegirEstiloHabitos = (valor: EstiloHabitos) => {
    setEstiloHabitos(valor);
    onEstiloHabitos?.(valor);
  };

  const elegirModoMicro = (valor: ModoMicro) => {
    setModoMicro(valor);
    onModoMicro?.(valor);
  };

  const cerrarSinGuardar = () => {
    if (aspecto !== aspectoOriginal) onAspecto?.(aspectoOriginal);
    if (estiloHabitos !== estiloHabitosOriginal) onEstiloHabitos?.(estiloHabitosOriginal);
    if (modoMicro !== modoMicroOriginal) onModoMicro?.(modoMicroOriginal);
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
      await guardarAjuste('identidadActivada', identidad);
      await guardarAjuste('perfilPersus', perfilPersus);
      await guardarAjuste('aspectoLive', aspecto);
      await guardarAjuste('estiloHabitos', estiloHabitos);
      await guardarAjuste('modoMicro', modoMicro);
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
            <div className="ajustes-titulo">Aire de la pantalla de hábitos</div>
            <div className="ajustes-fichas">
              {ESTILOS_HABITOS.map(e => (
                <button
                  key={e.valor}
                  className={`ajustes-ficha ${estiloHabitos === e.valor ? 'elegida' : ''}`}
                  onClick={() => elegirEstiloHabitos(e.valor)}
                >
                  <span className="ajustes-ficha-nombre">{e.nombre}</span>
                  <span className="ajustes-ficha-que">{e.que}</span>
                </button>
              ))}
            </div>
            <p className="ajustes-nota">
              Solo cambian los colores y la tipografía: el reparto, las cifras y lo que
              se puede tocar son los mismos en los dos.
            </p>
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
            <div className="ajustes-titulo">Micrófono</div>
            <div className="ajustes-fichas">
              {MODOS_MICRO.map(m => (
                <button
                  key={m.valor}
                  className={`ajustes-ficha ${modoMicro === m.valor ? 'elegida' : ''}`}
                  onClick={() => elegirModoMicro(m.valor)}
                >
                  <span className="ajustes-ficha-nombre">{m.nombre}</span>
                  <span className="ajustes-ficha-que">{m.que}</span>
                </button>
              ))}
            </div>
            <p className="ajustes-nota">
              Con ruido alrededor —gente hablando, la tele, una cafetería— manos libres
              toma cualquier voz por una orden y Perseo contesta a quien no le ha
              hablado. Pulsando solo entra lo que le dices a propósito.
            </p>
            {llamadaActiva && modoMicro !== modoMicroOriginal && (
              <p className="ajustes-nota">El modo del micrófono se aplicará al volver a llamar.</p>
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
            <label className="ajustes-interruptor">
              <input
                type="checkbox"
                checked={identidad}
                onChange={e => setIdentidad(e.target.checked)}
              />
              <span>Reconocer quién habla y quién sale por la cámara</span>
            </label>
            <p className="ajustes-nota">
              Perseo pone nombre a cada voz conocida y etiqueta cada cara; si no
              conoce a alguien, aprende su voz con la llamada. Todo se decide y
              se guarda en este ordenador — nunca sale nada a internet.
            </p>
            <p className="ajustes-nota">
              Marca con «Este soy yo» tu propio perfil: solo a ese le trata Perseo
              de señor Persus, y delante de cualquier otra persona se calla lo tuyo.
            </p>

            {identidad && (
              <div className="ajustes-biometria">
                {!grabando && !vozGrabada && (
                  <button className="ajustes-accion" onClick={ensenarVoz}>
                    Enseñar mi voz (6 segundos)
                  </button>
                )}
                {grabando && <p className="ajustes-nota">Grabando… habla ahora.</p>}
                {vozGrabada && (
                  <div className="ajustes-fila">
                    <input
                      autoFocus
                      placeholder="¿A qué nombre guardo esta voz?"
                      value={nombreVoz}
                      onChange={e => setNombreVoz(e.target.value)}
                    />
                    <button onClick={guardarVoz} disabled={!nombreVoz.trim() || guardandoVoz}>
                      {guardandoVoz ? 'Guardando…' : 'Guardar'}
                    </button>
                    <button onClick={() => setVozGrabada(null)}>Descartar</button>
                  </div>
                )}
                {!capturandoCara && !caraGrabada && (
                  <button className="ajustes-accion" onClick={ensenarCara}>
                    Enseñar mi cara (un fotograma)
                  </button>
                )}
                {capturandoCara && (
                  <p className="ajustes-nota">Abriendo la cámara… mira al objetivo y no te muevas.</p>
                )}
                {caraGrabada && (
                  <div className="ajustes-fila">
                    <input
                      autoFocus
                      placeholder="¿A qué nombre guardo esta cara?"
                      value={nombreCara}
                      onChange={e => setNombreCara(e.target.value)}
                    />
                    <button onClick={guardarCara} disabled={!nombreCara.trim() || guardandoCara}>
                      {guardandoCara ? 'Guardando…' : 'Guardar'}
                    </button>
                    <button onClick={() => setCaraGrabada(null)}>Descartar</button>
                  </div>
                )}
                {!bio && <p className="ajustes-nota">Preguntando al núcleo…</p>}
                {bio && (
                  <>
                    {!bio.disponibilidad.voz && (
                      <p className="ajustes-nota ajustes-aviso">{bio.disponibilidad.motivo_voz}</p>
                    )}
                    {!bio.disponibilidad.cara && (
                      <p className="ajustes-nota ajustes-aviso">{bio.disponibilidad.motivo_cara}</p>
                    )}
                    {bio.aprendiendo.voz && (
                      <p className="ajustes-nota">
                        Aprendiendo a «{bio.aprendiendo.voz.etiqueta}»:{' '}
                        {Math.round(bio.aprendiendo.voz.peso)} de {bio.aprendiendo.voz.objetivo} s
                        de voz.
                      </p>
                    )}
                    {bio.perfiles.length > 0 ? (
                      <ul className="ajustes-perfiles">
                        {bio.perfiles.map(p => (
                          <li key={p.nombre}>
                            {renombrarDe === p.nombre ? (
                              <>
                                <input
                                  autoFocus
                                  placeholder="Nombre nuevo"
                                  value={nombreNuevo}
                                  onChange={e => setNombreNuevo(e.target.value)}
                                  onKeyDown={e => { if (e.key === 'Enter') void confirmarRenombrar(); }}
                                />
                                <button onClick={confirmarRenombrar} disabled={!nombreNuevo.trim()}>
                                  Guardar
                                </button>
                                <button onClick={() => setRenombrarDe(null)}>Cancelar</button>
                              </>
                            ) : (
                              <>
                                <span className="perfil-nombre">{p.nombre}</span>
                                <span className="perfil-datos">
                                  {[p.voz ? 'voz' : null, p.caras > 0 ? `${p.caras} cara(s)` : null]
                                    .filter(Boolean)
                                    .join(' · ') || 'sin muestras'}
                                </span>
                                {esElSenor(p.nombre, perfilPersus) ? (
                                  <span className="perfil-yo">Eres tú</span>
                                ) : (
                                  <button onClick={() => setPerfilPersus(p.nombre)}>
                                    Este soy yo
                                  </button>
                                )}
                                <button
                                  onClick={() => { setRenombrarDe(p.nombre); setNombreNuevo(''); }}
                                >
                                  Renombrar
                                </button>
                                {borrarConfirmando === p.nombre ? (
                                  <>
                                    <button
                                      className="perfil-borrar-si"
                                      onClick={() => void borrarUno(p.nombre)}
                                    >
                                      ¿Seguro? Se pierde.
                                    </button>
                                    <button onClick={() => setBorrarConfirmando(null)}>No</button>
                                  </>
                                ) : (
                                  <button onClick={() => setBorrarConfirmando(p.nombre)}>
                                    Borrar
                                  </button>
                                )}
                              </>
                            )}
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="ajustes-nota">
                        Todavía no hay perfiles: hable delante del micrófono y
                        Perseo aprenderá solo, o use «Enseñar mi voz».
                      </p>
                    )}
                  </>
                )}
              </div>
            )}
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
