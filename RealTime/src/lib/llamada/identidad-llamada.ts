/**
 * Quién está delante durante una llamada, fuera de la pantalla que la pinta.
 *
 * El segundo de los tres ganchos que salen de `App.tsx`. Recoge lo que estaba
 * repartido entre tres efectos —el que conecta, el que registra los callbacks
 * del vigilante y el que pinta la transcripción— y que, junto, es una sola
 * cosa: **quién habla, quién se ve, y qué se le cuenta al modelo sobre eso**.
 *
 * Tres reglas viven aquí, y las tres tienen su porqué medido:
 *
 *  1. **Un aviso no se repite dentro de un minuto.** Sin esto, con la misma
 *     persona delante, se reinyectaba «ahora habla Persus» una y otra vez.
 *     Texto distinto —voz y cara son dos— pasa siempre la primera vez.
 *  2. **Solo con sesión viva.** Un resultado que llega cuando el socket ya cayó
 *     se descarta: encolarlo era que saliera horas después, en otra llamada.
 *  3. **El aviso de caras solo sale cuando cambia quién está delante**, no cada
 *     fotograma. El modelo no necesita lo mismo cada cuatro segundos.
 *
 * Y una cuarta que no es de aquí pero pasa por aquí: la voz del dueño es lo que
 * sostiene el modo confianza (`confianza.ts`). Si el que habla es otro, la
 * ventana se acaba sola y nadie tiene que acordarse de apagar nada. Hoy eso no
 * cambia nada porque las confirmaciones están apagadas (ADR 0005); el camino se
 * queda igual.
 *
 * Lo que NO está aquí: arrancar y parar el vigilante, que es del ciclo de vida
 * de la llamada y vive donde vive la conexión.
 */
import { useRef, useState } from 'react';

import { avisoCaras, avisoHablante, esElSenor } from '../identidad/quien-hay';
import { vigilante, type CaraDetectada } from '../identidad/identidad';

/** Cuánto tiene que pasar para repetir el mismo aviso, en ms. */
export const NO_REPETIR_MS = 60_000;

/** Cuántos avisos recientes se recuerdan para poder quitarlos de lo hablado. */
const AVISOS_RECORDADOS = 3;

/**
 * ¿Se manda este aviso, o es el mismo de hace nada?
 *
 * Aparte del gancho para poder probarla sin React: es la regla que evita que el
 * modelo reciba la misma presentación en bucle.
 */
export function avisoNuevo(
  ultimo: { texto: string; cuando: number },
  texto: string,
  ahora: number,
): boolean {
  return !(ultimo.texto === texto && ahora - ultimo.cuando < NO_REPETIR_MS);
}

/**
 * La huella de quiénes se ven ahora, para saber si ha cambiado.
 *
 * Las caras sin nombre se descartan: sin el filtro saldría un literal «null» en
 * el aviso. Ordenadas, porque el orden en que las detecte la cámara no es
 * información.
 */
export function huellaDeCaras(lista: CaraDetectada[]): string {
  return lista
    .map(c => c.nombre)
    .filter((n): n is string => !!n)
    .sort()
    .join(', ');
}

type Opciones = {
  /** El perfil que se considera el del dueño. Sale de Ajustes. */
  perfilDueno: () => string;
  /** Para decirle al modelo quién hay delante. */
  informar: (texto: string) => void;
  /** Una línea en la transcripción, que es también la bitácora de la llamada. */
  anotar: (texto: string) => void;
  /** Se le ha oído a él: sostiene el modo confianza. */
  alOirAlDueno: () => void;
  /** ¿Hay socket ahora mismo? Un aviso sin nadie al otro lado se tira.
   *
   *  Se pregunta en vez de guardarse aquí porque el mismo dato lo necesita el
   *  botón de hablar, y dos copias de «hay llamada» es como se desincronizan. */
  haySocket: () => boolean;
};

export function useIdentidad(opciones: Opciones) {
  const [hablante, setHablante] = useState<string | null>(null);
  const [caras, setCaras] = useState<CaraDetectada[]>([]);

  // Quién habla, para las herramientas. El estado se pinta; este espejo es el
  // que viaja al núcleo con cada llamada a una herramienta, porque el callback
  // que la ejecuta se registra una sola vez y no vería el estado nuevo.
  const hablanteRef = useRef<string | null>(null);
  const ultimoAviso = useRef<{ texto: string; cuando: number }>({ texto: '', cuando: 0 });
  // Los últimos avisos mandados, para reconocerlos si el modelo los lee en voz
  // alta y quitarlos de la transcripción. Tres bastan: uno viejo ya no puede
  // estar saliendo por la boca de Perseo.
  const avisos = useRef<string[]>([]);
  const carasVistas = useRef('');

  const avisar = (texto: string) => {
    if (!opciones.haySocket()) return;
    const ahora = Date.now();
    if (!avisoNuevo(ultimoAviso.current, texto, ahora)) return;
    ultimoAviso.current = { texto, cuando: ahora };
    avisos.current = [texto, ...avisos.current].slice(0, AVISOS_RECORDADOS);
    opciones.informar(texto);
  };

  /** Registra los callbacks del vigilante. Se llama una sola vez. */
  const enganchar = () => {
    vigilante.onHablante = (nombre) => {
      setHablante(nombre);
      hablanteRef.current = nombre;
      if (esElSenor(nombre, opciones.perfilDueno())) opciones.alOirAlDueno();
      if (nombre) {
        opciones.anotar(`Habla ${nombre}.`);
        // El aviso dice quién habla Y qué trato le toca. Un nombre a secas
        // dejaba al modelo llamando «señor Persus» a cualquiera que pasara por
        // delante de la cámara. Ver `quien-hay.ts`.
        avisar(avisoHablante(nombre, opciones.perfilDueno()));
      }
    };
    vigilante.onCaras = (lista) => {
      setCaras(lista);
      const clave = huellaDeCaras(lista);
      if (!clave) {
        carasVistas.current = '';
        return;
      }
      if (clave === carasVistas.current) return;
      carasVistas.current = clave;
      const aviso = avisoCaras(clave.split(', '), opciones.perfilDueno());
      if (aviso) avisar(aviso);
    };
  };

  /**
   * Se acabó la sesión: se olvida quién había.
   *
   * Presentación nueva en la llamada siguiente: sin limpiar la huella, si la
   * misma cara sigue delante al reconectar, nadie se lo diría otra vez al
   * modelo.
   */
  const olvidar = () => {
    setHablante(null);
    setCaras([]);
    carasVistas.current = '';
    ultimoAviso.current = { texto: '', cuando: 0 };
  };

  /**
   * Perseo acaba de preguntarle su nombre a quien tenía delante y lo ha
   * guardado. La pantalla no puede seguir enseñando «Desconocido 2» después de
   * que el propio interesado haya dicho cómo se llama.
   */
  const renombrar = (etiqueta: string, nombre: string) => {
    setHablante(previo => (previo === etiqueta ? nombre : previo));
    setCaras(previas => previas.map(c => (c.nombre === etiqueta ? { ...c, nombre } : c)));
    // El aviso de caras solo sale cuando cambia quién está delante: sin limpiar
    // la huella, el nombre nuevo no llegaría al modelo hasta que alguien
    // entrara o saliera del encuadre.
    carasVistas.current = '';
  };

  return {
    hablante,
    caras,
    hablanteRef,
    /** Los últimos avisos, para quitarlos de la transcripción si los lee. */
    avisos,
    avisar,
    enganchar,
    olvidar,
    renombrar,
  };
}
