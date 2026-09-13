/**
 * El micrófono de la llamada: en qué modo está y si el paso está abierto.
 *
 * El tercero que sale de `App.tsx`, y el que de verdad era una cosa aparte. No
 * es `useLlamada` —ver la nota del final—: es lo que decide **si el audio pasa
 * o no**, que son dos modos y cuatro reglas, y ni una de ellas tenía nada que
 * ver con pintar la pantalla.
 *
 * Los dos modos:
 *
 *  · **manos libres** — el paso abierto todo el rato; el modelo decide cuándo
 *    empieza y acaba tu turno.
 *  · **pulsar para hablar** — mudo hasta que se aprieta. El aviso al modelo va
 *    ANTES que el audio: con la detección automática apagada, un trozo que
 *    llegue antes del `activityStart` se tira.
 *
 * Y las reglas que no se ven pero se notan:
 *
 *  1. **Se rearma en cada reconexión.** `audioManager.stop()` devuelve el paso
 *     abierto, así que sin rearmarlo una llamada en modo pulsar volvía de una
 *     reconexión con el micrófono de par en par.
 *  2. **Sin llamada no se abre.** Apretar el botón con el socket caído no hace
 *     nada.
 *  3. **Soltar lo que no está apretado tampoco.** `dejar()` sale por donde
 *     entró si no había turno abierto; si no, cada `pointerleave` cerraría un
 *     turno que no existe.
 *  4. **Ningún botón se queda con el foco mientras se habla.** Pulsar «Llamar»
 *     deja el foco en «Colgar», y la barra espaciadora lo activaría por su
 *     cuenta. Es el cinturón, además del `preventDefault` de las dos teclas.
 *
 * **Por qué no hay un `useLlamada`.** Era el tercero de la lista, y al mirarlo
 * de cerca no es un gancho: es «lo que queda de `App.tsx`» —el socket, el
 * audio, la cámara, la pantalla, la transcripción y los avisos, todos atados al
 * mismo ciclo de vida—. Sacarlo a un fichero movería líneas sin separar nada, y
 * un fichero llamado `useLlamada` que contiene el componente entero miente más
 * que la línea que ahorra.
 */
import { useRef, useState } from 'react';

import { defaultConfig, type ModoMicro } from '../datos/config';

/** ¿Este modo deja el paso abierto sin que nadie apriete nada? */
export function pasoSiempreAbierto(modo: ModoMicro): boolean {
  return modo !== 'pulsar';
}

type Opciones = {
  /** Abre o cierra el paso del audio.
   *
   *  Se recibe en vez de importar el `audioManager`: ese módulo arrastra el
   *  cliente de Gemini entero, que lee `localStorage` al cargarse, y con eso
   *  este fichero no se podría probar fuera de un navegador. */
  transmitir: (abierto: boolean) => void;
  /** ¿Hay socket? Sin llamada, el botón de hablar no hace nada. */
  haySocket: () => boolean;
  /** Se abre un turno: al modelo primero, que el audio llega detrás. */
  abrirTurno: () => void;
  /** Se cierra: y que conteste. */
  cerrarTurno: () => void;
};

export function useMicrofono(opciones: Opciones) {
  const [modo, setModo] = useState<ModoMicro>(defaultConfig.modoMicro);
  // El espejo del modo, para los callbacks que se registran una sola vez y no
  // verían el estado nuevo.
  const modoRef = useRef<ModoMicro>(defaultConfig.modoMicro);
  const [pulsando, setPulsando] = useState(false);
  const pulsandoRef = useRef(false);

  /** Deja el micrófono como pide el modo. Al llamar y en cada reconexión. */
  const armar = () => {
    const actual = defaultConfig.modoMicro;
    modoRef.current = actual;
    setModo(actual);
    opciones.transmitir(pasoSiempreAbierto(actual));
    pulsandoRef.current = false;
    setPulsando(false);
  };

  /** Se aprieta el botón de hablar. */
  const empezar = () => {
    if (modoRef.current !== 'pulsar') return;
    if (!opciones.haySocket() || pulsandoRef.current) return;
    pulsandoRef.current = true;
    setPulsando(true);
    const enfocado = document.activeElement as HTMLElement | null;
    if (enfocado && enfocado.tagName === 'BUTTON') enfocado.blur();
    opciones.abrirTurno();
    opciones.transmitir(true);
  };

  /** Se suelta. */
  const dejar = () => {
    if (!pulsandoRef.current) return;
    pulsandoRef.current = false;
    setPulsando(false);
    opciones.transmitir(false);
    opciones.cerrarTurno();
  };

  /** Un turno que se queda a medias no se cierra solo: lo usa 'disconnected'. */
  const soltarTurno = () => {
    pulsandoRef.current = false;
    setPulsando(false);
  };

  /** Cambiar de modo desde Ajustes, que solo se permite fuera de llamada. */
  const cambiarModo = (nuevo: ModoMicro) => {
    modoRef.current = nuevo;
    setModo(nuevo);
  };

  return {
    modo,
    pulsando,
    /** «Escuchando» de verdad: en manos libres siempre, en pulsar solo si se aprieta. */
    abierto: pasoSiempreAbierto(modo) || pulsando,
    armar,
    empezar,
    dejar,
    soltarTurno,
    cambiarModo,
  };
}
