/**
 * El modo confianza de una llamada, fuera de la pantalla que la abre.
 *
 * Es el primero de los tres ganchos que salen de `App.tsx` —los otros dos,
 * `useLlamada` y `useIdentidad`, están enredados con el ciclo de vida de la
 * conexión y no salen sin una llamada de verdad delante—. Este sí sale entero:
 * son un reloj y una petición, y no toca ni el socket ni el micrófono.
 *
 * **Qué hace.** Mientras hay llamada hay una persona delante, así que lo
 * irreversible deja de pedir un sí que ya se está oyendo. La ventana es corta y
 * se rearma cada vez que se le oye a él, en vez de pedir una hora de golpe:
 * hasta el 2026-09-12 era una sola llamada de 60 minutos al conectar, y si el
 * señor Persus se levantaba dejando la llamada abierta con alguien delante, esa
 * hora seguía corriendo. Con el reconocimiento apagado no hay forma de saber
 * quién habla, y entonces se mantiene la ventana larga: es lo que hace falta
 * para dictar sin que cada frase pida permiso.
 *
 * **Y hoy no cambia nada**, porque las confirmaciones están apagadas enteras
 * (`docs/adr/0005-las-confirmaciones-estan-apagadas.md`). Se queda igualmente,
 * y con sus pruebas: el día que se rearmen, este es el camino.
 */
import { useRef } from 'react';

import * as nucleo from '../datos/panel';

/** Minutos de confianza por llamada cuando se sabe quién habla. */
export const MINUTOS_CON_IDENTIDAD = 10;

/** Y cuando no. Larga a propósito: ver la cabecera. */
export const MINUTOS_SIN_IDENTIDAD = 60;

/**
 * ¿Hay que volver a pedirla, o la que hay todavía cubre?
 *
 * Separada del gancho para poder probarla sin React: es la única regla que hay
 * aquí, y la que evita martillear al núcleo con una petición por frase.
 * Se renueva solo cuando queda **menos de la mitad** de la ventana.
 */
export function tocaRenovar(hastaAhora: number, minutos: number, ahora: number): boolean {
  const hasta = ahora + minutos * 60_000;
  return hasta - hastaAhora >= (minutos * 60_000) / 2;
}

type Confianza = {
  /** Enciende —o alarga— la confianza de esta llamada. */
  renovar: (minutos: number) => void;
  /** La apaga y olvida la ventana. Al colgar: sin persona delante, se acabó. */
  apagar: () => Promise<void>;
  /** Olvida la ventana sin tocar el núcleo. Lo usa el reconectar. */
  olvidar: () => void;
};

export function useConfianza(): Confianza {
  // Hasta cuándo dura, en ms de reloj.
  const hasta = useRef(0);

  const renovar = (minutos: number) => {
    if (!tocaRenovar(hasta.current, minutos, Date.now())) return;
    hasta.current = Date.now() + minutos * 60_000;
    nucleo.confianza(minutos).catch(e =>
      console.warn('[Confianza] No se pudo activar:', e)
    );
  };

  const apagar = async () => {
    hasta.current = 0;
    try {
      await nucleo.confianza(null);
    } catch (e) {
      console.warn('[Confianza] No se pudo apagar:', e);
    }
  };

  const olvidar = () => {
    hasta.current = 0;
  };

  return { renovar, apagar, olvidar };
}
