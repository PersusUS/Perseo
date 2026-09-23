/**
 * Cuánto esperar antes de volver a llamar a Gemini, y por qué.
 *
 * Vive aparte de `gemini-live.ts` porque es aritmética pura y es justo la parte
 * que se portó mal: el 2026-08-17 la app se quedó horas en «Conectando…»
 * abriendo una sesión por segundo. El contador de intentos se ponía a cero **al
 * abrir** el socket, no al sobrevivir con él, así que una sesión que moría un
 * segundo después de nacer dejaba la espera en `2^0 = 1 s` para siempre.
 */

/** Tope de la espera normal: una caída de red se arregla sola cuando vuelve. */
export const ESPERA_MAXIMA_RECONEXION = 30_000;

/**
 * Espera mínima cuando el servidor habla de límites. Un `1011` con «quota»
 * dentro no se arregla reintentando rápido: cada intento cuenta para el mismo
 * límite que acaba de saltar, así que reintentar a lo loco garantiza que el
 * siguiente también falle.
 */
export const ESPERA_TRAS_LIMITE = 65_000;

/** Tope de la espera cuando es un límite: cinco minutos, y no más. */
export const ESPERA_MAXIMA_TRAS_LIMITE = 300_000;

/**
 * Cuánto tiene que aguantar una sesión para considerarla buena. Por debajo de
 * esto el contador de intentos **no** se reinicia, que es lo que impide el
 * bucle de un intento por segundo.
 */
export const MS_SESION_ESTABLE = 30_000;

/** Cuánto se espera a que una conexión conteste antes de darla por muerta. */
export const MS_TOPE_CONEXION = 20_000;

export interface CierreConexion {
  codigo?: number;
  motivo?: string;
}

interface PlanReintento {
  esperaMs: number;
  /** `limite` cuando el servidor está diciendo que hemos pedido demasiado. */
  causa: 'limite' | 'normal';
}

/**
 * ¿Está el servidor hablando de un límite nuestro?
 *
 * El `1011` de la API en vivo es un «internal error» genérico y el motivo llega
 * en texto: «You exceeded your current quota…». Se mira el texto además del
 * código porque el mismo mensaje aparece con otros códigos.
 */
export function esLimite(cierre: CierreConexion): boolean {
  const motivo = cierre.motivo ?? '';
  if (/quota|exceed|rate limit|resource[ _]?exhausted|too many/i.test(motivo)) return true;
  return cierre.codigo === 1011 || cierre.codigo === 1013 || cierre.codigo === 429;
}

/**
 * Cuánto esperar antes del siguiente intento.
 *
 * `intentos` es cuántos van seguidos **sin una sesión estable**: se reinicia
 * cuando una llamada aguanta `MS_SESION_ESTABLE`, no cuando el socket abre.
 */
export function planificarReintento(intentos: number, cierre: CierreConexion): PlanReintento {
  const seguros = Math.max(0, Math.floor(intentos));
  if (esLimite(cierre)) {
    return {
      causa: 'limite',
      esperaMs: Math.min(ESPERA_TRAS_LIMITE * Math.pow(2, seguros), ESPERA_MAXIMA_TRAS_LIMITE),
    };
  }
  return {
    causa: 'normal',
    esperaMs: Math.min(Math.pow(2, seguros) * 1000, ESPERA_MAXIMA_RECONEXION),
  };
}

/** El texto que ve el usuario, para que la espera larga no parezca un cuelgue. */
export function avisoDeEspera(plan: PlanReintento): string {
  const segundos = Math.round(plan.esperaMs / 1000);
  const cuando = segundos >= 60 ? `${Math.round(segundos / 60)} min` : `${segundos} s`;
  return plan.causa === 'limite'
    // El nombre del proveedor no sale a pantalla: en la llamada todo se llama
    // «el enlace de voz», que es lo que el señor Persus ve. Lo que importa para
    // entender el aviso —que es un límite de conexiones y cuánto se espera— sí
    // está entero.
    ? `El enlace de voz está limitando las conexiones. Se reintenta en ${cuando}.`
    : `Reconectando en ${cuando}.`;
}

/**
 * La ventana de contexto se desliza en vez de llenarse.
 *
 * Con la pantalla puesta cada captura son cientos de tokens, y el contexto de
 * la sesión tiene techo: sin esto, una llamada larga acaba cortándose cuando
 * se llena. Con la ventana deslizante el servidor olvida lo más viejo y sigue.
 * Si un día la API la rechazara, `trasElCierre` lo nota y la llamada sigue sin
 * ella: nunca puede ser la causa de no poder hablar.
 */
export const CONTEXTO_DESLIZANTE = { slidingWindow: {} };

interface DecisionCierre {
  /** El testigo de reanudación no sirve: se tira y se empieza de cero. */
  tirarTestigo: boolean;
  /** Reintentar sin la espera larga que tocaría por los fallos. */
  reintentarYa: boolean;
  /** El servidor no acepta la ventana deslizante: fuera. */
  sinCompresion: boolean;
  porque: string;
}

/**
 * Qué hacer tras un cierre del socket, con el testigo y con la compresión.
 *
 * Un testigo de sesión caducado no da error: el servidor cierra con 1007
 * «Invalid session handle», o con 1008 «BidiGenerateContent session not found»
 * —lo segundo es lo que sale al despertar el portátil, medido en
 * `llamada.log` el 2026-09-23: dos intentos seguidos perdidos—. Como el testigo
 * se guardaba igual y el reintento lo volvía a mandar, cada intento fallaba
 * idéntico. Se tira y se empieza de cero, y se reintenta ya.
 *
 * Y el servidor no siempre dice que el testigo es el problema: puede aceptar la
 * sesión y cerrarla acto seguido. Dos intentos seguidos que ni llegan a
 * estables con el testigo puesto bastan para sospechar de él: una llamada sin
 * memoria vale infinitamente más que una llamada que no conecta.
 */
export function trasElCierre(cierre: {
  codigo?: number;
  motivo: string;
  hayTestigo: boolean;
  intentos: number;
}): DecisionCierre {
  const { codigo, motivo, hayTestigo, intentos } = cierre;
  const sinCompresion = /contextWindowCompression|sliding_?window/i.test(motivo);
  const testigoRechazado =
    codigo === 1007 ||
    /session handle/i.test(motivo) ||
    (codigo === 1008 && /session not found/i.test(motivo));

  if (hayTestigo && testigoRechazado) {
    return {
      tirarTestigo: true,
      reintentarYa: true,
      sinCompresion,
      porque: 'El servidor rechazó el testigo de sesión; se empieza de cero.',
    };
  }
  if (hayTestigo && intentos >= 1) {
    return {
      tirarTestigo: true,
      reintentarYa: sinCompresion,
      sinCompresion,
      porque: 'Dos sesiones cortas seguidas con testigo; se descarta.',
    };
  }
  return {
    tirarTestigo: false,
    reintentarYa: sinCompresion,
    sinCompresion,
    porque: sinCompresion ? 'El servidor no acepta la ventana deslizante; se sigue sin ella.' : '',
  };
}
