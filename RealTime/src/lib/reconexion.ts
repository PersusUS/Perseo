/**
 * Cuánto esperar antes de volver a llamar a Gemini, y por qué.
 *
 * Vive aparte de `gemini-live.ts` porque es aritmética pura y es justo la parte
 * que se portó mal: el 2026-08-17 la app se quedó horas en «Conectando…»
 * abriendo una sesión por segundo. El contador de intentos se ponía a cero **al
 * abrir** el socket, no al sobrevivir con él, así que una sesión que moría un
 * segundo después de nacer dejaba la espera en `2^0 = 1 s` para siempre. Ver
 * `bitacora/08_LLAMADA.md`.
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

export interface PlanReintento {
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
    ? `Gemini está limitando las conexiones. Se reintenta en ${cuando}.`
    : `Reconectando en ${cuando}.`;
}
