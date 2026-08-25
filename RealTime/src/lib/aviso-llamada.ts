/**
 * Qué se le dice al modelo sobre QUIÉN empezó la llamada y qué avisos quedan
 * por contar.
 *
 * Sin una etiqueta explícita del sentido, el modelo inventaba el motivo: el
 * señor Persus llamaba él a Perseo y este abría con «el sistema me ha llamado
 * para informarle de un encargo terminado» — y lo repetía tras cada
 * reconexión, porque su memoria de sesión arrastraba la invención. Visto en
 * llamada real el 2026-08-25. Las cadenas viven aquí solas para poder
 * probarlas sin WebSocket, como la aritmética de `reconexion.ts`.
 */

export type OrigenLlamada = 'entrante' | 'saliente';

/**
 * La etiqueta del sentido de la llamada, para las instrucciones de sistema.
 * Saliente exige motivo: un tag de «has llamado tú» sin decir por qué deja al
 * modelo igual de perdido que ninguno. Devuelve null cuando no toca etiqueta.
 */
export function etiquetaOrigen(
  origen: OrigenLlamada,
  motivo?: string | null,
): string | null {
  if (origen === 'entrante') {
    return '[LLAMADA ENTRANTE — te ha llamado EL SEÑOR PERSUS a ti. Tú NO has llamado ni el sistema por ti: no hay ningún encargo terminado que anunciar ni interrupción de la que disculparte. Escúchale y atiende lo que pida.]';
  }
  if (!motivo) return null;
  return `[LLAMADA SALIENTE — la has hecho TÚ por iniciativa del sistema, porque un encargo terminó. MOTIVO DE LA LLAMADA — cuéntaselo lo primero]: ${motivo}`;
}

/**
 * Encargos terminados fuera de la conversación. Va redactado como CONTEXTO y
 * nunca como motivo: «informáselo» a secas era justo lo que hacía que en una
 * llamada entrante el modelo creyera que había llamado para eso.
 */
export function etiquetaEncargosResueltos(pendientes: string[]): string | null {
  if (!pendientes.length) return null;
  return `[ENCARGOS QUE TERMINARON MIENTRAS NO HABLABAIS — es contexto para la charla, NO el motivo de esta llamada]:\n- ${pendientes.join('\n- ')}`;
}

/**
 * El cierre del asunto. Cuando el modelo ya ha contado los avisos, este texto
 * los da por informados: sin él seguía presentándolos como pendientes el
 * resto de la llamada — la escena del señor Persus diciendo «que no» una y
 * otra vez.
 */
export const CIERRE_DE_AVISOS =
  '[LOS ENCARGOS ANTERIORES YA LOS CONOCE EL SEÑOR PERSUS — QUEDAN INFORMADOS Y CERRADOS. Ese asunto ha TERMINADO ya: no vuelvas a mencionarlos como noticia, como pendiente ni como motivo de esta llamada.]';

/**
 * El mismo contenido por texto en vivo al abrir la sesión: es el único canal
 * que llega también a una sesión restaurada por testigo, que ignora las
 * instrucciones nuevas (comprobado el 2026-08-23).
 */
export function entregaEnVivo(
  motivo: string | null,
  pendientes: string[],
): string | null {
  const lineas: string[] = [];
  if (motivo) lineas.push(`Motivo de esta llamada: ${motivo}`);
  if (pendientes.length) {
    lineas.push('Resultados que debías contar:', ...pendientes.map(p => `- ${p}`));
  }
  return lineas.length ? lineas.join('\n') : null;
}
