/**
 * Qué versión de la interfaz es esta.
 *
 * El valor lo incrusta Vite al construir (`__PERSEO_BUILD__`, ver vite.config.ts)
 * y sale de `perseo actualizar`, que sella las dos interfaces con la **misma**
 * marca: esta, que viaja dentro del binario, y `<datos>/version.json`, que el
 * núcleo le sirve al móvil.
 *
 * Existe por un fallo que se repitió varias veces: se tocaba `RealTime/src`, se
 * daba por hecho, y la aplicación seguía abriendo la construcción anterior —
 * porque el `.exe` lleva el `dist` dentro y no se había vuelto a construir. Con
 * la marca a la vista en las dos pantallas, una versión vieja se ve en un
 * segundo en lugar de discutirse.
 */

/** La marca de esta construcción: `AAAAMMDD-HHMM` más la revisión de git.
 *
 *  El `typeof` no sobra: si alguien sirve esta interfaz con una configuración
 *  de Vite que no declare `__PERSEO_BUILD__`, la constante lanzaría al cargar
 *  el módulo y la pantalla entera se quedaría en blanco por un dato que solo
 *  sirve para un pie de página. Una marca desconocida es «?», no una caída. */
export const CONSTRUCCION: string =
  typeof __PERSEO_BUILD__ === 'undefined' ? 'dev' : __PERSEO_BUILD__;

/** `true` cuando esto corre desde el servidor de desarrollo y no de un binario
 *  construido: ahí la marca no significa nada y conviene decirlo. */
export const EN_DESARROLLO = CONSTRUCCION === 'dev';
