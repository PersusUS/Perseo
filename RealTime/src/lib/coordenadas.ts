/**
 * De donde señala el modelo a donde clica el ratón.
 *
 * El modelo no ve la pantalla: ve un JPEG de 1280×720 que le manda
 * `screen-manager.ts`, y señala sobre **esa** imagen con las coordenadas
 * normalizadas de 0 a 1000 con las que Gemini está entrenado para apuntar.
 * `perseo_core/pc.py`, en cambio, clica en píxeles de la pantalla real. Nadie
 * traducía entre las dos cosas, así que un «clica el primer resultado» acababa
 * en cualquier parte — normalmente arriba a la izquierda, porque 0-1000 sobre
 * una pantalla de 1920 se queda a la mitad.
 */

export interface GeometriaPantalla {
  ancho_imagen: number;
  alto_imagen: number;
  ancho_pantalla: number;
  alto_pantalla: number;
}

/** El lado del cuadrado normalizado en el que apunta Gemini. */
export const LADO_NORMALIZADO = 1000;

/** Las acciones de `controlar_pc` cuyo parámetro lleva coordenadas. */
export const ACCIONES_DE_RATON = new Set(['mover_raton', 'click_raton']);

/**
 * Todo lo que `pc.py` sabe hacer, tal cual lo escribe el núcleo.
 *
 * Va como `enum` en la declaración de la herramienta: con la lista solo en la
 * descripción, el modelo mandó `accion: "controlar_pc"` y el núcleo lo trató
 * como acción desconocida —o sea irreversible— dejando el trabajo esperando un
 * sí que nadie vio. Si se añade una acción en `pc.py`, se añade aquí.
 */
export const ACCIONES_PC = [
  'abrir_app',
  'escribir_teclado',
  'atajo_teclado',
  'volumen',
  'mover_raton',
  'click_raton',
  'buscar_youtube',
];

/**
 * Pasa un punto normalizado (0-1000) a píxeles de la pantalla.
 *
 * Se recorta al borde: el modelo redondea, y un 1000 clavado caería un píxel
 * fuera.
 */
export function aPixeles(
  x: number,
  y: number,
  geometria: GeometriaPantalla,
): [number, number] {
  const escala = (valor: number, lado: number) => {
    const pixel = Math.round((valor / LADO_NORMALIZADO) * lado);
    return Math.max(0, Math.min(pixel, lado - 1));
  };
  return [escala(x, geometria.ancho_pantalla), escala(y, geometria.alto_pantalla)];
}

/**
 * Traduce el parámetro de `controlar_pc` dejando intacto todo lo demás.
 *
 * Acepta las mismas formas que `pc.py`, porque el modelo no se pone de acuerdo
 * consigo mismo: `'300,450'`, `'derecho 300,450'`, `'izquierdo'` y `''`. Lo que
 * no lleva coordenadas se devuelve tal cual — un clic sin ellas ya lo rechaza el
 * núcleo, y no es aquí donde se decide eso.
 */
export function traducirParametroDeRaton(
  parametro: string,
  geometria: GeometriaPantalla,
): string {
  return parametro
    .split(/(\s+)/)
    .map((pieza) => {
      const punto = pieza.match(/^(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)$/);
      if (!punto) return pieza;
      const [x, y] = aPixeles(Number(punto[1]), Number(punto[2]), geometria);
      return `${x},${y}`;
    })
    .join('');
}
