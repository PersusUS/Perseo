/**
 * ¿Ha cambiado la pantalla lo bastante como para volver a mandarla?
 *
 * Hasta el 2026-09-23 se mandaba una captura cada dos segundos aunque fuera la
 * misma: 1.062 capturas en 25 llamadas, ~110 kB cada una, y cada una ocupando
 * contexto del modelo. Casi todas eran iguales a la anterior —se habla con la
 * pantalla quieta—, y cada una de más es latencia y coste sin nada nuevo.
 *
 * Cómo se decide: la captura se reduce a 32×18 en grises y se compara celda a
 * celda con la **última que se mandó** (no con la última que se capturó: si no,
 * un cambio lento nunca llegaría a notarse). Basta con que una celda cambie de
 * verdad para mandar. Así una línea de texto nueva sí cuenta —cambia media
 * docena de celdas de golpe— y el cursor que parpadea o el reloj de la barra
 * de tareas no —mueven una celda unos pocos niveles—.
 *
 * Y cada `REENVIO_MAXIMO_MS` se manda igual, cambie o no: el modelo no debe
 * trabajar con una imagen de hace minutos si algo se le escapó al umbral.
 */

export const ANCHO = 32;
export const ALTO = 18;

/** Cuánto tiene que cambiar el gris medio de una celda (0-255) para contar. */
export const UMBRAL_CELDA = 10;

/** Aunque no cambie nada, cada cuánto se vuelve a mandar. */
export const REENVIO_MAXIMO_MS = 20_000;

/** Celdas que han cambiado de verdad entre dos miniaturas del mismo tamaño. */
export function celdasCambiadas(a: Uint8Array, b: Uint8Array, umbral = UMBRAL_CELDA): number {
  if (a.length !== b.length) return a.length || b.length;
  let cambiadas = 0;
  for (let i = 0; i < a.length; i++) {
    if (Math.abs(a[i] - b[i]) > umbral) cambiadas++;
  }
  return cambiadas;
}

export class CambioDePantalla {
  private enviada: Uint8Array | null = null;
  private ultimoEnvio = -Infinity;
  /** Para el cuaderno: cuántas se mandaron y cuántas se ahorraron. */
  mandadas = 0;
  ahorradas = 0;

  /** ¿Se manda esta? `null` es «no se pudo reducir»: se manda, como siempre. */
  merece(miniatura: Uint8Array | null, ahoraMs: number): boolean {
    const cambia =
      miniatura === null ||
      this.enviada === null ||
      ahoraMs - this.ultimoEnvio >= REENVIO_MAXIMO_MS ||
      celdasCambiadas(this.enviada, miniatura) > 0;
    if (cambia) {
      this.enviada = miniatura;
      this.ultimoEnvio = ahoraMs;
      this.mandadas++;
    } else {
      this.ahorradas++;
    }
    return cambia;
  }

  /** Empieza de cero: la primera captura de una llamada se manda siempre. */
  olvidar(): void {
    this.enviada = null;
    this.ultimoEnvio = -Infinity;
    this.mandadas = 0;
    this.ahorradas = 0;
  }
}

/**
 * La captura en 32×18 grises, o `null` si el navegador no sabe reducirla.
 *
 * `createImageBitmap` descodifica y reduce fuera del hilo principal, que es
 * donde vive el audio: lo que cuesta aquí es solo el `atob` y 576 píxeles.
 */
export async function miniatura(base64Jpeg: string): Promise<Uint8Array | null> {
  if (typeof createImageBitmap !== 'function' || typeof OffscreenCanvas === 'undefined') {
    return null;
  }
  try {
    const binario = atob(base64Jpeg);
    const bytes = new Uint8Array(binario.length);
    for (let i = 0; i < binario.length; i++) bytes[i] = binario.charCodeAt(i);
    const mapa = await createImageBitmap(new Blob([bytes], { type: 'image/jpeg' }), {
      resizeWidth: ANCHO,
      resizeHeight: ALTO,
      resizeQuality: 'medium',
    });
    const lienzo = new OffscreenCanvas(ANCHO, ALTO);
    const ctx = lienzo.getContext('2d');
    if (!ctx) {
      mapa.close();
      return null;
    }
    ctx.drawImage(mapa, 0, 0);
    mapa.close();
    const { data } = ctx.getImageData(0, 0, ANCHO, ALTO);
    const gris = new Uint8Array(ANCHO * ALTO);
    for (let i = 0; i < gris.length; i++) {
      gris[i] = (data[4 * i] * 77 + data[4 * i + 1] * 150 + data[4 * i + 2] * 29) >> 8;
    }
    return gris;
  } catch {
    return null;
  }
}
