/**
 * Los parámetros de la corteza, fuera del componente que la dibuja.
 *
 * Una sola ruta, y aun así vive aquí: «las caras no piensan» no admite
 * excepciones por tamaño, porque la excepción de una línea es la que enseña a
 * hacer la siguiente. Ver `commands/arquitectura.py`.
 */
import { invoke } from '@tauri-apps/api/core';

/** Lo que el núcleo dice sobre cómo pintarse, o `null` si no lo sabe. */
export function parametros<T>(): Promise<T | null> {
  return invoke<T | null>('corteza_parametros');
}
