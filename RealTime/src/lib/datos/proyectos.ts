/**
 * Lo que la pantalla de proyectos le pide al núcleo y a Rust.
 *
 * Dos cosas distintas que comparten pantalla: **abrir un proyecto**, que es una
 * petición al núcleo, y **abrir una ventana**, que es cosa de Tauri y no sale
 * de la aplicación. Las dos viven aquí por el mismo motivo: lo que el
 * componente hace es pintar y llamar, no escribir nombres de comandos.
 */
import { invoke } from '@tauri-apps/api/core';

/** Qué hay que hacer para abrir este proyecto, según el núcleo. */
export function abrir<T>(id: string): Promise<T> {
  return invoke<T>('panel_abrir_proyecto', { id });
}

/** La lista de proyectos y de qué fichero sale. */
export function listar<T>(): Promise<T> {
  return invoke<T>('panel_proyectos');
}

/** La ventana del grafo de conocimiento. */
export function ventanaGrafo(ancho: number, alto: number): Promise<unknown> {
  return invoke('ventana_grafo', { ancho, alto });
}

/** Una ventana propia para un proyecto, con su color y su tamaño. */
export function ventanaProyecto(opciones: {
  /** El destino del servicio. Puede faltar: un proyecto en modo servicio sin
   *  destino abriría una ventana sin URL, y quien decide qué hacer con eso es
   *  Rust, igual que antes de que esta función existiera. El tipo lo dice en
   *  vez de taparlo con un valor inventado. */
  url: string | undefined;
  titulo: string;
  color: string;
  ancho: number;
  alto: number;
}): Promise<unknown> {
  return invoke('ventana_proyecto', opciones);
}
