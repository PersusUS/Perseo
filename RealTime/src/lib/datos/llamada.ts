/**
 * Lo que la pantalla de la llamada le pide a Rust, fuera de la pantalla.
 *
 * Tres rutas que no se parecen en nada salvo en quién las llama: la clave de
 * Gemini, la señal de que alguien ha pedido una llamada sola, y el puente por
 * el que se ejecuta una herramienta del catálogo desde fuera del socket.
 *
 * Están aquí por «las caras no piensan» (`commands/arquitectura.py`): lo que
 * `App.tsx` hacía era escribir a mano el nombre de un comando de Rust, y un
 * nombre mal escrito ahí es un fallo en tiempo de ejecución a mitad de una
 * llamada.
 */
import { invoke } from '@tauri-apps/api/core';

/** La clave de Gemini que guarda Rust. Cadena vacía si aún no hay ninguna. */
export function apiKey(): Promise<string> {
  return invoke<string>('obtener_api_key');
}

/** El motivo de una llamada pedida desde fuera, y la consume al leerla.
 *
 *  Devuelve vacío cuando no hay ninguna esperando, que es lo normal. */
export function consumirAutollamada(): Promise<string> {
  return invoke<string>('consumir_autollamada');
}

/** Ejecuta una herramienta del catálogo sin pasar por el socket de la llamada.
 *
 *  Lo usa lo que ocurre **alrededor** de la conversación —guardar lo hablado,
 *  por ejemplo—, no el modelo: el modelo las pide por su propio camino. */
export function ejecutarHerramienta(toolName: string, argumentos: unknown): Promise<unknown> {
  return invoke('ejecutar_herramienta', {
    toolName,
    argumentos: typeof argumentos === 'string' ? argumentos : JSON.stringify(argumentos),
  });
}
