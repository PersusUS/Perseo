/**
 * La puerta del panel al núcleo, en un sitio y no en cuatro pantallas.
 *
 * «Las caras no piensan» está en `AGENTS.md` desde el principio y lo comprueba
 * `commands/arquitectura.py`: un componente de React que llama a Rust mezcla
 * dos trabajos —pintar y decidir qué pedir— y, de paso, no se puede probar sin
 * la aplicación delante. Lo que hacían `comun.ts`, `piezas.tsx`, `ChatTab.tsx`
 * y `AgentesTab.tsx` era abrir esa puerta cada uno por su cuenta.
 *
 * Aquí no se decide nada tampoco: esto es el cartero. Cada función es una ruta
 * del núcleo con su nombre en castellano y sus tipos puestos, y lo único que
 * añade sobre `invoke` es que el nombre del comando de Rust —`panel_encolar`,
 * `chat_sesion`— se escribe **una vez**. Un nombre mal escrito era hasta hoy un
 * fallo en tiempo de ejecución dentro de una pestaña; ahora es un sitio que
 * mirar.
 *
 * Los tipos se importan con `import type` a propósito: se borran al compilar,
 * así que `comun.ts` puede seguir usando estas funciones sin que aparezca un
 * ciclo de verdad entre los dos ficheros.
 */
import { invoke } from '@tauri-apps/api/core';

import type { Estado, Trabajo } from '../../components/panel/comun';

// --------------------------------------------------------------------------- //
// La cola
// --------------------------------------------------------------------------- //

/** Encola un trabajo. Devuelve el trabajo recién nacido, aún pendiente. */
export function encolar(agente: string, peticion: unknown): Promise<Trabajo> {
  return invoke<Trabajo>('panel_encolar', { agente, peticion });
}

/** Un trabajo por su número, con el estado que tenga ahora mismo. */
export function trabajo(id: number): Promise<Trabajo> {
  return invoke<Trabajo>('panel_trabajo', { id });
}

/** Los últimos trabajos de la cola, los más nuevos primero. */
export async function trabajos(limite = 50): Promise<Trabajo[]> {
  const datos = await invoke<{ trabajos: Trabajo[] }>('panel_trabajos', { limite });
  return datos.trabajos ?? [];
}

/** Contesta a una confirmación pendiente: `aprobar` o `rechazar`. */
export function responder(id: number, decision: string): Promise<unknown> {
  return invoke('panel_responder', { id, decision });
}

/** El detalle de lo que hizo un encargo de `dev` mientras trabajaba. */
export function actividad<T>(id: number): Promise<T> {
  return invoke<T>('panel_actividad', { id });
}

/** El cuadro de estado entero: piezas, cola, cuota, máquina. */
export function estado(): Promise<Estado> {
  return invoke<Estado>('panel_estado');
}

/** Qué se ha hecho con cada correo triado, por id. */
export async function correos(): Promise<Record<string, string>> {
  const datos = await invoke<{ marcados: Record<string, string> }>('panel_correos');
  return datos.marcados ?? {};
}

/** Marca un correo: leído, archivado, lo que diga la pestaña. */
export function marcarCorreo(id: string, estado: string): Promise<unknown> {
  return invoke('panel_marcar_correo', { id, estado });
}

/** Enciende el modo confianza durante `minutos`, o lo apaga con `null`.
 *
 *  Con las confirmaciones apagadas (ADR 0005) esto no cambia nada, y por eso
 *  el panel ni siquiera pinta el botón. La función se queda: el día que se
 *  rearmen, el camino tiene que seguir aquí. */
export function confianza(minutos: number | null): Promise<unknown> {
  return invoke('panel_confianza', { minutos });
}

// --------------------------------------------------------------------------- //
// El chat escrito
// --------------------------------------------------------------------------- //

/** Las conversaciones guardadas, sin sus mensajes. */
export async function chatSesiones<T>(): Promise<T[]> {
  const datos = await invoke<{ sesiones: T[] }>('chat_sesiones');
  return datos.sesiones ?? [];
}

/** Una conversación con sus mensajes dentro. */
export function chatSesion<T>(id: number): Promise<T> {
  return invoke<T>('chat_sesion', { id });
}

/** Dice algo en una conversación. No devuelve la respuesta: se relee la sesión. */
export function chatHablar(id: number, texto: string): Promise<unknown> {
  return invoke('chat_hablar', { id, texto });
}

/** Abre una conversación nueva. */
export function chatCrear<T>(): Promise<T> {
  return invoke<T>('chat_crear');
}

/** Borra una conversación entera. */
export function chatBorrar(id: number): Promise<unknown> {
  return invoke('chat_borrar', { id });
}
