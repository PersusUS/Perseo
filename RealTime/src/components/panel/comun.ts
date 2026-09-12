import { invoke } from '@tauri-apps/api/core';

/**
 * Lo que comparten las pestañas del panel: los tipos que viajan del núcleo, las
 * tablas de nombres y los tres formateadores.
 *
 * Salió de `Panel.tsx` el 2026-09-12, cuando el fichero pasaba de mil
 * cuatrocientas líneas. Está aquí y no en cada pestaña porque `Trabajo` y
 * `Estado` son la forma de lo que contesta el núcleo, y dos definiciones de esa
 * forma es exactamente lo que se separa sin que nadie se entere.
 */

export type Pestana = 'chat' | 'agentes' | 'cola' | 'correo' | 'memoria' | 'estado';

/** Una conversación del chat escrito. `turno` es el semáforo que vive en el
 *  núcleo: mientras esté «ocupado», esta vista sondea el texto creciente. */
export type Sesion = { id: number; titulo: string; turno: string; actualizado_en: string };

/** Un mensaje del chat. El de Perseo nace vacío con estado `escribiendo` y su
 *  texto va creciendo en la base mientras el turno trabaja. */
export type Mensaje = {
  id: number;
  rol: 'usuario' | 'perseo';
  texto: string;
  herramientas: string[];
  estado: string;
  momento: string;
};

export type Trabajo = {
  id: number;
  estado: string;
  agente: string;
  origen: string;
  peticion?: any;
  resultado?: any;
  error?: string | null;
  confirmacion?: { resumen?: string; detalle?: string } | null;
  /** Por dónde va un encargo de `dev` que sigue corriendo — «Editando api.py».
   *  Solo llega mientras está en curso, y solo con el motor sobre el SDK: los
   *  que hablan por consola no cuentan nada hasta el final. */
  progreso?: string;
};

export type Pieza = { id: string; nombre: string; estado: string; detalle: string; arreglo: string };

export type Estado = {
  encendido_segundos: number;
  /** Cuándo lo reunió el núcleo, en ISO. Sirve para saber si esta pantalla se
   *  quedó congelada: la diferencia con el reloj se enseña en la lectura. */
  generado?: string;
  piezas: Pieza[];
  trabajos: Record<string, number>;
  agentes: string[];
  disparadores: { nombre: string; activo: boolean; intervalo: number }[];
  cuota: { dia: string; nota: string; servicios: { modelo: string; usadas: number; tope: number | null }[] };
  /** La máquina donde vive el núcleo. Sin `psutil` llega `disponible: false`. */
  maquina?: any;
  /** Qué se está haciendo, qué correo espera y qué toca en la agenda. */
  presencia?: any;
  /** La marca de la última construcción, la que sella `perseo actualizar`. */
  version?: { marca?: string; construido?: string };
};

export const ESTADOS_ABIERTOS = new Set(['pendiente', 'en_curso', 'esperando']);

/** Los estados como se leen. `en_curso` es el nombre que tiene en la base de
 *  datos, con su guion bajo, y enseñarlo tal cual delataba la fontanería. */
export const ESTADO_LEGIBLE: Record<string, string> = {
  pendiente: 'pendiente',
  en_curso: 'en curso',
  esperando: 'esperando',
  hecho: 'hecho',
  fallido: 'fallido',
  cancelado: 'cancelado',
  rechazado: 'rechazado',
};

/** Lo que se lee en la barra, que no tiene por qué ser el identificador
 *  interno: «agentes» no decía nada de qué va la pestaña. */
export const NOMBRES_PESTANA: Record<Pestana, string> = {
  chat: 'chat',
  agentes: 'encargos',
  cola: 'cola',
  correo: 'correo',
  memoria: 'memoria',
  estado: 'estado',
};

/** Cada cuánto se repregunta mientras el panel está delante.
 *  Se sondea en vez de escuchar el flujo SSE: el flujo se autentica por cookie y
 *  aquí no hay cookie — es justo la razón de que este panel exista. */
export const REFRESCO = 4000;
export const REFRESCO_ESTADO = 20000;

export const CLASES_CORREO: Record<string, string> = {
  requiere_accion: 'acción',
  interesante: 'interesante',
  no_seguro: 'sin decidir',
  ignorar: 'ignorar',
};

export const ORDEN_CAJONES = ['requiere_accion', 'no_seguro', 'interesante', 'ignorar'];

export const FILTROS: Record<string, (t: Trabajo) => boolean> = {
  todo: () => true,
  abiertos: t => ESTADOS_ABIERTOS.has(t.estado),
  esperando: t => t.estado === 'esperando',
  mios: t => t.origen !== 'disparador',
  solos: t => t.origen === 'disparador',
  fallidos: t => t.estado === 'fallido',
};

export const NOMBRES_FILTRO: Record<string, string> = {
  todo: 'todo',
  abiertos: 'abiertos',
  esperando: 'esperan un sí',
  mios: 'los pedí yo',
  solos: 'salieron solos',
  fallidos: 'fallidos',
};

export function duracion(segundos: number): string {
  const d = Math.floor(segundos / 86400);
  const h = Math.floor((segundos % 86400) / 3600);
  const m = Math.floor((segundos % 3600) / 60);
  if (d) return `${d} d ${h} h`;
  if (h) return `${h} h ${m} min`;
  return `${m} min`;
}

export function resumirPeticion(t: Trabajo): string {
  // Un trabajo de correo trae el lote entero dentro. Volcarlo llena la pantalla
  // del JSON de veinte correos antes de llegar al resultado.
  const mensajes = t.peticion?.mensajes;
  if (Array.isArray(mensajes)) {
    return `${mensajes.length} correo${mensajes.length === 1 ? '' : 's'} del buzón`;
  }
  return t.peticion?.texto ?? t.peticion?.accion ?? JSON.stringify(t.peticion ?? {});
}

/** Qué pasó con un trabajo, en una línea y en castellano.
 *
 *  El último recurso era `JSON.stringify(resultado)`, y se veía: guardar una
 *  conversación dejaba `{"accion":"conversacion","mensajes":4,"ruta":…,
 *  "titular":null}` en la cola. Un panel que enseña JSON es un panel que se deja
 *  de leer. */
export function resumirResultado(resultado: any): string {
  if (resultado == null) return '';
  if (typeof resultado === 'string') return resultado;
  if (resultado.titular) return String(resultado.titular);
  if (resultado.texto) return String(resultado.texto);
  if (resultado.ruta) {
    const cuantos = typeof resultado.mensajes === 'number'
      ? `${resultado.mensajes} mensaje${resultado.mensajes === 1 ? '' : 's'} · `
      : '';
    return `${cuantos}guardado en ${resultado.ruta}`;
  }
  // Lo que no se sepa resumir se enseña como pares, no como JSON: sigue siendo
  // feo, pero se lee.
  return Object.entries(resultado)
    .filter(([, v]) => v !== null && v !== undefined && v !== '')
    .map(([k, v]) => `${k}: ${typeof v === 'object' ? JSON.stringify(v) : v}`)
    .join(' · ');
}

/** Encola un trabajo y espera su resultado sondeando. */
export async function encolarYEsperar(agente: string, peticion: any, segundos = 20): Promise<any> {
  const trabajo = await invoke<Trabajo>('panel_encolar', { agente, peticion });
  const limite = Date.now() + segundos * 1000;
  while (Date.now() < limite) {
    await new Promise(r => setTimeout(r, 400));
    const actual = await invoke<Trabajo>('panel_trabajo', { id: trabajo.id });
    if (actual.estado === 'hecho') return actual.resultado;
    if (['fallido', 'cancelado', 'rechazado'].includes(actual.estado)) {
      throw new Error(actual.error || `El trabajo quedó ${actual.estado}`);
    }
  }
  throw new Error('Sigue en marcha; míralo en la cola.');
}

/** Una línea de correo triado, y qué se ha hecho con él.
 *
 *  Las acciones solo salen en la pestaña de Correo (`onMarcar`): en la cola,
 *  una línea de correo es el resultado de un trabajo —lo que pasó— y ahí no se
 *  decide nada. Un correo resuelto no se esconde, se apaga: esconderlo quitaría
 *  la única forma de ver que el triaje se ha comido algo. */
