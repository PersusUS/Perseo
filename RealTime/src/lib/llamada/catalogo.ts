/**
 * El catálogo de herramientas de la llamada, pedido al núcleo.
 *
 * Hasta el 2026-09-12 las herramientas estaban declaradas aquí enteras, y otra
 * vez enteras en `perseo_core/agentes/chat.py` para el chat escrito. Dos copias
 * de lo mismo se desincronizan, y se habían desincronizado en algo que no era
 * cosmético: una anunciaba una acción `navegar_url` que el agente `pc` no tiene,
 * y la receta de poner música decía «no pulses Enter» en un sitio y «Enter lanza
 * el resultado» en el otro.
 *
 * Ahora la fuente es `perseo_core/servicios/catalogo.py` y esta cara la pide.
 *
 * **La llamada no espera al catálogo.** Esta cara arranca sin núcleo —pasa cada
 * vez que se abre la app antes de que el núcleo termine de levantarse— y
 * quedarse sin voz porque el catálogo tardó sería mucho peor que hablar con la
 * copia de ayer. Por eso: una espera corta, y si no llega, la copia incrustada.
 *
 * Lo que impide que las dos se separen no es esta petición sino
 * `pruebas/test_catalogo.py`, que compara la copia con el núcleo y pone el CI en
 * rojo si difieren. Para regenerarla:
 *
 *     python commands/perseo.py catalogo --incrustar
 */

import { Behavior, Type } from '@google/genai';
import { invoke } from '@tauri-apps/api/core';

import { apuntar } from './diagnostico';
import { CATALOGO_INCRUSTADO } from './catalogo-incrustado';

/** El esquema tal y como lo manda el núcleo: JSON Schema de toda la vida. */
export type EsquemaNeutro = {
  type: string;
  description?: string;
  enum?: string[];
  properties?: Record<string, EsquemaNeutro>;
  required?: string[];
};

export type HerramientaNeutra = {
  name: string;
  description: string;
  parameters: EsquemaNeutro;
};

/**
 * Cuánto se espera al núcleo antes de tirar de la copia. Segundo y medio: lo
 * que tarda una petición a 127.0.0.1 es un puñado de milisegundos, así que esto
 * solo se agota cuando el núcleo no está — y en ese caso esperar más no lo trae.
 */
const ESPERA_MS = 1500;

const TIPOS: Record<string, Type> = {
  object: Type.OBJECT,
  string: Type.STRING,
  number: Type.NUMBER,
  integer: Type.NUMBER,
  boolean: Type.BOOLEAN,
  array: Type.ARRAY,
};

/** Traduce el esquema neutro al dialecto del SDK, que usa su propio `Type`. */
function aEsquema(neutro: EsquemaNeutro): Record<string, unknown> {
  const salida: Record<string, unknown> = { type: TIPOS[neutro.type] ?? Type.STRING };
  if (neutro.description) salida.description = neutro.description;
  if (neutro.enum) salida.enum = neutro.enum;
  if (neutro.properties) {
    const propiedades: Record<string, unknown> = {};
    for (const [nombre, detalle] of Object.entries(neutro.properties)) {
      propiedades[nombre] = aEsquema(detalle);
    }
    salida.properties = propiedades;
  }
  // Siempre presente, aunque esté vacío: el SDK distingue «sin obligatorios» de
  // «no lo he dicho», y la declaración de antes lo mandaba explícitamente.
  salida.required = neutro.required ?? [];
  return salida;
}

/**
 * Las declaraciones para `live.connect`.
 *
 * `NON_BLOCKING` en todas y a propósito: una herramienta que bloquea deja al
 * modelo mudo mientras el núcleo trabaja, y lo que se quiere es que siga la
 * conversación y cuente el resultado cuando llegue.
 */
export function aDeclaraciones(herramientas: HerramientaNeutra[]) {
  return herramientas.map((h) => ({
    name: h.name,
    behavior: Behavior.NON_BLOCKING,
    description: h.description,
    parameters: aEsquema(h.parameters),
  }));
}

/** El último catálogo bueno. Se pide una vez y vale para toda la sesión. */
let recordado: HerramientaNeutra[] | null = null;

/**
 * El catálogo del núcleo, o la copia incrustada si no contesta a tiempo.
 *
 * Nunca lanza: una llamada sin herramientas nuevas sigue siendo una llamada.
 */
export async function catalogoDeHerramientas(): Promise<HerramientaNeutra[]> {
  if (recordado) return recordado;
  try {
    const espera = new Promise<null>((resolver) => setTimeout(() => resolver(null), ESPERA_MS));
    const peticion = invoke<{ herramientas?: HerramientaNeutra[] }>('catalogo_herramientas');
    const respuesta = await Promise.race([peticion, espera]);
    const herramientas = respuesta?.herramientas;
    if (herramientas && herramientas.length > 0) {
      recordado = herramientas;
      return herramientas;
    }
    apuntar('catalogo: el nucleo no lo dio a tiempo; se usa la copia incrustada');
  } catch (e) {
    apuntar(`catalogo: no se pudo pedir (${e}); se usa la copia incrustada`);
  }
  return CATALOGO_INCRUSTADO;
}
