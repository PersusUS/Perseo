/**
 * Quién está delante, dicho de forma que el modelo no se confunda.
 *
 * El aviso de identidad viajaba antes como «[IDENTIDAD] Ahora habla X» y con
 * eso no bastaba: en la llamada del 2026-08-25 la cámara etiquetó al padre del
 * señor Persus como «Desconocido 1», Perseo no supo qué hacer con la etiqueta
 * y siguió tratándole de señor Persus. Un nombre a secas no dice lo único que
 * importa aquí — si quien habla es el dueño de la casa o es una visita—, así
 * que las cadenas lo dicen a las claras y viven en este módulo, sin WebSocket
 * ni React de por medio, para poder probarlas.
 *
 * Regla que sostiene todo lo demás: **solo Jesús es «el señor Persus»**. Con
 * cualquier otra persona ese trato es un error de bulto —le dice a un invitado
 * que es otro—, y además filtra a quien no debe la agenda, el buzón y la
 * memoria del dueño.
 */

/** Perfil que se considera el del dueño mientras Ajustes no diga otra cosa. */
export const PERFIL_PERSUS_POR_DEFECTO = 'Persus';

/**
 * Nombres que se dan por suyos aunque el ajuste apunte a otro perfil. El
 * reconocimiento aprende solo y el perfil puede acabar llamándose «Jesús»
 * porque así lo dijo él al enrolarse; tratarle de visita por una letra sería
 * peor que la suposición.
 */
const ALIAS_DEL_SENOR = ['persus', 'jesus', 'jesus perez bazarot'];

/** Minúsculas, sin tildes y sin espacios de más: comparar nombres, no bytes. */
function normalizar(texto: string): string {
  return texto
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .trim()
    .toLowerCase();
}

/** Una etiqueta que el núcleo puso solo, del tipo «Desconocido 3». */
export function esProvisional(nombre: string): boolean {
  return /^desconocido\s+\d+$/i.test(nombre.trim());
}

/** ¿Este perfil es el del dueño? */
export function esElSenor(nombre: string | null, perfilPersus: string): boolean {
  if (!nombre) return false;
  const limpio = normalizar(nombre);
  if (!limpio || esProvisional(nombre)) return false;
  return limpio === normalizar(perfilPersus) || ALIAS_DEL_SENOR.includes(limpio);
}

/**
 * Cómo se nombra a una persona DENTRO del aviso al modelo. Nunca sale un
 * nombre pelado: cada uno viaja con su condición, que es lo que decide el
 * trato.
 */
export function etiquetaPersona(nombre: string, perfilPersus: string): string {
  if (esElSenor(nombre, perfilPersus)) return 'el señor Persus';
  if (esProvisional(nombre)) {
    return `${nombre} (etiqueta provisional del reconocimiento, NO es su nombre ni es el señor Persus)`;
  }
  return `${nombre} (NO es el señor Persus)`;
}

/** El aviso de quién habla ahora, listo para `informarIdentidad`. */
export function avisoHablante(nombre: string, perfilPersus: string): string {
  if (esElSenor(nombre, perfilPersus)) {
    return '[IDENTIDAD] Quien habla ahora es el señor Persus. Trato de siempre.';
  }
  if (esProvisional(nombre)) {
    return (
      `[IDENTIDAD] Quien habla ahora NO es el señor Persus: es ${etiquetaPersona(nombre, perfilPersus)}. ` +
      'No le llames «señor Persus» ni le trates como a él. Preséntate, pregúntale su nombre con naturalidad ' +
      `y, en cuanto te lo diga, llama a 'nombrar_persona' con etiqueta='${nombre}' y su nombre real. ` +
      'Nada privado del señor Persus (agenda, correo, notas, encargos) sale de tu boca delante de él.'
    );
  }
  return (
    `[IDENTIDAD] Quien habla ahora es ${nombre}, que NO es el señor Persus. Trátale de usted por su nombre, ` +
    'nunca de «señor Persus». Si necesitas saber quién es, lee su nota en «Perseo/Personas» con el servidor ' +
    'MCP vault. Nada privado del señor Persus sale de tu boca delante de él sin que él lo autorice en voz alta.'
  );
}

/**
 * El aviso de a quién se ve por la cámara. Devuelve null cuando no hay nadie
 * identificado: un aviso vacío gasta turno y no dice nada.
 */
export function avisoCaras(nombres: string[], perfilPersus: string): string | null {
  if (!nombres.length) return null;
  const etiquetas = nombres.map(n => etiquetaPersona(n, perfilPersus));
  const visitas = nombres.filter(n => !esElSenor(n, perfilPersus));
  let texto = `[IDENTIDAD] Delante de la cámara: ${etiquetas.join(', ')}.`;
  if (visitas.length) {
    texto +=
      ' Hay alguien que no es el señor Persus: no le llames así, y no cuentes delante de esa persona' +
      ' nada privado suyo. Si aún no sabes quién es, pregúntaselo y guárdalo con la herramienta' +
      " 'nombrar_persona'.";
  }
  return texto;
}

/** Lo que hace falta saber de un perfil para presentarlo. */
export interface PerfilConocido {
  nombre: string;
  voz: boolean;
  caras: number;
}

/**
 * El censo de gente conocida, para las instrucciones de sistema.
 *
 * Es «registro de hablantes por prompt», que es lo que recomiendan los
 * trabajos de diarización con LLM: el modelo entra en la llamada sabiendo a
 * quién puede encontrarse en vez de descubrirlo por un aviso suelto a mitad de
 * frase. Los provisionales van marcados como lo que son — nombres que están
 * esperando a que alguien los diga.
 */
export function bloqueCenso(
  perfiles: PerfilConocido[],
  perfilPersus: string,
): string | null {
  if (!perfiles.length) return null;
  const lineas = perfiles.map(p => {
    const muestras =
      [p.voz ? 'voz' : null, p.caras > 0 ? 'cara' : null].filter(Boolean).join(' y ') ||
      'sin muestras';
    if (esElSenor(p.nombre, perfilPersus)) return `- ${p.nombre} — ES EL SEÑOR PERSUS (${muestras})`;
    if (esProvisional(p.nombre)) {
      return `- ${p.nombre} — alguien a quien reconoces pero cuyo nombre aún no sabes (${muestras})`;
    }
    return `- ${p.nombre} — NO es el señor Persus (${muestras})`;
  });
  return (
    '[PERSONAS QUE YA RECONOCES POR VOZ O POR CARA — el reconocimiento corre en este ordenador ' +
    'y te avisa con líneas [IDENTIDAD] durante la llamada]:\n' +
    lineas.join('\n')
  );
}
