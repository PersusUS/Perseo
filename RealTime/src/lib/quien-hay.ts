/**
 * Quién está delante, dicho de forma que el modelo no se confunda.
 * El aviso de identidad viajaba antes como «[IDENTIDAD] Ahora habla X» y con
 * eso no bastaba: en la llamada del 2026-08-25 la cámara etiquetó al padre del
 * señor Persus como «Desconocido 1», Perseo no supo qué hacer con la etiqueta
 * y siguió tratándole de señor Persus. Un nombre a secas no dice lo único que
 * importa aquí — si quien habla es el dueño de la casa o es una visita—, así
 * que las cadenas lo dicen a las claras y viven en este módulo, sin WebSocket
 * ni React de por medio, para poder probarlas.
 * Regla que sostiene todo lo demás: **el trato del dueño es de una sola
 * persona** —`TRATO_DUENO`, aquí abajo—. Con
 * cualquier otra persona ese trato es un error de bulto —le dice a un invitado
 * que es otro—, y además filtra a quien no debe la agenda, el buzón y la
 * memoria del dueño.
 */

/** Perfil que se considera el del dueño mientras Ajustes no diga otra cosa. */
export const PERFIL_PERSUS_POR_DEFECTO = 'Persus';

/**
 * **Los dos ajustes que cambia quien clone esto**, y los únicos dos sitios
 * donde el dueño de esta casa aparece escrito.
 * `TRATO_DUENO` es cómo le llama Perseo delante de cualquiera, y viaja dentro
 * de todos los avisos `[IDENTIDAD]` que este módulo redacta. Va con artículo
 * —«el señor Persus», «la señora Lovelace»— porque las frases lo necesitan;
 * cuando hace falta sin él, lo quita `sinArticulo`.
 * Cambiar esta constante cambia la llamada entera. El prompt largo de la voz
 * es aparte y se edita en Ajustes (`systemPrompt`).
 */
const TRATO_DUENO = 'el señor Persus';

/**
 * Nombres que se dan por suyos aunque el ajuste apunte a otro perfil. El
 * reconocimiento aprende solo y el perfil puede acabar llamándose «Jesús»
 * porque así lo dijo él al enrolarse; tratarle de visita por una letra sería
 * peor que la suposición.
 * Van en minúsculas y sin tildes: se comparan ya normalizados.
 */
const ALIAS_DEL_DUENO = ['persus', 'jesus', 'jesus perez bazarot'];

/** El trato sin el artículo: «señor Persus», para entrecomillarlo. */
function sinArticulo(trato: string): string {
  return trato.replace(/^(el|la|los|las)\s+/i, '');
}

/** El trato a gritos —«EL SEÑOR PERSUS»—, para el censo. */
const TRATO_MAYUSCULAS = TRATO_DUENO.toUpperCase();

/** «del señor Persus», «de la señora Lovelace»: la contracción, bien hecha. */
const TRATO_POSESIVO = /^el\s+/i.test(TRATO_DUENO)
  ? `del ${sinArticulo(TRATO_DUENO)}`
  : `de ${TRATO_DUENO}`;

/** El trato entrecomillado, para decirle al modelo que NO lo use con alguien. */
const TRATO_COMILLAS = `«${sinArticulo(TRATO_DUENO)}»`;

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
  return limpio === normalizar(perfilPersus) || ALIAS_DEL_DUENO.includes(limpio);
}

/**
 * Cómo se nombra a una persona DENTRO del aviso al modelo. Nunca sale un
 * nombre pelado: cada uno viaja con su condición, que es lo que decide el
 * trato.
 */
export function etiquetaPersona(nombre: string, perfilPersus: string): string {
  if (esElSenor(nombre, perfilPersus)) return TRATO_DUENO;
  if (esProvisional(nombre)) {
    return `${nombre} (etiqueta provisional del reconocimiento, NO es su nombre ni es ${TRATO_DUENO})`;
  }
  return `${nombre} (NO es ${TRATO_DUENO})`;
}

/** El aviso de quién habla ahora, listo para `informarIdentidad`. */
export function avisoHablante(nombre: string, perfilPersus: string): string {
  if (esElSenor(nombre, perfilPersus)) {
    return `[IDENTIDAD] Quien habla ahora es ${TRATO_DUENO}. Trato de siempre.`;
  }
  if (esProvisional(nombre)) {
    return (
      `[IDENTIDAD] Quien habla ahora NO es ${TRATO_DUENO}: es ${etiquetaPersona(nombre, perfilPersus)}. ` +
      `No le llames ${TRATO_COMILLAS} ni le trates como a él. Preséntate, pregúntale su nombre con naturalidad ` +
      `y, en cuanto te lo diga, llama a 'nombrar_persona' con etiqueta='${nombre}' y su nombre real. ` +
      `Nada privado ${TRATO_POSESIVO} (agenda, correo, notas, encargos) sale de tu boca delante de él.`
    );
  }
  return (
    `[IDENTIDAD] Quien habla ahora es ${nombre}, que NO es ${TRATO_DUENO}. Trátale de usted por su nombre, ` +
    `nunca de ${TRATO_COMILLAS}. Si necesitas saber quién es, lee su nota en «Perseo/Personas» con el servidor ` +
    `MCP vault. Nada privado ${TRATO_POSESIVO} sale de tu boca delante de él sin que él lo autorice en voz alta.`
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
      ` Hay alguien que no es ${TRATO_DUENO}: no le llames así, y no cuentes delante de esa persona` +
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
    if (esElSenor(p.nombre, perfilPersus)) return `- ${p.nombre} — ES ${TRATO_MAYUSCULAS} (${muestras})`;
    if (esProvisional(p.nombre)) {
      return `- ${p.nombre} — alguien a quien reconoces pero cuyo nombre aún no sabes (${muestras})`;
    }
    return `- ${p.nombre} — NO es ${TRATO_DUENO} (${muestras})`;
  });
  return (
    '[PERSONAS QUE YA RECONOCES POR VOZ O POR CARA — el reconocimiento corre en este ordenador ' +
    'y te avisa con líneas [IDENTIDAD] durante la llamada]:\n' +
    lineas.join('\n')
  );
}

/** La marca con la que viaja todo aviso de identidad. */
const MARCA = '[IDENTIDAD]';

/**
 * Ecos cortos: lo que el modelo dice cuando resume el aviso en vez de leerlo.
 * Van anclados al principio porque solo se aplican a lo que viene JUSTO
 * después de la marca; ahí, por construcción, no hay palabras de Perseo.
 */
const ECOS: RegExp[] = [
  /^Quien habla ahora[^.\n]*\.?\s*/i,
  /^Delante de la cámara:[^.\n]*\.?\s*/i,
  /^Ahora habla[^.\n]*\.?\s*/i,
  // El eco recortado: la marca, un nombre pelado y la frase de verdad pegada
  // detrás sin puntuación —«[IDENTIDAD] Persus Buenos días, señor Persus»—.
  /^(Desconocido \d+|[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÜÑáéíóúüñ]*)\s+(?=[«¡¿A-ZÁÉÍÓÚÑ])/,
];

function limpiarEco(resto: string): string {
  for (const eco of ECOS) {
    const limpio = resto.replace(eco, '');
    if (limpio !== resto) return limpio;
  }
  return resto;
}

/**
 * Quita de una transcripción el aviso de identidad que el modelo haya leído.
 * Las instrucciones ya le dicen que esas líneas no se leen en voz alta, y aun
 * así el 2026-08-26 abrió una llamada con «[IDENTIDAD] Persus Buenos días,
 * señor Persus». Que se cuele en el audio es cosa del modelo; que se quede
 * escrito en la pantalla y en la bitácora del vault, no: eso sí está en
 * nuestra mano.
 * `avisos` son los que la aplicación mandó hace poco. Cuando el modelo los lee
 * literales, esa comparación los borra enteros —es la única forma de saber
 * dónde acaba el aviso y empieza lo que Perseo dice—; cuando los resume, queda
 * la marca y de ella tira `ECOS`.
 * Solo toca el texto donde encuentra la marca o un aviso conocido, así que
 * pasar dos veces la misma cadena —la transcripción llega troceada y se
 * relimpia entera a cada trozo— no come nada de lo que Perseo sí dijo.
 */
export function sinAvisoDeIdentidad(texto: string, avisos: string[] = []): string {
  let salida = texto;
  for (const aviso of avisos) {
    if (!aviso) continue;
    // Con marca y sin ella: el modelo suele soltar la etiqueta y leer el resto.
    for (const forma of [aviso, aviso.replace(MARCA, '').trim()]) {
      if (forma && salida.includes(forma)) salida = salida.split(forma).join('');
    }
  }
  let corte = salida.indexOf(MARCA);
  while (corte !== -1) {
    const antes = salida.slice(0, corte);
    const resto = salida.slice(corte + MARCA.length).replace(/^[ \t]*/, '');
    salida = antes + limpiarEco(resto);
    corte = salida.indexOf(MARCA);
  }
  return salida.replace(/^\s+/, '');
}
