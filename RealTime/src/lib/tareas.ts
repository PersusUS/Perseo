/**
 * El dato del tablero de tareas, fuera de la pantalla que lo dibuja.
 *
 * Mismo reparto que los hábitos (`lib/habitos.ts`) y por el mismo motivo: la
 * pantalla dibuja, aquí se cuenta. Un tablero es media docena de reglas —a qué
 * columna va cada nota, en qué orden queda, qué pasa al tirarla— y esas reglas
 * se prueban sin montar React (`pruebas/tareas.test.ts`).
 *
 * Dónde está guardado: en el `localStorage` de la ventana, bajo `ALMACEN`.
 * Es la misma excepción consciente a «las caras no piensan» que los hábitos:
 * el núcleo no tiene agente de tareas y esto no encola trabajo, así que no hay
 * ninguna decisión aquí — solo notas que el señor Persus mueve con el ratón.
 * Si algún día Perseo tiene que leerlas en voz alta, el camino ya está abierto:
 * `resumen()` devuelve prosa, como el de hábitos, y el espejo se añade encima
 * sin tocar la pantalla.
 *
 * La papelera **no borra**: mueve. Borrar de verdad solo pasa cuando se vacía
 * o cuando se tira una nota que ya estaba en la papelera, y las dos cosas se
 * piden a mano. Una nota que desaparece porque la arrastraste mal es la forma
 * más rápida de dejar de fiarte de un tablero.
 */

/** La clave del almacén. Versionada: si algún día cambia la forma del dato, el
 *  tablero viejo se queda quieto en su clave en vez de reventar al leerlo. */
export const ALMACEN = 'perseo.tareas.v1';

/** Las cuatro columnas del corcho. El orden de este array es el orden en el
 *  que se pintan y el que recorren las flechas de la ficha abierta. */
export const COLUMNAS = ['sin_hacer', 'en_proceso', 'completadas', 'papelera'] as const;
export type Columna = (typeof COLUMNAS)[number];

export const NOMBRES_COLUMNA: Record<Columna, string> = {
  sin_hacer: 'Sin hacer',
  en_proceso: 'En proceso',
  completadas: 'Completadas',
  papelera: 'Papelera',
};

/** Los papeles. Son cinco y no una rueda de color continua a propósito: un
 *  tablero donde cada nota tiene su tono exacto deja de tener tonos, y la
 *  gracia del color es agrupar de un vistazo. Los valores están elegidos para
 *  una pantalla negra —papel apagado, letra oscura— y no para papel blanco. */
export const COLORES = ['amarillo', 'menta', 'cielo', 'rosa', 'lila'] as const;
export type Color = (typeof COLORES)[number];

/** Una nota del corcho.
 *
 *  `giro` se guarda con la tarea y no se calcula al pintar: una inclinación que
 *  se sortea en cada render hace que el tablero entero tiemble cada vez que
 *  mueves una sola nota. Nace una vez y se queda.
 *
 *  `previa` es de dónde vino al caer en la papelera, para que restaurar la
 *  devuelva a su sitio y no a un «sin hacer» genérico. */
export type Tarea = {
  id: string;
  titulo: string;
  detalle: string;
  columna: Columna;
  color: Color;
  giro: number;
  creada: string;
  movida: string;
  previa?: Columna;
};

export type Datos = { tareas: Tarea[] };

/** Un tablero recién estrenado. No arranca vacío del todo: un corcho sin una
 *  sola nota no enseña qué se puede hacer con él, y estas tres se tiran en diez
 *  segundos. Solo salen la primera vez; en cuanto hay algo guardado, manda lo
 *  guardado aunque esté vacío. */
export const TAREAS_INICIALES: { titulo: string; detalle: string; columna: Columna; color: Color }[] = [
  {
    titulo: 'Arrastra esta nota a «En proceso»',
    detalle:
      'Las notas se cogen con el ratón y se sueltan en otra columna. También se ' +
      'pueden soltar entre dos notas de la misma columna para ordenarlas.',
    columna: 'sin_hacer',
    color: 'amarillo',
  },
  {
    titulo: 'Púlsame para ver el detalle',
    detalle:
      'Un clic abre la ficha: el título, el texto largo, el color del papel y ' +
      'los botones para moverla o tirarla. Se guarda solo al escribir.',
    columna: 'sin_hacer',
    color: 'cielo',
  },
  {
    titulo: 'Lo que se tira va a la papelera',
    detalle: 'De la papelera se recupera. Solo se pierde al vaciarla a mano.',
    columna: 'completadas',
    color: 'menta',
  },
];

/** Un identificador que no choca. `randomUUID` no está en todos los contextos
 *  —le hace falta origen seguro—, así que hay reserva: el reloj más azar basta
 *  para un tablero de una sola ventana. */
function nuevoId(): string {
  try {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
      return crypto.randomUUID();
    }
  } catch {
    // Contexto sin cripto: se cae al reloj.
  }
  return `t${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
}

/** La inclinación de una nota nueva: entre -3 y 3 grados. Más que eso y las
 *  notas se pisan en la columna; menos y el corcho parece una hoja de cálculo. */
function giroAlAzar(): number {
  return Math.round((Math.random() * 6 - 3) * 10) / 10;
}

function esColumna(v: unknown): v is Columna {
  return typeof v === 'string' && (COLUMNAS as readonly string[]).includes(v);
}

function esColor(v: unknown): v is Color {
  return typeof v === 'string' && (COLORES as readonly string[]).includes(v);
}

/** Normaliza lo que venga del disco. Todo lo que no encaje se sustituye por un
 *  valor bueno en vez de tirar la nota: perder una tarea porque su color no se
 *  reconoce sería cambiar un fallo de pintura por uno de datos. */
function saneada(t: any): Tarea | null {
  const titulo = String(t?.titulo ?? '').trim();
  const id = String(t?.id ?? '').trim();
  if (!id || !titulo) return null;
  const momento = typeof t?.creada === 'string' ? t.creada : new Date().toISOString();
  return {
    id,
    titulo,
    detalle: String(t?.detalle ?? ''),
    columna: esColumna(t?.columna) ? t.columna : 'sin_hacer',
    color: esColor(t?.color) ? t.color : 'amarillo',
    giro: typeof t?.giro === 'number' && isFinite(t.giro) ? t.giro : 0,
    creada: momento,
    movida: typeof t?.movida === 'string' ? t.movida : momento,
    ...(esColumna(t?.previa) ? { previa: t.previa as Columna } : {}),
  };
}

/** Lee lo guardado. Un almacén ausente estrena tablero; uno ilegible también,
 *  porque una pantalla en blanco no le cuenta a nadie qué ha pasado. */
export function leer(): Datos {
  try {
    const crudo = localStorage.getItem(ALMACEN);
    if (crudo) {
      const d = JSON.parse(crudo) as { tareas?: any[] };
      if (Array.isArray(d.tareas)) {
        return { tareas: d.tareas.map(saneada).filter((t): t is Tarea => t !== null) };
      }
    }
  } catch {
    // Almacén roto: se sustituye. Avisar de esto no le sirve a nadie.
  }
  return { tareas: TAREAS_INICIALES.map(crear) };
}

/** Guarda entero, como los hábitos: son unos kilobytes y el tablero cabe en una
 *  escritura. */
export function guardar(datos: Datos): void {
  try {
    localStorage.setItem(ALMACEN, JSON.stringify(datos));
  } catch {
    // Almacén lleno o bloqueado: la pantalla sigue funcionando en memoria.
  }
}

/** Una nota nueva, ya con su id, su giro y sus fechas. */
export function crear(
  campos: { titulo: string; detalle?: string; columna?: Columna; color?: Color },
): Tarea {
  const ahora = new Date().toISOString();
  return {
    id: nuevoId(),
    titulo: campos.titulo.trim(),
    detalle: campos.detalle ?? '',
    columna: campos.columna ?? 'sin_hacer',
    color: campos.color ?? 'amarillo',
    giro: giroAlAzar(),
    creada: ahora,
    movida: ahora,
  };
}

/** Las notas de una columna, en su orden. El orden es el del array: mover una
 *  nota la saca y la vuelve a meter donde toque, y así lo que se ve en pantalla
 *  y lo que hay en el disco son la misma lista. */
export function deColumna(datos: Datos, columna: Columna): Tarea[] {
  return datos.tareas.filter(t => t.columna === columna);
}

export function cuenta(datos: Datos, columna: Columna): number {
  return deColumna(datos, columna).length;
}

/**
 * Mueve una nota a una columna, opcionalmente delante de otra.
 *
 * `antesDe` es el id de la nota ante la cual cae; sin él, va al final de la
 * columna. Es lo que permite ordenar dentro de la misma columna sin una segunda
 * operación: soltar entre dos notas es mover a la misma columna con vecino.
 *
 * Al entrar en la papelera se apunta de dónde venía. Al salir de ella se olvida,
 * porque a partir de ahí la nota vive donde la hayas dejado.
 */
export function mover(
  datos: Datos, id: string, columna: Columna, antesDe?: string,
): Datos {
  const tarea = datos.tareas.find(t => t.id === id);
  if (!tarea || id === antesDe) return datos;

  const movida: Tarea = {
    ...tarea,
    columna,
    movida: new Date().toISOString(),
    ...(columna === 'papelera'
      ? { previa: tarea.columna === 'papelera' ? tarea.previa : tarea.columna }
      : { previa: undefined }),
  };
  if (movida.previa === undefined) delete movida.previa;

  const resto = datos.tareas.filter(t => t.id !== id);
  const destino = antesDe ? resto.findIndex(t => t.id === antesDe) : -1;
  if (destino < 0) return { tareas: [...resto, movida] };
  return { tareas: [...resto.slice(0, destino), movida, ...resto.slice(destino)] };
}

/** A la papelera. Lo que ya está en la papelera no se tira dos veces: para eso
 *  está `borrar`, que sí es definitivo y se pide desde su propio botón. */
export function tirar(datos: Datos, id: string): Datos {
  return mover(datos, id, 'papelera');
}

/** Vuelve de la papelera a donde estaba, o a «sin hacer» si no consta. */
export function restaurar(datos: Datos, id: string): Datos {
  const tarea = datos.tareas.find(t => t.id === id);
  if (!tarea) return datos;
  return mover(datos, id, tarea.previa ?? 'sin_hacer');
}

/** Borra de verdad. Solo se llama desde la papelera. */
export function borrar(datos: Datos, id: string): Datos {
  return { tareas: datos.tareas.filter(t => t.id !== id) };
}

/** Vacía la papelera. Igual: definitivo y a mano. */
export function vaciarPapelera(datos: Datos): Datos {
  return { tareas: datos.tareas.filter(t => t.columna !== 'papelera') };
}

/** Cambia lo que se edita en la ficha. El id, la columna y las fechas no se
 *  tocan por aquí: mover es `mover` y el histórico no se reescribe. */
export function editar(
  datos: Datos, id: string, cambios: Partial<Pick<Tarea, 'titulo' | 'detalle' | 'color'>>,
): Datos {
  return {
    tareas: datos.tareas.map(t => (t.id === id ? { ...t, ...cambios } : t)),
  };
}

/** Añade una nota al final de su columna. */
export function anadir(datos: Datos, tarea: Tarea): Datos {
  return { tareas: [...datos.tareas, tarea] };
}

/** «hace 3 min», «ayer», «12/08». Lo que se pinta bajo el título de la ficha.
 *  Una fecha ISO en una nota de corcho no la lee nadie. */
export function haceCuanto(iso: string, ahora: Date = new Date()): string {
  const t = new Date(iso).getTime();
  if (!isFinite(t)) return '';
  const minutos = Math.floor((ahora.getTime() - t) / 60000);
  if (minutos < 1) return 'ahora mismo';
  if (minutos < 60) return `hace ${minutos} min`;
  const horas = Math.floor(minutos / 60);
  if (horas < 24) return `hace ${horas} h`;
  const dias = Math.floor(horas / 24);
  if (dias === 1) return 'ayer';
  if (dias < 7) return `hace ${dias} días`;
  const d = new Date(t);
  return `${String(d.getDate()).padStart(2, '0')}/${String(d.getMonth() + 1).padStart(2, '0')}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Lo que Perseo lee
// ─────────────────────────────────────────────────────────────────────────────

/** Días cumplidos desde una fecha ISO.
 *
 *  Cumplidos y no empezados: una nota movida hace veinte horas lleva cero días
 *  parada, que es lo que diría cualquiera. Una fecha ilegible cuenta como hoy —
 *  decir que algo lleva parado desde 1970 sería peor que no decir nada. */
export function diasDesde(iso: string, ahora: Date = new Date()): number {
  const t = new Date(iso).getTime();
  if (!isFinite(t)) return 0;
  return Math.max(0, Math.floor((ahora.getTime() - t) / 86400000));
}

/** Una nota ya masticada: sin fechas ISO que interpretar. La usan el resumen
 *  hablado y el espejo del núcleo, igual que `FotoHabito` en los hábitos. */
export type FotoTarea = {
  titulo: string;
  detalle: string;
  columna: Columna;
  /** Días cumplidos desde el último movimiento. */
  dias: number;
};

export type Foto = {
  fecha: string;
  sinHacer: number;
  enProceso: number;
  completadas: number;
  papelera: number;
  /** Las notas vivas, en el orden del tablero. La papelera no entra: lo tirado
   *  no es trabajo pendiente, y contarlo en la foto sería resucitarlo. */
  tareas: FotoTarea[];
};

/**
 * El estado del tablero en un momento dado, sin adornos.
 *
 * `ahora` entra por parámetro y no se lee del reloj aquí dentro por lo mismo
 * que en los hábitos: una función que consulta `new Date()` por su cuenta solo
 * se puede probar el día que toca.
 */
export function foto(datos: Datos, ahora: Date = new Date()): Foto {
  const vivas = datos.tareas.filter(t => t.columna !== 'papelera');
  return {
    fecha:
      `${String(ahora.getDate()).padStart(2, '0')}/` +
      `${String(ahora.getMonth() + 1).padStart(2, '0')}/${ahora.getFullYear()}`,
    sinHacer: cuenta(datos, 'sin_hacer'),
    enProceso: cuenta(datos, 'en_proceso'),
    completadas: cuenta(datos, 'completadas'),
    papelera: cuenta(datos, 'papelera'),
    tareas: vivas.map(t => ({
      titulo: t.titulo,
      detalle: t.detalle,
      columna: t.columna,
      dias: diasDesde(t.movida, ahora),
    })),
  };
}

function uno(n: number): string {
  return n === 1 ? '1 día' : `${n} días`;
}

/** Un texto largo, cortado para que quepa en una respuesta hablada. La ficha
 *  de una nota puede tener párrafos, y leerlos enteros en voz alta convierte
 *  «¿por dónde voy?» en un dictado. */
function recorte(texto: string, tope = 160): string {
  const limpio = texto.replace(/\s+/g, ' ').trim();
  if (limpio.length <= tope) return limpio;
  return limpio.slice(0, tope - 1).trimEnd() + '…';
}

/**
 * El tablero contado en castellano, para que Perseo lo lea en voz alta.
 *
 * Se devuelve texto y no JSON por lo mismo que en los hábitos: al modelo se le
 * pide que **hable** de esto, y a un modelo al que se le dan campos se le oye
 * leer campos.
 *
 * El orden es el de quien pregunta, no el de la pantalla: primero el recuento,
 * luego lo que tiene entre manos —con su detalle, que es lo que permite ayudar
 * de verdad en vez de repetir un título—, luego lo que lleva parado, después
 * los pendientes y al final lo cerrado esta semana. Lo hecho va al final pero
 * va: un tablero que solo recuerda lo que falta es una máquina de culpa.
 *
 * La papelera se cuenta y no se enumera. Lo que el señor Persus tiró no vuelve
 * a la conversación por la puerta de atrás.
 */
export function resumen(datos: Datos, ahora: Date = new Date()): string {
  const sin = deColumna(datos, 'sin_hacer');
  const proceso = deColumna(datos, 'en_proceso');
  const hechas = deColumna(datos, 'completadas');
  const enPapelera = cuenta(datos, 'papelera');
  if (!sin.length && !proceso.length && !hechas.length) {
    return 'El tablero de tareas del señor Persus está vacío.';
  }

  const dias = (t: Tarea) => diasDesde(t.movida, ahora);
  const lineas: string[] = [];
  lineas.push(
    `Tablero del señor Persus a ${foto(datos, ahora).fecha}: ${sin.length} sin hacer, ` +
    `${proceso.length} en proceso y ${hechas.length} completadas.`,
  );

  if (proceso.length) {
    lineas.push('Ahora mismo tiene entre manos:');
    for (const t of proceso) {
      const d = dias(t);
      const desde = d === 0 ? 'movida hoy' : `sin moverse desde hace ${uno(d)}`;
      const detalle = recorte(t.detalle);
      lineas.push(`- ${t.titulo} (${desde})${detalle ? `: ${detalle}` : ''}`);
    }
  }

  // Lo que lleva parado, dicho aparte y con su cifra. Enterrado en la lista de
  // arriba no lo ve nadie, y es lo único de todo esto sobre lo que se puede
  // preguntar algo útil: por qué sigue ahí.
  const atascadas = proceso.filter(t => dias(t) >= 3).sort((a, b) => dias(b) - dias(a));
  if (atascadas.length) {
    lineas.push(
      'Lleva tiempo parado: ' +
      atascadas.slice(0, 3).map(t => `${t.titulo}, ${uno(dias(t))}`).join('; ') + '.',
    );
  }

  if (sin.length) {
    lineas.push(
      'Pendientes: ' +
      sin.map(t => {
        const d = dias(t);
        return d >= 7 ? `${t.titulo} (apuntada hace ${uno(d)})` : t.titulo;
      }).join('; ') + '.',
    );
  }

  const recientes = hechas.filter(t => dias(t) <= 7);
  if (recientes.length) {
    lineas.push(
      'Cerradas en los últimos siete días: ' + recientes.map(t => t.titulo).join('; ') + '.',
    );
  }

  if (enPapelera) {
    lineas.push(
      `En la papelera hay ${enPapelera === 1 ? '1 nota' : `${enPapelera} notas`} ` +
      'que él tiró; no son trabajo pendiente.',
    );
  }

  return lineas.join('\n');
}

/** El resumen leyendo el almacén de esta ventana. Es lo que llama la
 *  herramienta de la llamada, que no tiene el estado de React a mano. */
export function resumenGuardado(ahora: Date = new Date()): string {
  return resumen(leer(), ahora);
}
