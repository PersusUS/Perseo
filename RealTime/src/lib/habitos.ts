/**
 * El dato de los hábitos, fuera de la pantalla que lo dibuja.
 *
 * Hasta el 2026-08-25 todo esto vivía dentro de `components/Habitos.tsx`: el
 * tipo, el almacén, las cuentas y la pantalla, juntos. Mientras el único que
 * leía los hábitos era la propia pantalla, daba igual. Dejó de dar igual en
 * cuanto **Perseo** tuvo que poder consultarlos en llamada: `gemini-live.ts` no
 * puede importar un componente de React para averiguar cuántos días seguidos
 * llevas yendo al gimnasio.
 *
 * Así que el dato vive aquí y la pantalla es una de sus dos lectoras. La otra
 * es la herramienta `consultar_habitos`, que llama a `resumen()` y le entrega
 * al modelo un texto en castellano —no un JSON—: lo que se le pide a Perseo es
 * que lo cuente hablando, y un modelo al que se le da prosa contesta prosa.
 *
 * Dónde está guardado: en el `localStorage` de la ventana, bajo `ALMACEN`. Es
 * la excepción consciente a «las caras no piensan» —el núcleo no tiene agente
 * de hábitos y esto no encola trabajo—, y por eso mismo el núcleo recibe un
 * **espejo** de solo lectura (ver `espejar()`): así el chat escrito y los
 * agentes ven lo mismo que la llamada, sin que haya dos sitios donde marcar.
 */

/** La clave del almacén. Fuera de la pantalla porque el espejo también la usa. */
export const ALMACEN = 'perseo.habitos.v1';

export type Habito = { id: string; nombre: string };

/** Un mes de datos. Las marcas van en un objeto plano con clave
 *  `idHabito|día` en vez de una matriz: añadir o quitar un hábito no tiene
 *  entonces que recolocar nada, y un hábito borrado se lleva sus marcas al
 *  filtrarlas por id. */
export type Mes = {
  marcas: Record<string, boolean>;
  animo: Record<string, number>;
  motivacion: Record<string, number>;
};

export type Datos = { habitos: Habito[]; meses: Record<string, Mes> };

export const MESES = [
  'enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
  'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre',
];

/** Los rótulos del encabezado, empezando en domingo porque `getDay()` devuelve
 *  0 para el domingo y así el índice es el propio día de la semana. */
export const DIAS_SEMANA = ['D', 'L', 'M', 'X', 'J', 'V', 'S'];
export const DIAS_SEMANA_LARGO = [
  'domingo', 'lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado',
];

/** Los doce de la plantilla, traducidos. Es la lista con la que se estrena la
 *  pantalla; a partir de ahí manda lo que haya guardado. */
const HABITOS_INICIALES: Habito[] = [
  { id: 'h1', nombre: 'Levantarse a las 06:00' },
  { id: 'h2', nombre: 'Meditar' },
  { id: 'h3', nombre: 'Gimnasio' },
  { id: 'h4', nombre: 'Ducha fría' },
  { id: 'h5', nombre: 'Trabajo' },
  { id: 'h6', nombre: 'Leer 10 páginas' },
  { id: 'h7', nombre: 'Aprender algo nuevo' },
  { id: 'h8', nombre: 'Sin azúcar' },
  { id: 'h9', nombre: 'Sin alcohol' },
  { id: 'h10', nombre: 'Una hora de redes' },
  { id: 'h11', nombre: 'Planificar el día' },
  { id: 'h12', nombre: 'Dormir antes de las 23:00' },
];

export const MES_VACIO: Mes = { marcas: {}, animo: {}, motivacion: {} };

export function clave(anio: number, mes: number): string {
  return `${anio}-${String(mes + 1).padStart(2, '0')}`;
}

export function diasDelMes(anio: number, mes: number): number {
  return new Date(anio, mes + 1, 0).getDate();
}

/** Lee lo guardado, y si no hay nada —o hay algo roto— arranca con la lista de
 *  fábrica en vez de dejar la pantalla en blanco.
 *
 *  Los hábitos se normalizan de paso: hasta el 2026-08-25 llevaban un campo
 *  `icono` con un emoji, y lo guardado de entonces sigue en el disco. Se cae
 *  aquí, en la puerta, y no en cada sitio donde se dibuja un hábito. */
export function leer(): Datos {
  try {
    const crudo = localStorage.getItem(ALMACEN);
    if (crudo) {
      const d = JSON.parse(crudo) as { habitos?: any[]; meses?: Record<string, Mes> };
      if (Array.isArray(d.habitos) && d.meses) {
        return {
          habitos: d.habitos.map(h => ({ id: String(h.id), nombre: String(h.nombre ?? '') })),
          meses: d.meses,
        };
      }
    }
  } catch {
    // Un almacén ilegible se sustituye; avisar de esto no le sirve a nadie.
  }
  return { habitos: HABITOS_INICIALES, meses: {} };
}

/** Guarda entero. Son unos pocos kilobytes y el mes cabe en una escritura:
 *  llevar un diario de cambios aquí sería fontanería sin cliente. */
export function guardar(datos: Datos): void {
  try {
    localStorage.setItem(ALMACEN, JSON.stringify(datos));
  } catch {
    // Almacén lleno o bloqueado: la pantalla sigue funcionando en memoria.
  }
}

/** Un mes sin nada marcado da 0 en vez de `NaN`. */
export function porcentaje(parte: number, total: number): number {
  if (!total) return 0;
  return (parte / total) * 100;
}

/** Las semanas son bloques de siete días desde el 1, no semanas naturales: es
 *  como están agrupadas las columnas de la rejilla, y la gráfica de la derecha
 *  tiene que contar lo mismo que se ve debajo. */
export function semanaDe(dia: number): number {
  return Math.ceil(dia / 7);
}

export function acotar(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

/**
 * Días seguidos marcados que llegan hasta `(anio, mes, hasta)`, contando hacia
 * atrás y **cruzando el cambio de mes**.
 *
 * Es la cifra que se mira de verdad en un seguimiento de hábitos: no cuántas
 * veces lo hiciste este mes, sino si la cadena sigue viva. Justo por eso no
 * puede pararse el día 1: una racha contada dentro del mes valía como mucho 1
 * cada primero de mes, y ver el contador de cuarenta días caer a uno por girar
 * la hoja del calendario es exactamente el motivo por el que la gente abandona
 * un tracker. Se retrocede mes a mes mientras el día anterior siga marcado.
 *
 * El tope de doce meses no es una elección de precisión: es el seguro de que
 * esto termina aunque alguien guarde un almacén con marcas raras. Una racha de
 * un año larga se cuenta como «más de un año» sin necesidad de recorrer 2019.
 */
export function racha(
  datos: Datos, id: string, anio: number, mes: number, hasta: number,
): number {
  let n = 0;
  let a = anio;
  let m = mes;
  let d = hasta;

  for (let saltos = 0; saltos <= 12; ) {
    const marcas = (datos.meses[clave(a, m)] ?? MES_VACIO).marcas;
    while (d >= 1) {
      if (!marcas[`${id}|${d}`]) return n;
      n++;
      d--;
    }
    // Agotado el mes por arriba, se sigue por el último día del anterior.
    const previo = new Date(a, m - 1, 1);
    a = previo.getFullYear();
    m = previo.getMonth();
    d = diasDelMes(a, m);
    saltos++;
  }
  return n;
}

// ─────────────────────────────────────────────────────────────────────────────
// Lo que Perseo lee
// ─────────────────────────────────────────────────────────────────────────────

/** Una foto de un hábito en un mes, ya masticada: sin porcentajes que calcular
 *  ni claves que componer. La usan el resumen hablado y el espejo del núcleo. */
export type FotoHabito = {
  nombre: string;
  hechos: number;
  de: number;
  racha: number;
  /** Si hoy está marcado. `null` fuera del mes en curso, donde «hoy» no cae. */
  hoy: boolean | null;
};

export type Foto = {
  fecha: string;
  mes: string;
  hechos: number;
  objetivo: number;
  porcentaje: number;
  habitos: FotoHabito[];
  animo: number | null;
  motivacion: number | null;
};

/** Media de una serie del estado mental sobre los días que tienen número.
 *  Los días en blanco no cuentan como cero: no haber apuntado el ánimo no es
 *  tener el ánimo por los suelos. */
function media(fuente: Record<string, number>, hasta: number): number | null {
  let suma = 0;
  let n = 0;
  for (let d = 1; d <= hasta; d++) {
    const v = fuente[d];
    if (typeof v === 'number') { suma += v; n++; }
  }
  return n ? suma / n : null;
}

/**
 * El estado de los hábitos en un momento dado, sin adornos.
 *
 * `ahora` entra por parámetro y no se lee del reloj aquí dentro para que las
 * pruebas puedan fijar el día: una función que consulta `new Date()` por su
 * cuenta solo se puede probar el día que toca.
 */
export function foto(datos: Datos, ahora: Date = new Date()): Foto {
  const anio = ahora.getFullYear();
  const mes = ahora.getMonth();
  const dia = ahora.getDate();
  const m = datos.meses[clave(anio, mes)] ?? MES_VACIO;

  const habitos: FotoHabito[] = datos.habitos.map(h => {
    let hechos = 0;
    for (let d = 1; d <= dia; d++) if (m.marcas[`${h.id}|${d}`]) hechos++;
    return {
      nombre: h.nombre,
      hechos,
      de: dia,
      racha: racha(datos, h.id, anio, mes, dia),
      hoy: !!m.marcas[`${h.id}|${dia}`],
    };
  });

  const hechos = habitos.reduce((s, h) => s + h.hechos, 0);
  const objetivo = datos.habitos.length * dia;

  return {
    fecha: `${String(dia).padStart(2, '0')}/${String(mes + 1).padStart(2, '0')}/${anio}`,
    mes: `${MESES[mes]} de ${anio}`,
    hechos,
    objetivo,
    porcentaje: Math.round(porcentaje(hechos, objetivo)),
    habitos,
    animo: media(m.animo, dia),
    motivacion: media(m.motivacion, dia),
  };
}

function uno(n: number): string {
  return n === 1 ? '1 día' : `${n} días`;
}

/**
 * La foto, contada en castellano para que Perseo la lea en voz alta.
 *
 * Se devuelve texto y no JSON a propósito. El modelo tiene que **hablar** de
 * esto: dándole un objeto se dedica a leer campos («hechos: 14, de: 25»), y
 * dándole prosa la resume como lo que es, la respuesta a «¿cómo voy este mes?».
 *
 * El orden no es el de la pantalla sino el que le importa a quien pregunta:
 * primero lo global, luego lo que hoy sigue sin hacer —que es lo accionable a
 * las once de la noche—, luego las rachas vivas —lo que hay que no romper— y
 * al final la lista entera por si preguntan por uno concreto.
 */
export function resumen(datos: Datos, ahora: Date = new Date()): string {
  const f = foto(datos, ahora);
  if (!f.habitos.length) {
    return 'El señor Persus no tiene ningún hábito dado de alta en la pantalla de hábitos.';
  }

  const lineas: string[] = [];
  lineas.push(
    `Hábitos del señor Persus a ${f.fecha} (${f.mes}): ${f.hechos} casillas de ` +
    `${f.objetivo} posibles en lo que va de mes, un ${f.porcentaje} %.`,
  );

  const pendientesHoy = f.habitos.filter(h => h.hoy === false).map(h => h.nombre);
  if (!pendientesHoy.length) {
    lineas.push('Hoy los lleva todos hechos.');
  } else if (pendientesHoy.length === f.habitos.length) {
    lineas.push('Hoy todavía no ha marcado ninguno.');
  } else {
    lineas.push(`Hoy le faltan por hacer: ${pendientesHoy.join(', ')}.`);
  }

  const vivas = f.habitos
    .filter(h => h.racha >= 3)
    .sort((a, b) => b.racha - a.racha)
    .slice(0, 5);
  if (vivas.length) {
    lineas.push(
      'Rachas vivas: ' +
      vivas.map(h => `${h.nombre}, ${uno(h.racha)} seguidos`).join('; ') + '.',
    );
  }

  // Lo peor del mes, dicho sin dramatismo: es lo que uno quiere que le
  // recuerden, y esconderlo entre la lista completa lo deja sin efecto.
  const flojos = [...f.habitos].sort((a, b) => a.hechos - b.hechos).slice(0, 3)
    .filter(h => h.hechos < f.habitos[0].de / 2);
  if (flojos.length) {
    lineas.push(
      'Los que peor van: ' +
      flojos.map(h => `${h.nombre}, ${h.hechos} de ${h.de}`).join('; ') + '.',
    );
  }

  lineas.push(
    'Detalle del mes: ' +
    f.habitos.map(h => `${h.nombre} ${h.hechos}/${h.de} (racha ${h.racha})`).join('; ') + '.',
  );

  const mental: string[] = [];
  if (f.animo !== null) mental.push(`ánimo medio ${f.animo.toFixed(1)} sobre 10`);
  if (f.motivacion !== null) mental.push(`motivación media ${f.motivacion.toFixed(1)} sobre 10`);
  if (mental.length) lineas.push(`Estado mental del mes: ${mental.join(', ')}.`);

  return lineas.join('\n');
}

/** El resumen leyendo el almacén de esta ventana. Es lo que llama la
 *  herramienta de la llamada, que no tiene el estado de React a mano. */
export function resumenGuardado(ahora: Date = new Date()): string {
  return resumen(leer(), ahora);
}
