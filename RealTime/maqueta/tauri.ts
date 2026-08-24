/**
 * El `invoke` de mentira de la maqueta.
 *
 * El panel habla con el núcleo a TRAVÉS de Rust, así que en un navegador no hay
 * con quién hablar. Esto contesta lo que contestaría el núcleo un martes por la
 * tarde: cola con trabajos de todos los estados, un correo sin resolver, piezas
 * en verde y en rojo. Sirve para mirar el panel —espaciados, jerarquía, estados
 * vacíos— sin reconstruir el binario cada vez, que son dos minutos por vuelta.
 *
 * No entra en la app: `vite.maqueta.config.ts` lo pone en el sitio de
 * `@tauri-apps/api/core` solo cuando se arranca la maqueta.
 */

const ahora = Date.now();
const hace = (minutos: number) => new Date(ahora - minutos * 60_000).toISOString();

const TRABAJOS = [
  {
    id: 271, estado: 'en_curso', agente: 'dev', origen: 'panel',
    peticion: { texto: 'Añade la pestaña de informes al panel de Armario y prueba que compila' },
    progreso: 'Editando Panel.tsx',
  },
  {
    id: 270, estado: 'esperando', agente: 'pc', origen: 'voz',
    peticion: { accion: 'abrir_app', parametro: 'spotify' },
    confirmacion: { resumen: 'Abrir Spotify', detalle: 'Perseo quiere abrir Spotify y buscar «Loser».' },
  },
  {
    id: 269, estado: 'hecho', agente: 'correo', origen: 'disparador',
    peticion: { accion: 'triar' },
    resultado: { titular: '3 correos triados · 1 requiere acción', clasificados: [] },
  },
  {
    id: 268, estado: 'hecho', agente: 'memoria', origen: 'chat',
    peticion: { accion: 'buscar', texto: 'erasmus' },
    resultado: { titular: '4 notas', notas: [] },
  },
  {
    id: 267, estado: 'fallido', agente: 'mcp', origen: 'voz',
    peticion: { accion: 'llamar', servidor: 'navegador', herramienta: 'browser_navigate' },
    error: "'navegador' no arrancó: npx tardó más de 120 s",
  },
  {
    id: 266, estado: 'hecho', agente: 'agenda', origen: 'disparador',
    peticion: { accion: 'avisar' }, resultado: { titular: 'Clase de 10:00 en 30 min' },
  },
];

const ESTADO = {
  encendido_segundos: 9420,
  encendido_desde: hace(157),
  generado: new Date(ahora).toISOString(),
  router_local: true,
  version: { marca: '20260824-2312+cf7a50d', construido: new Date(ahora).toISOString() },
  agentes: ['agenda', 'chat', 'correo', 'dev', 'eco', 'mcp', 'memoria', 'pc', 'web'],
  trabajos: { hecho: 214, fallido: 9, en_curso: 1, esperando: 1 },
  cuota: {
    dia: new Date(ahora).toISOString().slice(0, 10),
    servicios: [
      { modelo: 'gemini-3.5-flash-lite', usadas: 63, tope: 500 },
      { modelo: 'gemini-2.5-flash-native-audio', usadas: 11, tope: 0 },
    ],
  },
  presencia: {
    haciendo: { id: 271, agente: 'dev' },
    esperando_un_si: 1,
    correo: { requiere_accion: 1, interesante: 2 },
    proximo_evento: { titulo: 'Tutoría TFG', momento: hace(-42), lugar: 'Despacho F1.32' },
    eventos: [
      { titulo: 'Tutoría TFG', momento: hace(-42), lugar: 'Despacho F1.32' },
      { titulo: 'Entrenamiento', momento: hace(-320), lugar: '' },
    ],
  },
  maquina: {
    disponible: true,
    cpu: 23,
    nucleos: 16,
    memoria: { porcentaje: 61, legible: '9,6 GB de 15,7 GB' },
    disco: { porcentaje: 74, legible: '351 GB de 476 GB' },
    bateria: { porcentaje: 88, enchufado: true },
    red: { legible: '↑ 41 kB/s · ↓ 320 kB/s' },
    historial: Array.from({ length: 30 }, (_, i) => ({
      cpu: 18 + Math.round(14 * Math.sin(i / 3)),
      memoria: 58 + Math.round(5 * Math.cos(i / 4)),
    })),
  },
  piezas: [
    { id: 'ollama', nombre: 'Modelo local', estado: 'ok', detalle: 'qwen3:4b listo · 6 modelos en Ollama', arreglo: '' },
    { id: 'vault', nombre: 'Memoria', estado: 'ok', detalle: 'Plugin de Obsidian en https://127.0.0.1:27124', arreglo: '' },
    { id: 'google', nombre: 'Google', estado: 'ok', detalle: 'Credenciales buenas · Gmail y Calendar', arreglo: '' },
    { id: 'suplente', nombre: 'Modelo suplente', estado: 'apagado', detalle: 'Apagado: si Ollama no está, no se clasifica nada', arreglo: 'PERSEO_MODELO_SUPLENTE=gemma-4-31b-it' },
    { id: 'telegram', nombre: 'Telegram', estado: 'ok', detalle: 'Avisos al móvil desde https://msi.taild61051.ts.net', arreglo: '' },
    { id: 'mcp', nombre: 'MCP', estado: 'ok', detalle: '6 servidores: correo, navegador, subagentes, tiempo, vault, windows', arreglo: '' },
    { id: 'dev', nombre: 'Agente dev', estado: 'ok', detalle: 'Claude por el SDK · cuenta por dónde va', arreglo: '' },
    { id: 'web', nombre: 'Web', estado: 'aviso', detalle: 'Sin clave de búsqueda: solo lee páginas dadas', arreglo: 'PERSEO_BUSQUEDA_CLAVE' },
  ],
  disparadores: [
    { nombre: 'correo', activo: true, intervalo: 300 },
    { nombre: 'agenda', activo: true, intervalo: 120 },
  ],
};

const SESIONES = [
  { id: 4, titulo: 'Preparar la tutoría', turno: 'libre', actualizado_en: hace(12) },
  { id: 3, titulo: 'Correos de la beca', turno: 'libre', actualizado_en: hace(190) },
  { id: 2, titulo: 'Ideas para la web', turno: 'libre', actualizado_en: hace(1500) },
];

const MENSAJES = [
  { id: 1, rol: 'usuario', texto: '¿Qué tengo pendiente para mañana?', herramientas: [], estado: 'hecho', momento: hace(14) },
  {
    id: 2, rol: 'perseo', estado: 'hecho', momento: hace(13),
    herramientas: ['agenda.proximos', 'correo.triados'],
    texto: 'Mañana tiene la tutoría del TFG a las 10:00 y nada más en el calendario.\n\nDel correo, uno requiere acción: la beca pide el justificante antes del viernes.',
  },
  { id: 3, rol: 'usuario', texto: 'Recuérdamelo por la mañana', herramientas: [], estado: 'hecho', momento: hace(12) },
  { id: 4, rol: 'perseo', texto: 'Apuntado. Se lo digo a las 8:30.', herramientas: ['memoria.anotar'], estado: 'hecho', momento: hace(12) },
];

const CORREOS = {
  correos: [
    { id: 'a1', asunto: 'Justificante de la beca — antes del viernes', de: 'becas@us.es', clase: 'requiere accion', motivo: 'Pide un documento con fecha límite', momento: hace(180) },
    { id: 'a2', asunto: 'Tu resumen semanal de GitHub', de: 'noreply@github.com', clase: 'interesante', motivo: 'Actividad de tus repositorios', momento: hace(400) },
  ],
};

export async function invoke(comando: string, argumentos?: any): Promise<any> {
  await new Promise(r => setTimeout(r, 120));
  switch (comando) {
    case 'panel_estado': return ESTADO;
    case 'panel_trabajos': return { trabajos: TRABAJOS };
    case 'panel_trabajo': return TRABAJOS.find(t => t.id === argumentos?.id) ?? TRABAJOS[0];
    case 'panel_encolar': return { ...TRABAJOS[0], id: 999, estado: 'hecho' };
    case 'panel_correos': return CORREOS;
    case 'chat_sesiones': return { sesiones: SESIONES };
    case 'chat_sesion': return { ...SESIONES[0], mensajes: MENSAJES };
    case 'chat_crear': return { id: 5, titulo: 'Nueva', turno: 'libre', actualizado_en: hace(0) };
    default: return {};
  }
}
