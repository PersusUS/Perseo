// Reproduce la configuración de `gemini-live.ts` en **una sola conexión**, manda
// audio PCM de 16 kHz como hace la app y anota lo que pasa con marca de tiempo.
// Sirve para separar "la API no acepta lo que mandamos" de "el cliente se porta
// mal", que es lo que costó una tarde el 2026-08-17. Se ejecuta a mano
// desde `RealTime/`, y no forma parte de la app:
//
//     node sonda_llamada.mjs [sin-audio] [sin-herramientas]
//
// Sin argumentos manda audio y declara las cuatro herramientas, que es lo que
// hace una llamada de verdad. Una conexión por ejecución: no lo pongas en bucle.

import { readFileSync } from 'node:fs';
import { Behavior, GoogleGenAI, Modality, Type } from '@google/genai';

const clave = readFileSync('../perseo_core/datos/gemini.txt', 'utf-8').trim();
const sinAudio = process.argv.includes('sin-audio');
const sinHerramientas = process.argv.includes('sin-herramientas');

// El prompt de verdad de la app, sacado de config.ts tal cual.
const fuenteConfig = readFileSync('src/lib/config.ts', 'utf-8');
const promptDelSistema = fuenteConfig.split('systemPrompt: `')[1].split('`,')[0];

const t0 = Date.now();
const log = (...a) => console.log(`[+${((Date.now() - t0) / 1000).toFixed(1)}s]`, ...a);

const ai = new GoogleGenAI({ apiKey: clave, apiVersion: 'v1alpha' });

const herramientas = [{
  functionDeclarations: [
    {
      name: 'buscar_en_memoria',
      behavior: Behavior.NON_BLOCKING,
      description: 'Busca en las notas del vault de Obsidian del señor Persus.',
      parameters: { type: Type.OBJECT, properties: { texto: { type: Type.STRING, description: 'Palabras a buscar.' } }, required: ['texto'] },
    },
    {
      name: 'leer_nota',
      behavior: Behavior.NON_BLOCKING,
      description: 'Lee una nota entera del vault y devuelve su texto.',
      parameters: { type: Type.OBJECT, properties: { ruta: { type: Type.STRING, description: 'Ruta de la nota.' } }, required: ['ruta'] },
    },
    {
      name: 'guardar_recuerdo',
      behavior: Behavior.NON_BLOCKING,
      description: 'Guarda una nota nueva en el vault de Obsidian.',
      parameters: {
        type: Type.OBJECT,
        properties: {
          entidad: { type: Type.STRING, description: 'Nombre.' },
          descripcion_visual: { type: Type.STRING, description: 'Descripción visual.' },
          contexto: { type: Type.STRING, description: 'Contexto.' },
        },
        required: ['entidad', 'descripcion_visual', 'contexto'],
      },
    },
    {
      name: 'controlar_pc',
      behavior: Behavior.NON_BLOCKING,
      description: 'Permite usar la computadora local del usuario (Windows).',
      parameters: {
        type: Type.OBJECT,
        properties: {
          accion: { type: Type.STRING, description: 'La acción a realizar.' },
          parametro: { type: Type.STRING, description: 'El parámetro de la acción.' },
        },
        required: ['accion', 'parametro'],
      },
    },
  ],
}];

const config = {
  responseModalities: [Modality.AUDIO],
  proactivity: { proactiveAudio: true },
  sessionResumption: { handle: undefined },
  inputAudioTranscription: {},
  outputAudioTranscription: {},
  ...(sinHerramientas ? {} : { tools: herramientas }),
  speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Orus' } } },
  systemInstruction: { parts: [{ text: promptDelSistema }] },
  thinkingConfig: { thinkingLevel: 'MINIMAL' },
};

log(`conectando (audio=${!sinAudio}, herramientas=${!sinHerramientas})`);

let temporizador = null;
const sesion = await ai.live.connect({
  model: 'gemini-2.5-flash-native-audio-latest',
  config,
  callbacks: {
    onopen: () => log('socket abierto'),
    onmessage: (m) => {
      const claves = Object.keys(m).filter((k) => m[k] !== undefined);
      if (m.serverContent?.modelTurn?.parts?.some((p) => p.inlineData)) log('audio del modelo');
      else log('mensaje:', claves.join(','), m.setupComplete ? '(setupComplete)' : '');
    },
    onerror: (e) => log('ERROR:', e?.message ?? String(e)),
    onclose: (e) => {
      log(`CERRADO code=${e?.code} reason=${e?.reason}`);
      if (temporizador) clearInterval(temporizador);
      process.exit(0);
    },
  },
});

log('connect() resuelto');

if (!sinAudio) {
  // 100 ms de silencio PCM 16-bit a 16 kHz = 1600 muestras = 3200 bytes,
  // el mismo tamaño y ritmo que manda el worklet de la app.
  const trozo = Buffer.alloc(3200).toString('base64');
  temporizador = setInterval(() => {
    try {
      sesion.sendRealtimeInput({ audio: { mimeType: 'audio/pcm;rate=16000', data: trozo } });
    } catch (e) {
      log('fallo mandando audio:', e?.message ?? String(e));
    }
  }, 100);
  log('mandando audio cada 100 ms');
}

setTimeout(() => {
  log('sobrevivió 40 s sin cerrarse');
  if (temporizador) clearInterval(temporizador);
  try { sesion.close(); } catch {}
  process.exit(0);
}, 40000);
