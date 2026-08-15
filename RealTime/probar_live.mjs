// Comprobación de un solo disparo: ¿acepta el modelo 2.5 la configuración que
// necesita la Fase C? Conecta, espera a que el servidor confirme el setup y
// cierra. Se ejecuta a mano, no forma parte de la app:
//
//     node probar_live.mjs
//
// Gasta una conexión de la cuota diaria. No lo dejes en un bucle.

import { readFileSync } from 'node:fs';
import { GoogleGenAI, Behavior, Modality, Type } from '@google/genai';

const clave = readFileSync('.env', 'utf-8')
  .split('\n')
  .find((l) => l.startsWith('VITE_GEMINI_API_KEY='))
  ?.split('=')[1]
  ?.trim();

if (!clave) {
  console.error('No hay VITE_GEMINI_API_KEY en RealTime/.env');
  process.exit(1);
}

const modelo = process.argv[2] ?? 'gemini-2.5-flash-live-preview';
const version = process.argv[3] ?? 'v1beta';
const conProactividad = process.argv[4] !== 'sin-proactividad';
const ai = new GoogleGenAI({ apiKey: clave, apiVersion: version });

console.log(`Probando ${modelo} (${version}, proactividad=${conProactividad})…`);

const terminado = new Promise((resolve) => {
  let resuelto = false;
  const acabar = (estado, detalle) => {
    if (resuelto) return;
    resuelto = true;
    resolve({ estado, detalle });
  };

  ai.live
    .connect({
      model: modelo,
      config: {
        responseModalities: [Modality.AUDIO],
        inputAudioTranscription: {},
        outputAudioTranscription: {},
        // Lo que hay que confirmar: llamadas asíncronas, audio proactivo y
        // sesión reanudable, que son las tres patas de la Fase C.
        ...(conProactividad ? { proactivity: { proactiveAudio: true } } : {}),
        sessionResumption: {},
        tools: [
          {
            functionDeclarations: [
              {
                name: 'tarea_larga',
                behavior: Behavior.NON_BLOCKING,
                description: 'Prueba de llamada asíncrona.',
                parameters: {
                  type: Type.OBJECT,
                  properties: { texto: { type: Type.STRING } },
                  required: ['texto'],
                },
              },
            ],
          },
        ],
        speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Charon' } } },
        systemInstruction: { parts: [{ text: 'Responde con una sola palabra.' }] },
        thinkingConfig: { thinkingLevel: 'MINIMAL' },
      },
      callbacks: {
        onopen: () => console.log('  socket abierto'),
        onmessage: (m) => {
          if (m.setupComplete) acabar('ok', 'setupComplete recibido');
          if (m.sessionResumptionUpdate) {
            console.log('  sessionResumptionUpdate:', JSON.stringify(m.sessionResumptionUpdate));
          }
        },
        onerror: (e) => acabar('error', e?.message ?? String(e)),
        onclose: (e) => acabar('cerrado', `${e?.code ?? ''} ${e?.reason ?? ''}`.trim()),
      },
    })
    .then((sesion) => setTimeout(() => { try { sesion.close(); } catch {} }, 6000))
    .catch((e) => acabar('error', e?.message ?? String(e)));

  setTimeout(() => acabar('sin respuesta', 'agotados 20 s'), 20000);
});

const { estado, detalle } = await terminado;
console.log(`\n[${estado}] ${detalle}`);
process.exit(estado === 'ok' ? 0 : 1);
