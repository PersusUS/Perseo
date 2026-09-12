# Perseo Live — la cara de voz

Esta carpeta es **una de las caras de Perseo**, no el asistente entero. Habla con Gemini
Live por voz, mira la cámara y la pantalla, y cuando hace falta hacer algo, **encola un
trabajo en el núcleo** (`perseo_core/`) y espera el resultado.

**La regla que gobierna esta carpeta: las caras no piensan.** Aquí no hay memoria, ni
herramientas, ni decisiones. Hasta agosto de 2026 sí las había —la app arrancaba
`TOOLS/server.py` y le hablaba por tuberías—, y eso significaba dos sistemas con dos
memorias: lo que pedías por voz no existía para la web del móvil. Ya no.

Construida con React 19, TypeScript y Vite, empaquetada con Tauri 2.

---

## Qué hace

- **Voz en las dos direcciones**, con transcripción real de ambos lados en el overlay
  (`inputAudioTranscription` y `outputAudioTranscription`).
- **Cámara y pantalla** en directo, para que el modelo vea de qué le hablas.
- **Vive en la bandeja del sistema.** Cerrar la ventana la esconde; se sale desde el menú
  del icono. El detector de aplausos la despierta sin arrancarla, que es lo que quita los
  diez segundos de espera.
- **Sesión reanudable**: si la conexión se cae, se retoma con el testigo guardado en vez
  de empezar de cero.
- **Audio proactivo**: puede callarse ante lo que no va con él, que hace falta con un
  micrófono siempre cerca.
- **Manos libres o pulsar para hablar**: en un sitio con ruido la detección automática
  toma cualquier voz de fondo por una orden, así que en Ajustes se puede cerrar el
  micrófono y abrirlo solo mientras se mantiene el botón —o la barra espaciadora—. El
  modo viaja en el `setup` del socket, así que cambiarlo entra en la llamada siguiente.
- **Ajustes que se quedan**: clave, voz, modo de micrófono, instrucciones del sistema y
  si se guardan las conversaciones. Se persisten en el almacén local que gestiona Rust.

## Cómo habla con el núcleo

`src-tauri/src/nucleo.rs` es el único sitio que sabe de los dos lados. El modelo declara
cuatro herramientas y `traducir()` las convierte en trabajos para los agentes:

| Lo que ve el modelo | A dónde va |
|---|---|
| `consultar_base_vectorial` | Agente `memoria`, acción `buscar` |
| `guardar_recuerdo` | Agente `memoria`, acción `anotar` |
| `guardar_conversacion` | Agente `memoria`, acción `conversacion` |
| `controlar_pc` | Agente `pc` |

Tres cosas que conviene no deshacer:

- **El token no pasa por el frontend.** Lo lee Rust de `perseo_core/datos/token.txt`, o de
  `PERSEO_TOKEN`. El navegador embebido nunca lo ve.
- **Un trabajo que tarda no bloquea la conversación.** A los 30 segundos se contesta
  "sigue en marcha" y el trabajo continúa en la cola: colgar la llamada no lo mata.
- **Si el agente pide confirmación, aquí no se espera.** Se devuelve la pregunta y el
  número de trabajo; el sí se da desde la web o desde Telegram.

Las herramientas se declaran `NON_BLOCKING`, así que Perseo sigue hablando mientras el
trabajo corre y te cuenta el resultado cuando calla.

---

## Requisitos

- **Node.js 18+**
- **Rust y las herramientas de compilación** que pide Tauri
  ([prerrequisitos](https://tauri.app/start/prerequisites/))
- **Clave de Gemini** con acceso a `gemini-2.5-flash-native-audio-latest` por la Live API
  (`bidiGenerateContent`). Se comprueba con `node probar_live.mjs` — ojo, cada ejecución
  gasta una conexión de la cuota diaria.

  La clave se mete **desde la propia aplicación** (botón ⚙) y la guarda Rust. **No** en un
  `.env` con prefijo `VITE_`: Vite incrusta esas variables dentro del JavaScript
  compilado, así que la clave acababa en claro dentro del `.exe`.

## Arrancar

**El núcleo primero**, o la voz se queda sin memoria y sin manos:

```bash
python -m perseo_core          # desde la raíz del repositorio
```

Y luego la app:

```bash
npm install
npm run tauri dev
```

Para compilar: `npm run tauri build`.

`PERSEO_CORE_URL` cambia dónde busca el núcleo; por defecto, `http://127.0.0.1:8787`.

---

## Por dentro

| Fichero | Qué hace |
|---|---|
| `src/lib/llamada/gemini-live.ts` | La sesión con Gemini: conexión, reanudación, transcripciones y llamadas a herramientas |
| `src/lib/audio/audio-manager.ts` | El micrófono: captura y encolado |
| `src/lib/audio/audio-player.ts` | La reproducción. **No juntar los dos `AudioContext`** (16 y 24 kHz): está en la lista de intocables |
| `src/lib/llamada/camera-manager.ts`, `screen-manager.ts` | Los fotogramas de cámara y pantalla |
| `src/App.tsx` | Orquesta la interfaz, la transcripción y la autollamada |
| `src-tauri/src/nucleo.rs` | El cliente del núcleo |
| `src-tauri/src/bandeja.rs` | El icono de la bandeja y el esconder en vez de cerrar |
| `src-tauri/src/autollamada.rs` | Vigila el marcador que deja el detector de aplausos |

## Verificación

Desde la raíz del repositorio:

```bash
npx tsc --noEmit          # dentro de RealTime/
npm run build             # dentro de RealTime/
cargo check               # dentro de RealTime/src-tauri/
```

Las tres las corre GitHub en cada push, junto con las pruebas del núcleo. Ver
`.github/workflows/verificacion.yml`.
