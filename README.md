# Perseo AI Ecosystem 🧠

Perseo es una Inteligencia Artificial "personalizada", un sistema completo diseñado para ofrecer una experiencia avanzada de asistente por voz, videollamada y control de background ininterrumpido.
Ha sido construido modularmente para mantener varias facetas: una app nativa inmersiva, un cerebro de RAG para conocimiento integrado y comandos fantasmas de escucha persistente.

## 🗂️ Arquitectura del Proyecto

El repositorio está dividido en 4 pilares tecnológicos principales:

### 1. `RealTime/` (Perseo Live - UI Nativa)
Esquíema Frontend y Motor principal del Asistente (La Cara y Voz de Perseo).
*   **Stack:** Tauri v2 (Rust) + React 19 + TypeScript + Vite.
*   **Funciones:**
    *   Conexión de Streaming WebSocket (Gemini Flash 3.1 Live Preview).
    *   Screen capture y Camera Sharing para que la IA perciba tu entorno.
    *   Autogestión de micrófono y reconexión automática anti-caídas.
    *   Interfaz transparente `borderless` o Fullscreen estilo HUD avanzado.

### 2. `commands/` (Listeners Fantasma de Background)
Controladores silenciosos y gatilladores persistentes escritos en Python.
*   **Stack:** Python 3 (PyGame, SoundDevice, SpeechRecognition).
*   **`clap_detector.py`:** Un recolector de ondas a $44.1$ kHz que consume el 0.01% de tu CPU. Detecta dos aplausos e inicia una confirmación por IA *(espera oír tu voz diciendo "Perseo" antes de encender la aplicación RealTime).*
*   **`manage_startup.py`:** Editor del registro (HKCU) de Windows para que el `clap_detector` nazca en segundo plano invisiblemente cada vez que enciendas tu PC.
*   **`loading_splash.py`:** UI de carga en tkinter minimalista mientras el `npm run tauri dev` se está despertando en el background. 

### 3. `RAG/` & `TOOLS/` (El Cerebro)
Infraestructura en Python para la base de conocimientos y automatización *Retrival-Augmented Generation* de la IA. *(Aún en construcción/ampliación).*

---

## 🚀 Requisitos Previos

*   **Node.js v18+** y herramientas de red como npm.
*   **Rust & CLI Build Tools** (para la compilación de `RealTime`).
*   **Python 3.10+** (para `commands` e IA local).
*   **Clave de Gemini**: se introduce **una sola vez desde la propia aplicación** (botón ⚙) y queda guardada en el almacén local que gestiona Rust. Alternativamente, puedes definirla como variable de entorno del sistema con `setx GEMINI_API_KEY "AI..."`.

    > No uses `VITE_GEMINI_API_KEY` en un `.env`: Vite incrusta las variables con ese prefijo dentro del JavaScript compilado, de modo que la clave acababa en claro dentro del `.exe`. Ver `bitacora/02_HALLAZGOS.md` (H-17).

## ⚙️ Cómo Poner a Perseo en Marcha

1.  **Frontend / Interfaz:**
    Ve a la carpeta `/RealTime`.
    ```bash
    npm install
    npm run tauri dev
    ```
2.  **Listener en Background (Escucha de Aplausos):**
    Ve a la carpeta `/commands` e instala los recursos.
    ```bash
    pip install -r requirements.txt
    python manage_startup.py
    ```
    *(Nota: Debes tener o ubicar tu archivo `opening.mp3` en la propia carpeta commands si quieres ambientación en el encendido de emergencia)*

## 🛡️ Privacidad (.gitignore)

El sistema ha sido purgado por Git, asegurando que tus claves de Gemini (`.env`), tus compilaciones locales (`node_modules/`, `target/`), audios privados (`*.mp3`) y todos los datos en crudo como credenciales nunca subirán al nube.

## 👨‍💻 Creado por:
**Jesús Pérez Bazarot ("Señor Persus")**
Mantenido por Perseo.