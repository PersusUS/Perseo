# Perseo Live (RealTime)

**Perseo Live** es un cliente interactivo en tiempo real que permite establecer una comunicación de voz, video y pantalla compartida directamente con el modelo de IA **Gemini**. 

Está construido utilizando **React.js**, **Vite**, **TypeScript** y encapsulado como una aplicación de escritorio nativa mediante **Tauri**.

## 🚀 Funcionalidades Principales

- 🎙️ **Comunicación Bidireccional por Voz**: Transcripción y respuesta por audio en tiempo real usando `@google/genai` (Gemini Live).
- 📸 **Streaming de Cámara**: Posibilidad de compartir la cámara local para que el modelo pueda analizar lo que ves.
- 🖥️ **Captura y Compartición de Pantalla**: Envío de la pantalla entera o aplicaciones específicas en tiempo real para que el modelo interactúe sobre el contexto mostrado.
- 🤖 **PerseoFace Avatar**: Representación visual y animada del asistente en base al nivel de volumen detectado y los estados de la conexión.
- ⚙️ **Configuraciones en Tiempo Real**: Cambio directo desde la UI de la API Key, el System Prompt y la Voz sintetizada del modelo.

## 🛠️ Tecnologías Utilizadas

- **Frontend Core**: React 19, TypeScript, Vite.
- **Aplicación de Escritorio**: Tauri v2, Rust (en el directorio `src-tauri`).
- **Integración con IA**: Protocolo WebSocket oficial a través del Google Gemini SDK (`@google/genai`).

## 📦 Requisitos Previos

Para desarrollar y compilar este proyecto, necesitas lo siguiente en tu entorno local:
- **Node.js** (v18 o superior)
- **Rust y herramientas C/C++** correspondientes (Requerido por Tauri). [Guía de pre-requisitos de Tauri](https://tauri.app/v1/guides/getting-started/prerequisites).
- **Google GenAI API Key**: Necesitas una clave con acceso al modelo
  `gemini-2.5-flash-native-audio-latest` por la Live API (`bidiGenerateContent`).
  Para comprobar que tu clave lo tiene: `node probar_live.mjs`.

## ⚙️ Instalación y Uso

1. **Instalar dependencias del proyecto:**
   Navega a la carpeta de `RealTime` en tu terminal y ejecuta:
   ```bash
   npm install
   ```

2. **Ejecutar en modo Web (solo UI, sin permisos nativos completos):**
   ```bash
   npm run dev
   ```

3. **Ejecutar la App Nativa de Escritorio (Modo recomendado):**
   ```bash
   npm run tauri dev
   ```
   *Esto levantará el frontend con Vite y simultáneamente abrirá la ventana nativa de Tauri.*

4. **Kits para Producción (Build Final):**
   ```bash
   npm run tauri build
   ```

## 🧠 Arquitectura de la Carpeta

El núcleo lógico de la aplicación reside en la carpeta `src/lib/`, que está altamente modularizado:

- `gemini-live.ts`: Instancia global del cliente que gestiona el WebSocket con la API de Google, incluyendo la gestión de sesión, manejo de pausas, recepciones de audio continuo (`audioPlayer`) e ingesta de video/fotos.
- `camera-manager.ts` y `screen-manager.ts`: Obtienen y transforman los *frames* de Video/Pantalla local en una codificación amigable para el envío de datos multimodales al modelo.
- `audio-manager.ts`: Controlador del micrófono; realiza la captura a la frecuencia deseada y encola los paquetes para emisión.
- `App.tsx`: Orquesta el UI, enlazando eventos del modelo a la visualización gráfica (componente `<PerseoFace />`), manejando la transcripción de la conversación y controlando la barra de estado.

## 🤝 Primeros pasos

Al abrir la aplicación por primera vez, pulsa sobre el botón del engranaje (⚙️) para insertar tu **API Key** y configurar el System Prompt inicial si lo requieres.
