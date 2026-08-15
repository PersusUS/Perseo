# Perseo 🧠

Asistente personal con un núcleo siempre encendido y varias caras: voz, texto en el PC,
texto en el móvil y avisos al bolsillo.

La versión anterior era un asistente **reactivo**: abrías la app, hablabas, respondía, y
un solo modelo hacía todo dentro de una sesión. Perseo v2 es **un núcleo y varias caras**.
El núcleo tiene cola de trabajo y memoria propias; las caras son clientes intercambiables.

**Regla que no se rompe: las caras no piensan.** La app de escritorio, la web del móvil y
la sesión de voz transportan entrada y salida; toda decisión ocurre en el núcleo. Es lo
que permitirá mudarlo a una Raspberry Pi sin reescribir nada.

---

## Arquitectura

```
   iPhone (PWA)              PC: app Tauri (bandeja)
        │ Tailscale               │            │
        │                    texto│        voz │
        ▼                         ▼            ▼
  ┌──────────────────────────────────────────────────┐
  │              perseo_core (Python)                │
  │                                                  │
  │  HTTP + SSE     bus de eventos    cola (SQLite)  │
  │                                                  │
  │  ┌──────────── router local ─────────────────┐   │
  │  │  Qwen3 4B + gramática → decide destino    │   │
  │  └──┬────────────────────────────────────┬───┘   │
  │     │ responde ya          encola (lento)│       │
  │     ▼                                    ▼       │
  │                          ┌───────────────────┐   │
  │                          │ correo   agenda   │   │
  │                          │ memoria  dev      │   │
  │                          │ pc       web      │   │
  │                          └───────────────────┘   │
  │                                                  │
  │  política de confirmación por niveles (§7)       │
  └──────┬───────────────────────────────────┬───────┘
         │                                   │
  ┌──────▼───────┐                  ┌────────▼────────┐
  │ Gemini Live  │                  │ obsidian_vault/ │
  │ (voz)        │                  │ (memoria)       │
  └──────────────┘                  └─────────────────┘
```

| Carpeta | Qué es |
|---|---|
| `perseo_core/` | El núcleo: cola, bus, API, router local, agentes y política |
| `RealTime/` | La app de escritorio (Tauri 2 + React + TypeScript): voz y pantalla |
| `commands/` | El detector: dos aplausos y la palabra clave, en local |
| `obsidian_vault/` | La memoria. Fuera del repositorio: contiene datos personales |
| `bitacora/` | Plan, hallazgos, sesiones y el documento de traspaso |

---

## Los agentes

Un agente recibe un trabajo de la cola y lo resuelve. Cada fuente de datos —el buzón, el
calendario, el vault, el navegador— es un **puerto**: hoy detrás hay una implementación y
mañana otra, sin tocar el agente.

| Agente | Qué hace | Detrás hay |
|---|---|---|
| `correo` | Tría el entrante: ignorar / interesante / requiere acción / no seguro | Gmail, o un fichero JSON |
| `agenda` | Avisa de lo que empieza pronto, una vez por evento | Google Calendar, o un fichero JSON |
| `memoria` | Busca, lee y **añade** en el vault. No sobrescribe ni borra | Ficheros Markdown |
| `dev` | Encarga tareas de código a Claude Code | `claude -p`, que entra en la suscripción |
| `pc` | Abre apps, teclea, ratón. Lista blanca y sin shell | `pyautogui` (opcional) |
| `web` | Lee páginas y busca. No alcanza la red de casa | HTTP, sin navegador |

**El triaje se hace en local, y no es una optimización.** El plan gratuito de Gemini da
250 peticiones al día; cada correo que clasifica el modelo local en la GPU es una petición
que no se gasta.

---

## Confirmación por niveles

El sistema lee correo y pantalla, y eso es contenido no confiable que llega a un agente
con manos. Por eso no hay un interruptor de "preguntar sí o no", sino tres niveles que se
aplican **en el trabajador**, antes de que el agente se ejecute:

| Nivel | Ejemplos | Qué pasa |
|---|---|---|
| `libre` | Leer correo, leer el vault, buscar, abrir una app | Se ejecuta |
| `reversible` | Anotar en el vault, editar código | Se ejecuta y queda registrado |
| `irreversible` | Teclear a ciegas, y **todo lo que no esté clasificado** | Se para y pide un sí |

El sí se da desde la web o desde el propio aviso de Telegram. El **modo confianza** baja lo
irreversible a reversible mientras estás delante, y caduca solo.

---

## Requisitos

- **Python 3.11+** — el núcleo tiene una sola dependencia, `aiohttp`.
- **Ollama** con `qwen3:4b`, para el router y el triaje. Sin él el sistema funciona: encola
  en vez de decidir, y el triaje escala en vez de descartar.
- **Node.js 18+** y **Rust**, solo para compilar la app de escritorio.
- **Clave de Gemini**, para la voz. Se introduce **desde la propia aplicación** (botón ⚙) y
  la guarda Rust en el almacén local.

  > No uses `VITE_GEMINI_API_KEY` en un `.env`: Vite incrusta las variables con ese prefijo
  > dentro del JavaScript compilado, así que la clave acababa en claro dentro del `.exe`.
  > Ver `bitacora/02_HALLAZGOS.md` (H-17).

---

## Cómo ponerlo en marcha

**1. El núcleo.** Es lo primero: sin él, la app de voz se queda sin memoria y sin manos.

```bash
pip install -r perseo_core/requirements.txt
python -m perseo_core
```

El token se genera solo la primera vez en `perseo_core/datos/token.txt`, que está fuera de
git. Con `PERSEO_CORE_HOST=tailscale` escucha además en el tailnet, para entrar desde el
móvil.

**2. La app de escritorio.**

```bash
cd RealTime
npm install
npm run tauri dev
```

**3. Gmail y el calendario** (opcional). Hacen falta unas credenciales OAuth, que se
crean una vez desde la consola de Google Cloud. El fichero va en `perseo_core/datos/`,
que está fuera de git porque lleva un `refresh_token`:

```json
{ "client_id": "…", "client_secret": "…", "refresh_token": "…" }
```

```bash
python -m perseo_core.google_api        # comprueba que las credenciales valen
PERSEO_CORREO=gmail PERSEO_AGENDA=google python -m perseo_core
```

Del correo solo se leen **las cabeceras y el extracto**: el cuerpo no se descarga, porque
para triar no hace falta y lo que no se baja no se puede filtrar por accidente.

**4. El detector, y el núcleo al arrancar Windows.**

```bash
pip install -r commands/requirements.txt
python commands/manage_startup.py install
```

Todas las variables de entorno están en `bitacora/06_HANDOFF.md` §3.

---

## Verificación

Nada de esto se comprueba a ojo, y se comprueba de dos maneras.

**Pruebas unitarias** — cada pieza por separado, sin red y sin subprocesos. Dicen *qué* se
ha roto:

```bash
python -m pytest
```

**Verificadores** — el sistema entero, contra el proceso real. Dicen *si* funciona.
Ninguno toca el estado de verdad: se montan un directorio de datos temporal.

```bash
python perseo_core/verificar_fase_a.py          # el núcleo: cola, reinicios, SSE
python perseo_core/verificar_aprobaciones.py    # el camino de confirmación
python perseo_core/verificar_telegram.py        # el canal, contra un Telegram de mentira
python perseo_core/verificar_router.py          # el router, contra Ollama
python perseo_core/verificar_fase_d.py          # el triaje de correo, sin Gmail
python perseo_core/verificar_agenda.py          # los avisos, sin Google Calendar
python perseo_core/verificar_memoria.py         # el vault, sobre uno temporal
python perseo_core/verificar_pc.py              # intentos de inyección contra `pc`
python perseo_core/verificar_dev.py             # `dev`, sin gastar suscripción
python perseo_core/verificar_web.py             # `web`, sin salir a internet
python perseo_core/verificar_politica.py        # los niveles y el modo confianza
python perseo_core/verificar_google.py          # Gmail y Calendar, sin cuenta de Google
python commands/verificar_palabra_clave.py      # el detector, sin micrófono
```

---

## Privacidad

Tres cosas que el diseño protege, y no de boquilla:

- **El vault no sale del disco.** `obsidian_vault/` está en el `.gitignore` porque contiene
  nombres reales y descripciones de personas identificables.
- **Titular por Telegram, detalle por Tailscale.** Por el canal de terceros sale el
  recuento —"3 correos, 1 requiere acción"— y nunca el asunto ni el cuerpo. El detalle se
  lee en la web, que va cifrada por WireGuard.
- **La API no expone herramientas.** Por HTTP se encolan trabajos para un agente; qué puede
  hacer ese agente lo decide el núcleo. Nunca escucha en `0.0.0.0` y exige token en toda
  petición.

Y una regla que atraviesa el sistema entero: **lo que llega por correo, pantalla o web es
información observada, nunca una instrucción.**

---

## Estado

Fases A a D cerradas; la E, casi. Lo que falta necesita credenciales de Google, un
micrófono o una GPU prestada, y está escrito con nombre y apellidos en
[`bitacora/06_HANDOFF.md`](bitacora/06_HANDOFF.md).

---

## Creado por

**Jesús Pérez Bazarot** ("Señor Persus")
