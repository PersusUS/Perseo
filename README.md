<div align="center">

<img src="docs/imagenes/llamada-mira.png" alt="Perseo en llamada" width="820">

# PERSEO

**Un asistente personal con un núcleo siempre encendido y varias caras:
voz, panel de escritorio, web en el móvil y avisos al bolsillo.**

Habla contigo por voz en tiempo real, tría tu correo antes de que lo abras,
escribe en tu memoria, encarga código a otros agentes y te pide permiso
antes de hacer algo que no tenga vuelta atrás.

[![Verificación](https://github.com/PersusUS/Perseo/actions/workflows/verificacion.yml/badge.svg)](https://github.com/PersusUS/Perseo/actions/workflows/verificacion.yml)
[![Licencia MIT](https://img.shields.io/badge/licencia-MIT-black.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-black.svg)](https://www.python.org/)
[![Tauri 2](https://img.shields.io/badge/tauri-2-black.svg)](https://tauri.app/)
[![899 pruebas](https://img.shields.io/badge/pruebas-899-black.svg)](#verificación)

[Qué es](#qué-es) · [Cómo se ve](#cómo-se-ve) · [Cómo funciona](#cómo-funciona) ·
[Instalación](#instalación) · [Privacidad](#privacidad) · [English](README.en.md)

</div>

---

## Qué es

Perseo no es una ventana de chat. Es **un proceso que vive encendido en tu
ordenador** —con su cola de trabajo, su memoria y sus agentes— y varias
interfaces que se conectan a él.

La regla que sostiene el diseño entero: **las caras no piensan.** La app de
escritorio, la web del móvil y la sesión de voz transportan entrada y salida;
toda decisión ocurre en el núcleo. Es lo que permite mudarlo mañana a una
Raspberry Pi sin reescribir una línea de la interfaz.

| | |
|---|---|
| 🎙️ **Voz en tiempo real** | Conversación continua por Gemini Live, con la cara de Perseo, transcripción y manos: abre programas, teclea, mira tu pantalla |
| 📬 **Correo triado antes de que lo leas** | Cada mensaje cae en un cajón —ignorar, interesante, requiere acción, no seguro— decidido por un modelo **local**, en tu GPU |
| 🧠 **Memoria de verdad** | Notas Markdown en tu vault de Obsidian. Busca, lee y **añade**; nunca sobrescribe ni borra |
| 👤 **Sabe quién habla** | Reconoce voces y caras con modelos locales, y aprende solo a quien no conoce. Apagado de fábrica |
| 🛑 **Pide permiso, y sabe a quién** | Cuatro niveles de confirmación aplicados en el trabajador, no en el prompt. Lo irreversible se para y espera tu sí — y una orden de una visita se para aunque estés delante |
| 📱 **Te sigue al móvil** | Una PWA por la VPN de casa: chat, cola, correo y estado. Sin build y en un solo fichero |
| 🤖 **Delega código** | Encarga tareas a subagentes (Claude Code u opencode) y te cuenta por dónde van mientras trabajan |
| 🔌 **Habla MCP** | Cliente propio para servidores locales y remotos: vault, navegador, Windows, correo triado, subagentes |

---

## Cómo se ve

### La llamada

La pantalla que ves mientras hablas con él. Tres composiciones, la misma
información: se eligen en Ajustes y se cambian en caliente.

<div align="center">
<img src="docs/imagenes/llamada-mira.png" alt="Composición: mira" width="800">
<br><sub><b>Mira</b> — anillos, retícula y lecturas alrededor de la cara</sub>
</div>

<div align="center">
<img src="docs/imagenes/llamada-mando.png" alt="Composición: puesto de mando" width="800">
<br><sub><b>Puesto de mando</b> — instrumentos a un lado, registro de la llamada al otro</sub>
</div>

<div align="center">
<img src="docs/imagenes/llamada-cartel.png" alt="Composición: cartel" width="800">
<br><sub><b>Cartel</b> — composición descentrada, con el estado en grande</sub>
</div>

### El panel

Seis pestañas sobre lo mismo que mueve el núcleo. Se abre sin cortar la llamada.

<table>
<tr>
<td width="50%"><img src="docs/imagenes/panel-chat.png" alt="Panel: chat"><br><sub><b>Chat</b> — la conversación escrita, con las mismas manos que la voz</sub></td>
<td width="50%"><img src="docs/imagenes/panel-cola.png" alt="Panel: cola"><br><sub><b>Cola</b> — cada trabajo, su estado y quién lo pidió</sub></td>
</tr>
<tr>
<td><img src="docs/imagenes/panel-correo.png" alt="Panel: correo"><br><sub><b>Correo</b> — el buzón ya triado, y qué hiciste con cada uno</sub></td>
<td><img src="docs/imagenes/panel-memoria.png" alt="Panel: memoria"><br><sub><b>Memoria</b> — buscar y anotar en el vault sin salir de aquí</sub></td>
</tr>
<tr>
<td><img src="docs/imagenes/panel-encargos.png" alt="Panel: encargos"><br><sub><b>Encargos</b> — los subagentes de código, y por dónde van</sub></td>
<td><img src="docs/imagenes/panel-estado.png" alt="Panel: estado"><br><sub><b>Estado</b> — de qué está capado el sistema hoy, y la máquina</sub></td>
</tr>
</table>

### El tablero y los hábitos

<table>
<tr>
<td width="50%"><img src="docs/imagenes/tareas.png" alt="Tablero de tareas"><br><sub><b>Tablero</b> — notas que se clavan y se arrastran. Perseo también las mueve, por voz</sub></td>
<td width="50%"><img src="docs/imagenes/habitos.png" alt="Seguimiento de hábitos"><br><sub><b>Hábitos</b> — el mes entero de un vistazo, en dos aires</sub></td>
</tr>
</table>

### El móvil

Una PWA que sirve el propio núcleo. Se abre contra la VPN de casa, se pega el
token una vez y se guarda en la pantalla de inicio.

<table>
<tr>
<td width="33%"><img src="docs/imagenes/movil-token.png" alt="Móvil: token"><br><sub>El token, una sola vez</sub></td>
<td width="33%"><img src="docs/imagenes/movil-cola.png" alt="Móvil: cola"><br><sub>La cola, con lo que espera tu sí</sub></td>
<td width="33%"><img src="docs/imagenes/movil-estado.png" alt="Móvil: estado"><br><sub>Qué funciona hoy y qué no</sub></td>
</tr>
</table>

---

## Cómo funciona

```
   iPhone (PWA)              PC: app Tauri (bandeja)
        │ VPN                     │            │
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
  │                          │ chat     mcp      │   │
  │                          └───────────────────┘   │
  │                                                  │
  │  política de confirmación por niveles            │
  └──────┬───────────────────────────────────┬───────┘
         │                                   │
  ┌──────▼───────┐                  ┌────────▼────────┐
  │ Gemini Live  │                  │ vault Obsidian  │
  │ (voz)        │                  │ (memoria)       │
  └──────────────┘                  └─────────────────┘
```

| Carpeta | Qué es |
|---|---|
| [`perseo_core/`](perseo_core) | El núcleo: cola, bus, API, router local, los agentes y la política |
| [`RealTime/`](RealTime) | La app de escritorio (Tauri 2 + React + TypeScript): voz, panel y pantalla |
| [`commands/`](commands) | El detector (dos aplausos y palabra clave), el lanzador `perseo` y los servidores MCP propios |
| [`pruebas/`](pruebas) | Las pruebas del núcleo |
| [`docs/`](docs) | Configuración, API y privacidad, en detalle |

### Los agentes

Un agente recibe un trabajo de la cola y lo resuelve. Cada fuente de datos —el
buzón, el calendario, el vault, el navegador— es un **puerto**: hoy detrás hay
una implementación y mañana otra, sin tocar el agente.

| Agente | Qué hace | Detrás hay |
|---|---|---|
| `correo` | Tría el entrante: ignorar / interesante / requiere acción / no seguro | Gmail, o un fichero JSON |
| `agenda` | Avisa de lo que empieza pronto, una vez por evento | Google Calendar, o un fichero JSON |
| `memoria` | Busca, lee y **añade** en el vault. No sobrescribe ni borra | Ficheros Markdown, o el plugin REST de Obsidian |
| `chat` | Sostiene el chat escrito del panel y del móvil, y usa las herramientas de los demás | Gemini REST con *function calling* |
| `dev` | Encarga tareas de código a un subagente y **cuenta por dónde va** mientras trabaja | `claude-agent-sdk`, o `claude -p`, o opencode |
| `pc` | Abre apps, teclea, mueve el ratón. Lista blanca y sin shell | `pyautogui` (opcional) |
| `web` | Lee páginas y busca. No alcanza la red de casa | HTTP, sin navegador |
| `mcp` | Habla con los servidores MCP declarados: vault, navegador, Windows, tiempo, correo triado | JSON-RPC por stdio con cliente propio; los remotos, por HTTP con el SDK oficial |
| `eco`, `simulacro` | Los dos de pruebas: devuelven lo que reciben, sin tocar nada | — |

**El triaje se hace en local, y no es una optimización.** El plan gratuito de
Gemini da 250 peticiones al día; cada correo que clasifica el modelo local en
tu GPU es una petición que no se gasta.

### Reconocimiento de personas

En llamada, Perseo puede poner **nombre a cada voz** —un chip dice quién
habla— y **etiquetar las caras** que ve por la cámara. Y a quien no conoce lo
aprende solo: unos doce segundos de voz de un desconocido bastan para fijar un
perfil («Desconocido 1», renombrable desde Ajustes), y con las caras pasa otro
tanto contando detecciones.

Todo se decide en el núcleo con modelos locales —ECAPA-TDNN para la voz,
YuNet + SFace para caras— y todo se queda en tu disco: los perfiles son solo
vectores de números, fuera de git. Nada viaja a ningún servicio nuevo.

**Va apagado de fábrica**, porque voz y cara son datos biométricos: se
enciende a mano, y borrar un perfil borra sus números de verdad.

### Confirmación por niveles

El sistema lee correo y mira pantallas, y eso es contenido no confiable
llegando a un agente con manos. Por eso no hay un interruptor de «preguntar sí
o no», sino tres niveles que se aplican **en el trabajador**, antes de que el
agente se ejecute:

| Nivel | Ejemplos | Qué pasa |
|---|---|---|
| `libre` | Leer correo, leer el vault, buscar, abrir una app | Se ejecuta |
| `reversible` | Anotar en el vault, editar código | Se ejecuta y queda registrado |
| `irreversible` | Teclear a ciegas, y **todo lo que no esté clasificado** | Se para y pide un sí |

Y hay un cuarto, `critico` —borrar, tocar el registro, matar procesos—, que
pregunta **siempre**, con modo confianza o sin él.

El sí se da desde la web, desde el aviso de Telegram o **en voz alta durante la
llamada**. El *modo confianza* baja lo irreversible a reversible mientras estás
delante, y caduca solo: durante una llamada se renueva con tu voz, así que se
apaga sola si te levantas.

**Quién lo pide también cuenta.** Cada trabajo viaja con el perfil de quien
habló —lo pone el reconocimiento de voz, que corre en tu ordenador—, y una
orden de alguien que no eres tú se para aunque la confianza esté encendida. Un
sí tuyo vale además para las repeticiones exactas de lo mismo durante diez
minutos: dictar una dirección son seis órdenes idénticas, y preguntar seis
veces enseña a decir que sí sin leer.

---

## Instalación

> **Lo mínimo que funciona:** Python 3.11 y `pip install -r perseo_core/requirements.txt`.
> Con eso ya tienes núcleo, cola, memoria en ficheros, API y la web del móvil.
> Todo lo demás —voz, triaje local, Gmail, Telegram, biometría— se enciende
> cuando quieras y **el sistema arranca igual sin ello**, diciéndote qué falta.

### Requisitos

| | Para qué | ¿Obligatorio? |
|---|---|---|
| **Python 3.11+** | El núcleo. Una sola dependencia: `aiohttp` | **Sí** |
| **Node.js 18+ y Rust** | Compilar la app de escritorio | Solo para la voz y el panel |
| **Clave de Gemini** | La voz y el chat escrito | Solo para hablar con él |
| **[Ollama](https://ollama.com) con `qwen3:4b`** | El router y el triaje de correo, en local | No: sin él encola en vez de decidir |
| **Obsidian** | La memoria por el plugin REST | No: sin él la memoria va a ficheros |
| **Credenciales de Google** | Gmail y Calendar de verdad | No: hay buzón y agenda de mentira en JSON |

### 1 · El núcleo

Es lo primero: sin él, la app de voz se queda sin memoria y sin manos.

```bash
git clone https://github.com/PersusUS/Perseo.git
cd Perseo
pip install -r perseo_core/requirements.txt
python -m perseo_core
```

Escucha en `http://127.0.0.1:8787` y **se genera solo un token** la primera
vez, en `perseo_core/datos/token.txt`. Esa carpeta está fuera de git: dentro
viven la cola —con el texto literal de lo que le pides— y esa credencial.

Compruébalo:

```bash
curl http://127.0.0.1:8787/salud
```

### 2 · Dile para quién trabaja

Perseo viene con el nombre de su autor puesto. Dos variables lo cambian en
todo lo que piensa el núcleo —el router, el triaje y el chat—:

```bash
export PERSEO_DUENO="Ada Lovelace"
export PERSEO_TRATO="la señora Lovelace"
```

El personaje largo de la voz —el mayordomo, la casa, las mascotas— se edita
aparte, en **Ajustes → Instrucciones del sistema**, dentro de la propia app.

### 3 · La app de escritorio

```bash
cd RealTime
npm install
npm run tauri dev
```

La **clave de Gemini se introduce desde la propia aplicación** (botón ⚙) y la
guarda Rust en el almacén local del sistema.

> No la pongas en un `.env` como `VITE_GEMINI_API_KEY`: Vite incrusta las
> variables con ese prefijo dentro del JavaScript compilado, y la clave
> acabaría en claro dentro del `.exe`. Si prefieres una variable de entorno,
> usa `GEMINI_API_KEY` —sin prefijo—, que se lee en tiempo de ejecución.

### 4 · El móvil (opcional)

Abre `http://<la-ip-de-tu-VPN>:8787` en el navegador del teléfono, pega el
token una vez y añádelo a la pantalla de inicio. Para que el núcleo escuche
también fuera del bucle local:

```bash
PERSEO_CORE_HOST=127.0.0.1,100.64.0.1 python -m perseo_core
```

Con [Tailscale](https://tailscale.com), la palabra `tailscale` resuelve sola la
dirección del tailnet: `PERSEO_CORE_HOST=tailscale`.

> El micrófono del navegador exige contexto seguro. Si quieres hablarle desde
> el móvil, sirve por HTTPS con `PERSEO_TLS_CERT` y `PERSEO_TLS_CLAVE`
> —`tailscale cert` los emite—.

### 5 · Gmail y el calendario (opcional)

Hacen falta unas credenciales OAuth, que se crean una vez en la consola de
Google Cloud. El fichero va en `perseo_core/datos/`, fuera de git, porque
lleva un `refresh_token`:

```json
{ "client_id": "…", "client_secret": "…", "refresh_token": "…" }
```

```bash
python -m perseo_core.servicios.autorizar_google     # abre el consentimiento y guarda el testigo
python -m perseo_core.servicios.google_api           # comprueba que las credenciales valen
PERSEO_CORREO=gmail PERSEO_AGENDA=google python -m perseo_core
```

Del correo solo se leen **las cabeceras y el extracto**: el cuerpo no se
descarga, porque para triar no hace falta y **lo que no se baja no se puede
filtrar por accidente**.

¿Sin cuenta de Google? Pon dos ficheros JSON y pruébalo entero:

```bash
PERSEO_CORREO=falso PERSEO_AGENDA=falso python -m perseo_core
```

### 6 · El detector de aplausos (opcional, Windows)

```bash
pip install -r commands/requirements.txt
python commands/clap_detector.py
```

Dos aplausos, o la palabra clave, y Perseo entra en llamada solo.

### Todo a la vez

Cuando ya está instalado, un comando desde cualquier terminal:

```bash
perseo            # enciende lo que falte: núcleo, detector y app
perseo estado     # qué hay vivo ahora mismo, sin tocar nada
perseo parar      # cierra el núcleo y la app
```

**Todas las variables de entorno están en [`docs/CONFIGURACION.md`](docs/CONFIGURACION.md)**,
y las rutas de la API en [`docs/API.md`](docs/API.md).

---

## Verificación

Nada de esto se comprueba a ojo, y se comprueba de dos maneras.

**Pruebas unitarias** — cada pieza por separado, sin red y sin subprocesos.
Dicen *qué* se ha roto: **899** en total.

```bash
python commands/perseo.py comprobar    # todo, en orden de coste

python -m pytest                       # 756, el núcleo y los comandos
cd RealTime && npm test                # 143, la interfaz
cd RealTime/src-tauri && cargo check   # y que el Rust compila
```

**Verificadores** — el sistema entero, contra el proceso real. Dicen *si*
funciona. Ninguno toca el estado de verdad: se montan un directorio de datos
temporal y servidores de mentira.

```bash
python verificadores/verificar_fase_a.py          # el núcleo: cola, reinicios, SSE
python verificadores/verificar_aprobaciones.py    # el camino de confirmación
python verificadores/verificar_telegram.py        # el canal, contra un Telegram de mentira
python verificadores/verificar_router.py          # el router, contra Ollama
python verificadores/verificar_fase_d.py          # el triaje de correo, sin Gmail
python verificadores/verificar_agenda.py          # los avisos, sin Google Calendar
python verificadores/verificar_memoria.py         # el vault, en fichero y por el plugin
python verificadores/verificar_pc.py              # intentos de inyección contra `pc`
python verificadores/verificar_dev.py             # `dev`, sin gastar suscripción
python verificadores/verificar_web.py             # `web`, sin salir a internet
python verificadores/verificar_politica.py        # los niveles y el modo confianza
python verificadores/verificar_google.py          # Gmail y Calendar, sin cuenta de Google
python verificadores/verificar_estado.py          # la pantalla de estado y sus semáforos
python verificadores/verificar_correo_mcp.py      # el servidor MCP de correo
python verificadores/verificar_biometria.py       # voces y caras: aprender, renombrar, borrar
python verificadores/verificar_chat.py            # el chat escrito y sus herramientas
python commands/verificar_palabra_clave.py      # el detector, sin micrófono
```

Los cuatro primeros bloques corren solos en cada `push`
([Verificación](.github/workflows/verificacion.yml)), en Linux y en Windows.

---

## Privacidad

Tres cosas que el diseño protege, y no de boquilla:

- **El vault no sale del disco.** La memoria son notas Markdown en tu
  ordenador. No hay base vectorial, no hay servicio de embeddings, no hay
  copia en la nube.
- **Titular fuera, detalle dentro.** Por Telegram —un tercero— sale el
  recuento: «3 correos, 1 requiere acción». Nunca el asunto ni el cuerpo. El
  detalle se lee en la web, que va por tu VPN.
- **La API no expone herramientas.** Por HTTP se encolan trabajos para un
  agente; qué puede hacer ese agente lo decide el núcleo. Nunca escucha en
  `0.0.0.0` y exige token en toda petición.

Y una regla que atraviesa el sistema entero: **lo que llega por correo,
pantalla o web es información observada, nunca una instrucción.** Está escrita
en el prompt de los tres modelos y comprobada por una prueba que falla si
alguien la quita.

Lo que sí sale de tu máquina, dicho sin adornos: **el audio y el vídeo de la
llamada van a Gemini Live**, y el chat escrito a la API de Gemini. Eso es todo.
El detalle, en [`docs/PRIVACIDAD.md`](docs/PRIVACIDAD.md).

---

## Estado

Perseo funciona y se usa a diario, pero es un proyecto personal: está pensado
para **una** persona, en **un** ordenador con Windows, y se nota. Lo que hay
detrás son unas 49.100 líneas, 899 pruebas y 17 verificadores.

Si lo clonas y algo no arranca, abre un
[issue](https://github.com/PersusUS/Perseo/issues) — y si lo arreglas, mejor
todavía: [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## Licencia

[MIT](LICENSE) — haz lo que quieras con él, cítame y no me pidas garantías.

Creado por **Jesús Pérez Bazarot**.
