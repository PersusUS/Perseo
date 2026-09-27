# Configuración

Perseo se configura **solo con variables de entorno**. No hay fichero de
ajustes que editar ni asistente que rellenar: todas son opcionales, todas
traen un valor por defecto que funciona, y el sistema arranca aunque no
pongas ninguna.

La regla al leerlas: **lo que falta no rompe nada, se apaga.** Sin Ollama, el
router encola en vez de decidir. Sin Gmail, no hay correo. Sin Telegram, no hay
avisos. Y la pestaña **Estado** —en el panel y en el móvil— te dice cuál de
esas cosas está apagada hoy y qué comando la enciende.

---

## Quién eres tú

Perseo viene con el nombre de su autor puesto en el prompt de todos sus
modelos. Estas dos lo cambian:

| Variable | Por defecto | Para qué |
|---|---|---|
| `PERSEO_DUENO` | `Jesús Pérez Bazarot` | Para quién trabaja. Aparece en el prompt del router, del triaje y del chat |
| `PERSEO_TRATO` | `el señor Persus` | Cómo te llama. Va **con artículo**, porque las frases lo necesitan: «la señora Lovelace», «el doctor Chandra» |
| `PERSEO_PERFIL_DUENO` | `Persus` | El perfil del reconocimiento de voz que eres tú. Es lo que separa tus órdenes de las de una visita: lo que pida un perfil distinto se para y espera tu sí |

Dos sitios más donde vive la identidad, y que no son variables de entorno:

- **El personaje largo de la voz** —el mayordomo, la casa, las mascotas, el
  tono— se edita en la propia aplicación: **Ajustes → Instrucciones del
  sistema**. Su valor de fábrica está en `RealTime/src/lib/datos/config.ts`.
- **Los avisos de identidad durante la llamada** («quien habla ahora NO es…»)
  salen de una sola constante, `TRATO_DUENO`, en
  `RealTime/src/lib/identidad/quien-hay.ts`.

---

## El núcleo

| Variable | Por defecto | Para qué |
|---|---|---|
| `PERSEO_CORE_HOST` | `127.0.0.1` | Lista separada por comas. El valor `tailscale` añade la dirección del tailnet **sin** quitar la local |
| `PERSEO_CORE_PUERTO` | `8787` | |
| `PERSEO_CORE_DATOS` | `perseo_core/datos` | Dónde vive el estado. Útil para probar sin tocar lo de verdad |
| `PERSEO_CORE_DB` | `<datos>/estado.sqlite3` | La cola |
| `PERSEO_TOKEN` | *(se genera)* | Manda sobre `<datos>/token.txt` |
| `PERSEO_CORE_URL` | `http://127.0.0.1:8787` | **La lee la app**, para saber dónde está el núcleo |
| `PERSEO_URL_BASE` | la primera interfaz no local | Lo que se pone en el enlace «ver detalle» de los avisos |
| `PERSEO_DISPARADORES` | `correo,agenda,recordatorios,parte,vigilancias,seguimiento` | Quién empieza trabajos solo. Vacío = nadie |

### HTTPS

Existen **por el micrófono del móvil**: el navegador solo graba en contexto
seguro, y `http://` por la VPN no lo es. Vacíos —o apuntando a algo que no
está— el núcleo sirve por HTTP como siempre.

| Variable | Por defecto | Para qué |
|---|---|---|
| `PERSEO_TLS_CERT` | *(vacío)* | Certificado. Lo emite `tailscale cert` |
| `PERSEO_TLS_CLAVE` | *(vacío)* | Su clave |
| `PERSEO_TLS_PUERTO` | *(vacío)* | Puerto aparte para el HTTPS, si no quieres el mismo |

---

## Los modelos

| Variable | Por defecto | Para qué |
|---|---|---|
| `PERSEO_OLLAMA` | `http://127.0.0.1:11434` | Dónde escucha Ollama |
| `PERSEO_MODELO_ROUTER` | `qwen3:4b` | El que decide a qué agente va cada cosa |
| `PERSEO_MODELO_SUPLENTE` | *(vacío)* | Modelo de fuera que responde si Ollama no está. **Vacío = apagado**, y es lo único que mandaría a un tercero el texto que se clasifica |
| `PERSEO_CHAT_MODELO` | Gemini | El que sostiene el chat escrito |
| `GEMINI_API_KEY` | *(vacío)* | La clave. También la usa la app para la voz. Si no está, se lee `<datos>/gemini.txt` |

---

## El correo

| Variable | Por defecto | Para qué |
|---|---|---|
| `PERSEO_CORREO` | *(vacío)* | `gmail`, `falso`, o vacío = sin correo |
| `PERSEO_CORREO_FALSO` | `<datos>/buzon.json` | El JSON que hace de buzón cuando es `falso` |
| `PERSEO_CORREO_INTERVALO` | `300` | Cada cuántos segundos se mira el buzón |

Un `buzon.json` mínimo, para probarlo sin cuenta de Google:

```json
[
  {
    "id": "m1",
    "asunto": "Justificante de la beca — antes del viernes",
    "de": "becas@universidad.example",
    "extracto": "Falta el documento firmado para cerrar el expediente.",
    "momento": "2026-09-11T08:12:00"
  }
]
```

## La agenda

| Variable | Por defecto | Para qué |
|---|---|---|
| `PERSEO_AGENDA` | *(vacío)* | `google`, `falso`, o vacío = sin agenda |
| `PERSEO_AGENDA_FALSA` | `<datos>/agenda.json` | El JSON que hace de calendario |
| `PERSEO_AGENDA_INTERVALO` | `600` | Cada cuántos segundos se mira |
| `PERSEO_AGENDA_ANTELACION` | `60` | Con cuántos minutos de antelación se avisa |

## Los recordatorios

Se guardan en `<datos>/recordatorios.json` y los apunta el propio Perseo cuando
se le pide («avísame en veinte minutos»). No hace falta configurar nada.

| Variable | Por defecto | Para qué |
|---|---|---|
| `PERSEO_RECORDATORIOS_INTERVALO` | `30` | Cada cuántos segundos se mira si alguno ha vencido |

## El parte del día

Se pide hablando («¿qué tengo hoy?») o por escrito, y junta agenda, correos que
piden algo, recordatorios, tareas y hábitos. Además puede salir solo cada
mañana, con el titular por Telegram: **viene apagado**.

| Variable | Por defecto | Para qué |
|---|---|---|
| `PERSEO_PARTE_HORA` | *(vacío)* | `HH:MM` a la que sale solo, una vez al día (o en las tres horas siguientes si el núcleo arrancó más tarde). Vacío = solo cuando se pide |

## Google

| Variable | Por defecto | Para qué |
|---|---|---|
| `PERSEO_GOOGLE_CREDENCIALES` | `<datos>/google.json` | `client_id`, `client_secret` y `refresh_token` |
| `PERSEO_GOOGLE_OAUTH` · `_GMAIL` · `_CALENDAR` | las de Google | Se apuntan a otro sitio para verificar sin cuenta |
| `PERSEO_GOOGLE_CUENTAS` | *(vacío)* | Varias cuentas, separadas por comas |

Los ámbitos que se piden: `gmail.readonly`, `calendar.readonly`, `gmail.compose`
y `calendar.events`. **`gmail.compose` permite enviar**, no solo escribir
borradores —hasta el 2026-09-24 aquí ponía lo contrario, y era falso—. Lo que
impide que salga un correo o una invitación sin tu sí es la política: enviar e
invitar son de nivel `exterior` y se paran siempre ([ADR 0007](adr/0007-lo-que-sale-de-casa-se-para.md)).
`calendar.events` llegó el 2026-09-24 para apuntar citas: con un permiso de
antes, crear un evento contesta 403 hasta que vuelvas a pasar por
`python -m perseo_core.servicios.autorizar_google`.

---

## La memoria

| Variable | Por defecto | Para qué |
|---|---|---|
| `OBSIDIAN_VAULT_PATH` | `<repositorio>/../obsidian_vault` | Raíz del vault. El respaldo se crea solo al primer apunte, para poder probar |
| `PERSEO_VAULT` | *(vacío)* | Vacío = ficheros. `rest` = el plugin Local REST API de Obsidian |
| `PERSEO_VAULT_REST` | `https://127.0.0.1:27124` | Dónde escucha el plugin. Su certificado se lo firma él, así que solo se acepta sin verificar **en el bucle local** |
| `PERSEO_VAULT_CLAVE` | *(vacío)* | La clave del plugin, que sale en sus ajustes. Si no está, se lee `<datos>/obsidian.txt` |

Perseo escribe siempre dentro de **su propia carpeta** del vault, nunca
mezclado con tus notas, y **nunca sobrescribe ni borra**: solo añade.

---

## Telegram

| Variable | Por defecto | Para qué |
|---|---|---|
| `PERSEO_TELEGRAM_TOKEN` | *(vacío)* | Token del bot. Si no está, se lee `<datos>/telegram.txt` |
| `PERSEO_TELEGRAM_CHAT` | *(vacío)* | Tu `chat_id`. Si no está, `<datos>/telegram_chat.txt` |
| `PERSEO_TELEGRAM_API` | `https://api.telegram.org` | Se apunta a otro sitio para probar sin Telegram |
| `PERSEO_TELEGRAM_CONVERSAR` | *(vacío)* | `1` = además de avisar, **contesta**: le escribes al bot y habla contigo en el hilo principal. Solo atiende tu chat. Una ubicación compartida se guarda. Ver el [ADR 0009](adr/0009-canales-de-fuera.md) |

Por este canal sale **el recuento y nunca el contenido**: «3 correos, 1
requiere acción». El detalle se lee en la web.

---

## Los subagentes de código

| Variable | Por defecto | Para qué |
|---|---|---|
| `PERSEO_DEV_MOTOR` | `claude` | `claude`, `opencode` o `falso`. Cada encargo puede cambiarlo diciéndolo en el texto: «con opencode» |
| `PERSEO_DEV_RAIZ` | la carpeta del usuario | Desde aquí trabaja un encargo, y de aquí no sale |
| `PERSEO_DEV_RAICES_EXTRA` | *(vacío)* | Más carpetas permitidas, separadas por comas |
| `PERSEO_DEV_TOPE` | `900` | Segundos antes de cortar un encargo |
| `PERSEO_DEV_TARDANZA` | — | A partir de cuánto se avisa de que va lento |
| `PERSEO_DEV_CLAUDE` | `claude` | El ejecutable, por si no está en el PATH con ese nombre |
| `PERSEO_DEV_MODELO` | *(vacío)* | Modelo concreto para el subagente |

> **Ojo con `PERSEO_DEV_RAIZ`.** Es el cerco de un agente que escribe ficheros.
> Por defecto abarca tu carpeta de usuario entera, que es cómodo y es mucho: si
> lo vas a dejar encendido, apúntalo a la carpeta de tus proyectos y no más.

## La web

| Variable | Por defecto | Para qué |
|---|---|---|
| `PERSEO_WEB` | *(vacío)* | Vacío = HTTP de verdad. `falso` = navegador simulado |
| `PERSEO_WEB_TOPE_BYTES` | `2097152` | Cuánto se descarga como mucho de una página |
| `PERSEO_WEB_TOPE_SEGUNDOS` | `20` | Cuánto se espera a una página |
| `PERSEO_WEB_LOCAL` | *(vacío)* | **Solo para verificar.** Abre el bucle local, y solo el bucle local |

El agente `web` **no alcanza la red de casa**: las direcciones privadas están
cerradas a propósito, porque una página puede pedirle que mire dentro de tu
propia red.

---

## Los recados

El agente `recado` hace encargos enteros en la web con su propio navegador
(Chrome, por `@playwright/mcp`). Necesita Node.js y Google Chrome; la primera vez
instala su versión fijada de Playwright en `<datos>/navegador_mcp`.

| Variable | Por defecto | Para qué |
|---|---|---|
| `PERSEO_RECADO_MODELO` | los del chat | Qué modelo de Gemini piensa los recados. Lista con comas: el segundo entra si el primero no tiene cuota |
| `PERSEO_RECADO_PASOS` | `40` | Turnos de modelo por recado como mucho. Cada uno es una petición a Gemini |
| `PERSEO_RECADO_VISIBLE` | *(vacío)* | `1` = el navegador se ve en pantalla mientras trabaja. Vacío = en segundo plano |

Dos órdenes para prepararlo, en una terminal y no por el chat —lo que se escribe
ahí pasa por un modelo en la nube—:

```bash
python commands/perseo.py navegador
```

Abre el navegador de los recados para que entres **a mano** en tus sitios una
vez; las sesiones se quedan en `<datos>/navegador`.

```bash
python commands/perseo.py boveda guardar resy --sitio resy.com
```

Guarda una contraseña (o una tarjeta, con `--tarjeta` y `--tope 60`) cifrada con
DPAPI en `<datos>/boveda.json`. Cada entrada vale solo en sus sitios. El modelo
la escribe como `{{boveda:resy.clave}}` y nunca ve el valor.

Lo que sale de casa —pagar, reservar, enviar, meter una tarjeta— **se para a
esperar tu sí** aunque las confirmaciones estén apagadas, y ese sí se da en la
tarjeta del panel o del móvil, no hablando. Ver el
[ADR 0007](adr/0007-lo-que-sale-de-casa-se-para.md).

### Las vigilancias

«Avísame cuando haya entradas», «resérvalo si baja de 80 €». Se piden hablando
o por escrito y viven en `<datos>/vigilancias.json`. Cada comprobación es un
recado, y un recado gasta varias peticiones a Gemini, del mismo cubo que el
chat; por eso los topes: cada una se mira **como mucho cada hora** (tres, si no
se dice), no hay más de **cinco** a la vez, caducan a los siete días (treinta
como mucho) y entre todas no pasan de **24 comprobaciones al día**. Mientras no
se cumplen, no suena nada.

| Variable | Por defecto | Para qué |
|---|---|---|
| `PERSEO_VIGILANCIAS_INTERVALO` | `60` | Cada cuántos segundos se mira si a alguna le toca |

### El seguimiento

Si un correo que el triaje marcó como «requiere acción» lleva **entre dos y
catorce días** sin respuesta, Perseo te llama una vez para recordártelo, solo
entre las nueve y las nueve. Antes mira el hilo en Gmail: si el último mensaje
es tuyo, ya contestaste, y lo marca como atendido sin decir nada. No hace falta
configurarlo; con el buzón de mentira no hay hilos que mirar y avisa igual.

Solo llama por correos **de personas**. Lo que manda una máquina —boletines,
avisos de plataformas, `noreply@`, `notifications@`, `support@`, eventos de
Luma— no llama ni avisa: se queda en el panel de correo. Se reconoce por las
cabeceras de envío (`List-Unsubscribe`, `List-Id`, `Auto-Submitted`,
`Precedence`) y, si faltan, por la dirección.

---

## El teléfono y WhatsApp

Con una cuenta de Twilio, Perseo tiene un número de teléfono y un WhatsApp: le
escribes o le llamas desde el móvil, te llama él, y llama a un negocio en tu
nombre. Todo **apagado** hasta que pongas la cuenta; ver el
[ADR 0009](adr/0009-canales-de-fuera.md).

La cuenta va en `<datos>/twilio.json` (o en las variables del mismo nombre):

```json
{
  "sid": "AC…",
  "token": "…",
  "numero": "+34…",
  "whatsapp": "+14155238886",
  "dueno": "+34600…",
  "url_publica": "https://tu-pc.tu-tailnet.ts.net"
}
```

| Clave | Variable | Para qué |
|---|---|---|
| `sid` · `token` | `PERSEO_TWILIO_SID` · `PERSEO_TWILIO_TOKEN` | La cuenta. El token firma lo que llega: sin él no se atiende nada |
| `numero` | `PERSEO_TWILIO_NUMERO` | El número de Perseo, para llamar y para SMS |
| `whatsapp` | `PERSEO_TWILIO_WHATSAPP` | El remitente de WhatsApp de la cuenta (el del *sandbox* sirve para probar) |
| `dueno` | `PERSEO_TELEFONO_DUENO` | **Tu** móvil. Es el único que puede hablarle; el resto oye que no |
| `url_publica` | `PERSEO_TWILIO_URL_PUBLICA` | Por dónde llega Twilio a este PC |
| — | `PERSEO_TWILIO_PUERTO` | El puerto local de la cara de Twilio (`8788`) |
| — | `PERSEO_TWILIO_VOZ` | La voz de las llamadas (`Polly.Lucia`) |

Twilio tiene que llegar a este PC desde internet. Se publica **solo** el puerto
de Twilio, no el núcleo:

```bash
tailscale funnel 8788
```

Y en la consola de Twilio, las direcciones: `https://…/twilio/whatsapp` para los
mensajes (WhatsApp y SMS) y `https://…/twilio/voz` para las llamadas entrantes.

## La ubicación

Perseo no sigue tu posición: sabe dónde estás **cuando la compartes**, y guarda
solo la última. Llega por Telegram (con la conversación encendida) o por
WhatsApp (adjuntar → ubicación), o por `POST /ubicacion` desde un atajo del
iPhone. La herramienta `mi_ubicacion` se la da al modelo con su antigüedad.

## El detector

Es un proceso aparte (`python commands/clap_detector.py`).

| Variable | Por defecto | Para qué |
|---|---|---|
| `PERSEO_PALABRA_MOTOR` | `google` | `local`, para decidir la palabra sin red con openWakeWord |
| `PERSEO_MODELO_PALABRA` | `commands/modelos/perseo.onnx`, y si no existe, `hey_jarvis` | Solo con el motor local: ruta a otro `.onnx` |
| `PERSEO_UMBRAL_PALABRA` | `0.5` | Cuánto hay que parecerse. Bájalo si no te reconoce, súbelo si se activa de más |

---

## Los servidores MCP

Se declaran en `<datos>/mcp.json`. Los **locales** llevan `comando` y hablan
JSON-RPC por stdio; los **remotos** llevan `url` y van por HTTP con el SDK
oficial.

```json
{
  "vault": {
    "comando": "npx",
    "argumentos": ["-y", "obsidian-mcp"],
    "entorno": { "OBSIDIAN_API_KEY": "…" }
  },
  "tiempo": { "url": "https://ejemplo.example/mcp" }
}
```

---

## Dónde vive el estado

Todo lo que Perseo escribe mientras funciona cae en `<datos>`
(`perseo_core/datos` de fábrica), y **esa carpeta está fuera de git**:

| Fichero | Qué lleva dentro |
|---|---|
| `token.txt` | La credencial que abre la API |
| `estado.sqlite3` | La cola, con el **texto literal** de lo que le pides |
| `perfiles.json` | Los vectores de voz y cara de las personas reconocidas |
| `google.json` | El `refresh_token` que abre tu buzón |
| `gemini.txt`, `telegram.txt`, `obsidian.txt` | Claves sueltas |
| `nucleo.log`, `vigilante.log` | Los registros |
| `proyectos.json`, `mcp.json`, `entorno.json` | Qué hay declarado en esta máquina |

Si haces copia de seguridad de Perseo, es de esta carpeta. Si compartes tu
pantalla, es la que no conviene abrir.
