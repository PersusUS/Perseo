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

Dos sitios más donde vive la identidad, y que no son variables de entorno:

- **El personaje largo de la voz** —el mayordomo, la casa, las mascotas, el
  tono— se edita en la propia aplicación: **Ajustes → Instrucciones del
  sistema**. Su valor de fábrica está en `RealTime/src/lib/config.ts`.
- **Los avisos de identidad durante la llamada** («quien habla ahora NO es…»)
  salen de una sola constante, `TRATO_DUENO`, en
  `RealTime/src/lib/quien-hay.ts`.

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
| `PERSEO_DISPARADORES` | `correo,agenda` | Quién empieza trabajos solo. Vacío = nadie |

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

## Google

| Variable | Por defecto | Para qué |
|---|---|---|
| `PERSEO_GOOGLE_CREDENCIALES` | `<datos>/google.json` | `client_id`, `client_secret` y `refresh_token` |
| `PERSEO_GOOGLE_OAUTH` · `_GMAIL` · `_CALENDAR` | las de Google | Se apuntan a otro sitio para verificar sin cuenta |
| `PERSEO_GOOGLE_CUENTAS` | *(vacío)* | Varias cuentas, separadas por comas |

Los ámbitos que se piden son los más pequeños que sirven: `gmail.readonly`,
`calendar.readonly` y `gmail.compose` —que permite escribir un borrador pero
**no** enviarlo—.

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
