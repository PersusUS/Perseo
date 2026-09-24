"""Los dos puertos del agente `recado`: quién piensa y quién toca la web.

Como el buzón, el vault y el motor de `dev`: el agente no sabe si detrás hay
Gemini o un guion escrito a mano, ni si el navegador es Chrome o un diccionario
de páginas de mentira. Eso es lo que deja probar el bucle entero —la bóveda, la
parada antes de pagar, seguir tras el sí— sin gastar una petición ni abrir un
navegador, y lo que hará que cambiar de modelo sea escribir una clase.

**El navegador de los recados es suyo, no el del chat.** Otro proceso de
`@playwright/mcp`, con su perfil en `<datos>/navegador`: ahí se quedan las
sesiones de los sitios en los que el señor Persus entre una vez a mano
(`perseo navegador`), que es lo que Instinct llama «un ordenador con tus
sesiones». No comparte perfil con el servidor `navegador` de `mcp.json` porque
Chrome no deja abrir el mismo perfil dos veces, y un recado largo no debe
bloquear al chat ni el chat a él.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Protocol

import aiohttp

from ..infra import almacen
from ..servicios.mcp_transportes import ErrorMcp, abrir_servidor

logger = logging.getLogger(__name__)

#: Fijada, no `@latest`: el esquema de las herramientas ha cambiado entre
#: versiones —`ref` pasó a llamarse `target`— y el agente lee los nombres de
#: los campos para decidir qué es lo que se pulsa. Subirla es probarla antes.
PAQUETE_PLAYWRIGHT = "@playwright/mcp@0.0.82"

#: Lo que el modelo puede pedirle al navegador. Lo que falta, falta a
#: propósito: `browser_file_upload` subiría ficheros del disco a una web,
#: `browser_evaluate` y `browser_run_code` ejecutan código que ninguna
#: comprobación de aquí sabe leer.
HERRAMIENTAS_NAVEGADOR = (
    "browser_navigate",
    "browser_navigate_back",
    "browser_snapshot",
    "browser_click",
    "browser_type",
    "browser_fill_form",
    "browser_select_option",
    "browser_press_key",
    "browser_hover",
    "browser_wait_for",
    "browser_tabs",
    "browser_handle_dialog",
)


class ErrorCerebro(Exception):
    """El modelo no contestó. El recado se para con lo que lleve hecho."""


# --------------------------------------------------------------------------- #
# Quién piensa
# --------------------------------------------------------------------------- #


class Cerebro(Protocol):
    async def pensar(
        self, sistema: str, contents: list[dict[str, Any]], declaraciones: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Un turno del modelo, como `content` de la API: `{"role": "model", "parts": [...]}`."""
        ...

    async def cerrar(self) -> None: ...


def _modelos() -> tuple[str, ...]:
    """Los del chat, salvo que `PERSEO_RECADO_MODELO` diga otros (lista con comas)."""
    from .chat import MODELOS_POR_DEFECTO

    pedidos = os.environ.get("PERSEO_RECADO_MODELO", "").strip()
    if not pedidos:
        return MODELOS_POR_DEFECTO
    return tuple(m.strip() for m in pedidos.split(",") if m.strip()) or MODELOS_POR_DEFECTO


class CerebroGemini:
    """`generateContent` sin streaming: aquí nadie lee el texto mientras sale.

    El turno del modelo se devuelve **tal cual**, con sus `thoughtSignature`
    pegadas a las llamadas: los modelos que piensan exigen recibirlas de vuelta
    en la ronda siguiente, y reconstruir las partes a mano es como se pierden
    (ver `_parte_de_llamada` en `chat.py`, que lo aprendió así).

    Ante un 429 hace lo mismo que el chat: primero el modelo hermano, que tiene
    su propio cubo de cuota; solo si todos dicen que no, espera y reintenta.
    """

    def __init__(self, clave: str) -> None:
        self._clave = clave
        self._sesion: aiohttp.ClientSession | None = None

    async def pensar(
        self, sistema: str, contents: list[dict[str, Any]], declaraciones: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if not self._clave:
            raise ErrorCerebro("No hay clave de Gemini: <datos>/gemini.txt o GEMINI_API_KEY.")
        if self._sesion is None:
            self._sesion = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120))
        base = os.environ.get("PERSEO_GEMINI_API", "https://generativelanguage.googleapis.com").rstrip("/")
        cuerpo = {
            "contents": contents,
            "tools": [{"functionDeclarations": declaraciones}],
            "systemInstruction": {"parts": [{"text": sistema}]},
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2048},
        }
        modelos = _modelos()
        for intento in range(3):
            for modelo in modelos:
                url = f"{base}/v1beta/models/{modelo}:generateContent"
                try:
                    async with self._sesion.post(url, params={"key": self._clave}, json=cuerpo) as r:
                        await asyncio.to_thread(almacen.apuntar_uso, modelo)
                        if r.status == 429:
                            logger.info("Recado: %s sin cuota (429); se prueba el siguiente.", modelo)
                            continue
                        datos = await r.json(content_type=None)
                        if r.status != 200:
                            mensaje = (datos.get("error") or {}).get("message") if isinstance(datos, dict) else None
                            raise ErrorCerebro(f"Gemini respondió {r.status}: {mensaje or str(datos)[:200]}")
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    raise ErrorCerebro(f"No se pudo hablar con Gemini: {e}") from e
                candidatos = datos.get("candidates") or []
                contenido = (candidatos[0] if candidatos else {}).get("content") or {}
                if not contenido.get("parts"):
                    motivo = (candidatos[0] if candidatos else {}).get("finishReason", "sin partes")
                    raise ErrorCerebro(f"Gemini devolvió un turno vacío ({motivo}).")
                contenido["role"] = "model"
                return contenido
            if intento < 2:
                await asyncio.sleep(20 * (intento + 1))
        raise ErrorCerebro("Ningún modelo tiene cuota ahora mismo: " + ", ".join(modelos))

    async def cerrar(self) -> None:
        if self._sesion is not None:
            await self._sesion.close()
            self._sesion = None


# --------------------------------------------------------------------------- #
# Quién toca la web
# --------------------------------------------------------------------------- #


class Manos(Protocol):
    #: Las herramientas del navegador con su `inputSchema`, tras `arrancar`.
    herramientas: list[dict[str, Any]]

    async def arrancar(self) -> None: ...

    async def llamar(self, herramienta: str, argumentos: dict[str, Any]) -> str: ...

    async def detener(self) -> None: ...


def carpeta_paquete(directorio_datos: Path) -> Path:
    return Path(directorio_datos) / "navegador_mcp"


def cli_instalado(directorio_datos: Path) -> Path | None:
    cli = carpeta_paquete(directorio_datos) / "node_modules" / "@playwright" / "mcp" / "cli.js"
    return cli if cli.is_file() else None


async def instalar_paquete(directorio_datos: Path) -> Path:
    """Instala la versión fijada en `<datos>/navegador_mcp`, una sola vez.

    **Por qué no `npx`, que es lo que usa `mcp.json`.** Medido el 2026-09-24 en
    este PC: `npx -y @playwright/mcp@0.0.82` tardaba entre 40 y 50 segundos en
    contestar al saludo —con el paquete ya en caché, y con `--prefer-offline`
    igual—, y `node cli.js` del mismo paquete, 1,4. Con npx cada recado
    empezaba con casi un minuto de espera, y a veces pasaba del plazo de 90 s y
    fallaba sin haber hecho nada.
    """
    import shutil

    npm = shutil.which("npm")
    if npm is None:
        raise ErrorMcp("Los recados necesitan Node.js (npm no está en el PATH).")
    carpeta = carpeta_paquete(directorio_datos)
    carpeta.mkdir(parents=True, exist_ok=True)
    logger.info("Recados: instalando %s en %s (solo la primera vez).", PAQUETE_PLAYWRIGHT, carpeta)
    proceso = await asyncio.create_subprocess_exec(
        npm, "install", "--prefix", str(carpeta), "--no-audit", "--no-fund", "--no-save", PAQUETE_PLAYWRIGHT,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        salida, _ = await asyncio.wait_for(proceso.communicate(), timeout=300)
    except asyncio.TimeoutError:
        proceso.kill()
        raise ErrorMcp("Instalar el navegador de los recados pasó de cinco minutos.") from None
    cli = cli_instalado(directorio_datos)
    if proceso.returncode != 0 or cli is None:
        detalle = salida.decode("utf-8", "replace").strip().splitlines()[-3:]
        raise ErrorMcp("No se pudo instalar el navegador de los recados: " + " / ".join(detalle))
    return cli


def comando_playwright(directorio_datos: Path, visible: bool, cli: Path | None = None) -> list[str]:
    """La línea que arranca el navegador de los recados.

    `--snapshot-mode none` no es por ahorrar: con el modo de fábrica, cada clic
    **guarda la instantánea en un fichero**, y una instantánea enseña lo
    tecleado —también en un campo de contraseña, medido el 2026-09-24—. Eran
    claves en claro en el disco. Así no se escribe ninguna, y el agente pide la
    suya con `browser_snapshot`, que vuelve por la tubería y no toca el disco.

    `--output-dir` dentro de `<datos>` por lo que aún escribe (el registro de la
    consola): sin él lo dejaba en la raíz del repositorio. Se vacía al acabar
    cada recado, ver `vaciar_salida`.
    """
    comando = [
        *(["node", str(cli)] if cli is not None else ["npx", "-y", PAQUETE_PLAYWRIGHT]),
        "--user-data-dir",
        str(Path(directorio_datos) / "navegador"),
        "--output-dir",
        str(carpeta_salida(directorio_datos)),
        "--snapshot-mode",
        "none",
    ]
    if not visible:
        comando.append("--headless")
    return comando


def carpeta_salida(directorio_datos: Path) -> Path:
    return Path(directorio_datos) / "navegador_salida"


def vaciar_salida(directorio_datos: Path) -> None:
    """Borra lo que Playwright dejó escrito. Un GET con una clave la lleva en la URL,
    y la URL acaba en el registro de la consola."""
    for fichero in carpeta_salida(directorio_datos).glob("*"):
        try:
            if fichero.is_file():
                fichero.unlink()
        except OSError:
            pass


class ManosPlaywright:
    """El navegador de verdad: un `@playwright/mcp` por stdio, solo para recados."""

    def __init__(self, directorio_datos: Path, visible: bool = False, comando: list[str] | None = None) -> None:
        self._datos = Path(directorio_datos)
        self._visible = visible
        self._comando = comando
        self._servidor = None
        self.herramientas: list[dict[str, Any]] = []

    async def arrancar(self) -> None:
        if self._servidor is None:
            comando = self._comando
            if comando is None:
                cli = cli_instalado(self._datos) or await instalar_paquete(self._datos)
                comando = comando_playwright(self._datos, self._visible, cli)
            self._servidor = abrir_servidor(
                "recados",
                {
                    "comando": comando,
                    "env": {},
                    # Una página lenta más su instantánea caben de sobra; lo que
                    # pase de aquí es un navegador colgado, y se dice.
                    "tope_segundos": 90,
                    "herramientas": list(HERRAMIENTAS_NAVEGADOR),
                    "nivel": "reversible",
                },
            )
        if not self._servidor.vivo:
            await self._servidor.arrancar()
        permitidas = set(HERRAMIENTAS_NAVEGADOR)
        self.herramientas = [h for h in self._servidor.herramientas if h.get("name") in permitidas]

    async def llamar(self, herramienta: str, argumentos: dict[str, Any]) -> str:
        if self._servidor is None:
            await self.arrancar()
        assert self._servidor is not None
        return await self._servidor.llamar(herramienta, argumentos)

    async def detener(self) -> None:
        if self._servidor is not None:
            await self._servidor.detener()


# --------------------------------------------------------------------------- #
# Lo que ve el modelo
# --------------------------------------------------------------------------- #

#: Lo que la API de Gemini entiende de un esquema JSON. `$schema` y
#: `additionalProperties`, que Playwright pone en todos, los rechaza.
_CLAVES_ESQUEMA = ("type", "description", "enum", "items", "properties", "required", "nullable", "format")


def _esquema_gemini(esquema: Any) -> Any:
    if isinstance(esquema, list):
        return [_esquema_gemini(e) for e in esquema]
    if not isinstance(esquema, dict):
        return esquema
    limpio: dict[str, Any] = {}
    for clave in _CLAVES_ESQUEMA:
        if clave not in esquema:
            continue
        valor = esquema[clave]
        if clave == "properties" and isinstance(valor, dict):
            limpio[clave] = {nombre: _esquema_gemini(sub) for nombre, sub in valor.items()}
        elif clave == "items":
            limpio[clave] = _esquema_gemini(valor)
        else:
            limpio[clave] = valor
    return limpio


#: Lo que el modelo lee de `target`, en vez de lo que dice Playwright («o un
#: selector único»): el agente rechaza los selectores, y anunciárselos es
#: invitarle a probar.
_TARGET = "El ref del elemento en la última instantánea, p. ej. «e12». Nunca un selector."

PROPIAS: list[dict[str, Any]] = [
    {
        "name": "boveda_listar",
        "description": (
            "Las contraseñas y tarjetas guardadas: nombre, tipo, en qué sitios valen y la "
            "referencia de cada campo. NUNCA el valor. Para teclear uno, escribe su referencia "
            "literal ({{boveda:nombre.campo}}) en el `text` o el `value`."
        ),
    },
    {
        "name": "terminar",
        "description": (
            "Da el recado por acabado. `resumen` cuenta qué se hizo: qué, dónde, cuándo, "
            "cuánto y el código de confirmación si lo hay. Si no se pudo, dilo y por qué."
        ),
        "parameters": {
            "type": "object",
            "properties": {"resumen": {"type": "string"}},
            "required": ["resumen"],
        },
    },
    {
        "name": "pedir_ayuda",
        "description": (
            "Para cuando hace falta el señor Persus: un CAPTCHA, un código de verificación, "
            "una sesión que no está iniciada o una decisión que el encargo no aclara. "
            "`pregunta` dice qué hace falta, corto."
        ),
        "parameters": {
            "type": "object",
            "properties": {"pregunta": {"type": "string"}},
            "required": ["pregunta"],
        },
    },
]


def declaraciones(herramientas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Las del navegador que haya, en el dialecto de Gemini, y las propias."""
    salida: list[dict[str, Any]] = []
    for h in herramientas:
        nombre = str(h.get("name") or "")
        if nombre not in HERRAMIENTAS_NAVEGADOR:
            continue
        declaracion: dict[str, Any] = {"name": nombre, "description": str(h.get("description") or "")[:600]}
        esquema = _esquema_gemini(h.get("inputSchema") or {})
        propiedades = esquema.get("properties") if isinstance(esquema, dict) else None
        if propiedades:
            _reescribir_target(esquema)
            declaracion["parameters"] = esquema
        salida.append(declaracion)
    return salida + json.loads(json.dumps(PROPIAS))


def _reescribir_target(esquema: dict[str, Any]) -> None:
    for nombre, sub in (esquema.get("properties") or {}).items():
        if nombre == "target" and isinstance(sub, dict):
            sub["description"] = _TARGET
        elif isinstance(sub, dict):
            if sub.get("type") == "object":
                _reescribir_target(sub)
            elif sub.get("type") == "array" and isinstance(sub.get("items"), dict):
                _reescribir_target(sub["items"])


#: Cuántas respuestas del navegador se quedan enteras en la conversación. Las
#: de antes se recortan: la página de hace diez pasos no sirve y cada petición
#: la volvería a pagar entera.
RESPUESTAS_ENTERAS = 2


# --------------------------------------------------------------------------- #
# La conversación
# --------------------------------------------------------------------------- #


def compactar(contents: list[dict[str, Any]]) -> None:
    """Recorta las respuestas viejas del navegador. Cambia la lista en su sitio."""
    vistas = 0
    for turno in reversed(contents):
        if turno.get("role") != "user":
            continue
        for parte in turno.get("parts") or []:
            respuesta = (parte.get("functionResponse") or {}).get("response")
            if not isinstance(respuesta, dict) or not isinstance(respuesta.get("resultado"), str):
                continue
            if "### Snapshot" not in respuesta["resultado"] and "[ref=" not in respuesta["resultado"]:
                continue
            vistas += 1
            if vistas > RESPUESTAS_ENTERAS and not respuesta.get("recortada"):
                respuesta["resultado"] = respuesta["resultado"][:300] + "\n[… página antigua recortada]"
                respuesta["recortada"] = True


def respuesta_de(llamada: dict[str, Any], texto: str) -> dict[str, Any]:
    parte: dict[str, Any] = {"name": llamada.get("name"), "response": {"resultado": texto}}
    if llamada.get("id"):
        parte["id"] = llamada["id"]
    return {"functionResponse": parte}


def llamadas_de(turno: dict[str, Any]) -> list[dict[str, Any]]:
    return [p["functionCall"] for p in turno.get("parts") or [] if isinstance(p.get("functionCall"), dict)]


def texto_de(turno: dict[str, Any]) -> str:
    return " ".join(str(p.get("text") or "") for p in turno.get("parts") or [] if not p.get("thought")).strip()
