"""Los dos transportes de MCP: un proceso hijo y un servidor de la red.

Salieron de `mcp.py` el 2026-09-12, cuando el fichero pasaba de mil líneas. La
costura: una cosa es **cómo se habla** con un servidor —por tuberías con un
proceso que lanzamos, o por HTTP con uno que ya está— y otra **qué se le pide y
qué se hace con la respuesta**, que se queda en `mcp.py`.

Los dos exponen lo mismo (`listar`, `llamar`, `cerrar`) para que arriba no haya
que saber cuál es cuál.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import subprocess
from typing import Any

from .mcp_argumentos import (
    _normalizar_search_files,
    _acomodar,
    _es_error_de_argumentos,
    _esquema_de,
    _faltan_requeridos,
    _pista_esquema,
    recortar,
)

logger = logging.getLogger(__name__)

#: Dónde se declaran los servidores, dentro del directorio de datos. Se nombra
#: en los errores para que quien los lea sepa qué fichero abrir.
NOMBRE_FICHERO = "mcp.json"


#: Versión del protocolo que le ofrecemos al servidor. La más extendida: todos
#: los servidores oficiales la aceptan, y la respuesta manda — si el servidor
#: vive en otra versión, la suya es la buena.
VERSION_PROTOCOLO = "2024-11-05"

#: Segundos por defecto esperando a un servidor. Un `tools/call` de un
#: navegador headless puede tardar; un servidor muerto no debe colgar un trabajo
#: para siempre. Se ajusta por servidor con `tope_segundos`.
TOPE_POR_DEFECTO = 60.0


#: El entorno que sí viaja a un servidor hijo. Es una **lista blanca** y no el
#: `os.environ` entero a propósito: un servidor MCP es código de terceros, y
#: heredarle el ambiente completo sería regalarle todo lo que haya por ahí —
#: claves de Gemini, tokens, lo que sea. Lo que necesita cualquier lanzador
#: razonable (Node, uvx, Python) es el PATH y las rutas del sistema; lo demás,
#: cada servidor lo declara en su `env` del fichero, que lo escribe una persona.
_CLAVES_ENV_HEREDADAS = (
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "SystemRoot",
    "WINDIR",
    "COMSPEC",
    "TEMP",
    "TMP",
    "HOME",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    "APPDATA",
    "LOCALAPPDATA",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "COMMONPROGRAMFILES",
    "COMPUTERNAME",
    "USERNAME",
    "OS",
)

_ERRORES = (-32601,)  # method not found: respuesta educada a lo que pidan


class ErrorMcp(RuntimeError):
    """El servidor no contestó, no existe o rechazó la llamada."""


# -- Configuración ----------------------------------------------------------- #


def _entorno_hijo(extra: dict[str, str]) -> dict[str, str]:
    """El entorno de un servidor hijo: lista blanca + lo que declare su fichero.

    Ver `_CLAVES_ENV_HEREDADAS` para el porqué. Las claves del `env` del
    servidor **mandan** sobre las heredadas, igual que antes.
    """
    entorno = {k: v for k in _CLAVES_ENV_HEREDADAS if (v := os.environ.get(k))}
    entorno.update(extra)
    return entorno


def _url_aceptable(url: str) -> bool:
    """Solo `https://`, o `http://` contra el bucle local.

    Un servidor MCP remoto recibe los argumentos que compone Perseo, y esos
    argumentos vienen de correos y de pantallas. En claro por una red que no es
    la de casa, no.
    """
    from urllib.parse import urlparse

    partes = urlparse(url)
    if partes.scheme == "https":
        return True
    return partes.scheme == "http" and (partes.hostname or "") in ("127.0.0.1", "localhost", "::1")


def _hay_sdk_mcp() -> bool:
    """¿Está el SDK oficial de MCP? Solo hace falta para los remotos."""
    import importlib.util

    return importlib.util.find_spec("mcp") is not None


def _preparar_llamada(
    nombre: str,
    herramientas: list[dict[str, Any]],
    permitidas: set[str],
    herramienta: str,
    argumentos: dict[str, Any],
    por_defecto: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Lo que hay que comprobar antes de que una llamada salga de casa.

    Devuelve el esquema de la herramienta y los argumentos ya acomodados, o
    revienta con `ErrorMcp` diciendo qué falta. Los dos transportes —el proceso
    hijo por stdio y el servidor remoto por HTTP— hacían exactamente esto, cada
    uno con su copia de veinte líneas: existe la herramienta, está permitida,
    encajan los argumentos. Dos copias de una comprobación de seguridad son la
    forma más fácil de que un día solo una de las dos se entere de algo.
    """
    if herramienta not in {h.get("name") for h in herramientas}:
        disponibles = ", ".join(sorted(str(h.get("name")) for h in herramientas))
        raise ErrorMcp(
            f"'{nombre}' no tiene ninguna herramienta '{herramienta}'. Tiene: {disponibles}"
        )
    if permitidas and herramienta not in permitidas:
        raise ErrorMcp(f"'{herramienta}' no está en la lista de '{nombre}' en {NOMBRE_FICHERO}.")

    esquema = _esquema_de(herramientas, herramienta)
    argumentos = _acomodar(argumentos, esquema, por_defecto)
    # Si el modelo llama a search_files del vault con lenguaje natural en vez de
    # un glob, se convierte para que no falle con un -32602.
    if nombre == "vault" and herramienta == "search_files":
        argumentos = _normalizar_search_files(argumentos)
    faltan = _faltan_requeridos(argumentos, esquema)
    if faltan:
        # El viaje se ahorra: el servidor iba a contestar -32602 y el modelo se
        # iba a quedar sin saber cómo se llaman los campos.
        raise ErrorMcp(
            f"A '{nombre}.{herramienta}' le faltan argumentos: "
            + ", ".join(faltan)
            + _pista_esquema(esquema, herramienta)
        )
    return esquema, argumentos

class ServidorMcpRemoto:
    """Un servidor MCP que vive en otra máquina, hablado por HTTP.

    Aquí sí se usa el **SDK oficial** (`mcp`) y no el cliente de casa, y por un
    motivo concreto: el de casa habla JSON-RPC por las tuberías de un proceso
    hijo, que es un transporte entero distinto. Reimplementar *streamable HTTP*
    —con su sesión, sus reintentos y su SSE— sería escribir por segunda vez
    algo que ya está escrito y probado. Lo de stdio se queda como está: funciona
    y lleva dentro decisiones nuestras (entorno recortado, cerrojo por
    servidor) que no se regalan a cambio de nada.

    **Una sesión por llamada**, a propósito. Mantenerla abierta obliga a entrar
    y salir del contexto asíncrono desde la misma tarea, y aquí las llamadas
    vienen del trabajador y el cierre viene del apagado — dos tareas. Pagar un
    saludo por llamada es más barato que un cierre que revienta al apagar.
    """

    def __init__(self, nombre: str, definicion: dict[str, Any]) -> None:
        self.nombre = nombre
        self.url: str = definicion["url"]
        self.cabeceras: dict[str, str] = definicion.get("cabeceras") or {}
        self.tope = float(definicion["tope_segundos"])
        self.definicion = definicion
        self.herramientas: list[dict[str, Any]] = []
        #: Un remoto no tiene proceso que se muera: se da por vivo en cuanto
        #: se le ha preguntado una vez por sus herramientas.
        self.vivo = False

    def _permitidas(self) -> set[str]:
        return {str(h) for h in (self.definicion.get("herramientas") or [])}

    @contextlib.asynccontextmanager
    async def _sesion(self):
        """Una conversación abierta con el servidor remoto, y cerrada al salir.

        Las cabeceras —el testigo del servidor, casi siempre— viajan en el
        cliente HTTP, que es donde el SDK deja ponerlas.
        """
        import httpx2
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        async with httpx2.AsyncClient(
            headers=self.cabeceras or None, timeout=self.tope
        ) as http:
            async with streamable_http_client(self.url, http_client=http) as (leer, escribir):
                async with ClientSession(leer, escribir) as sesion:
                    await sesion.initialize()
                    yield sesion

    async def arrancar(self) -> None:
        """Saluda y se queda con el catálogo. Es lo único que hay que 'arrancar'."""
        if not _hay_sdk_mcp():
            raise ErrorMcp(
                f"'{self.nombre}' es un servidor remoto y hace falta el SDK de "
                "MCP para hablarlo: pip install mcp"
            )
        try:
            async with asyncio.timeout(self.tope):
                async with self._sesion() as sesion:
                    catalogo = await sesion.list_tools()
        except TimeoutError:
            raise ErrorMcp(f"'{self.nombre}' no contestó en {self.tope:.0f} s.") from None
        except Exception as e:  # noqa: BLE001 - la red falla de mil maneras
            raise ErrorMcp(f"'{self.nombre}' no contestó ({recortar(str(e), 160)}).") from None

        self.herramientas = [
            {"name": h.name, "description": h.description or "", "inputSchema": h.input_schema}
            for h in catalogo.tools
        ]
        self.vivo = True

    async def detener(self) -> None:
        """No hay nada que cerrar: cada llamada abrió y cerró lo suyo."""
        self.vivo = False

    async def llamar(self, herramienta: str, argumentos: dict[str, Any]) -> str:
        esquema, argumentos = _preparar_llamada(
            self.nombre,
            self.herramientas,
            self._permitidas(),
            herramienta,
            argumentos,
            self._por_defecto(herramienta),
        )

        try:
            async with asyncio.timeout(self.tope):
                async with self._sesion() as sesion:
                    resultado = await sesion.call_tool(herramienta, argumentos)
        except TimeoutError:
            raise ErrorMcp(f"'{self.nombre}.{herramienta}' pasó de {self.tope:.0f} s.") from None

        textos = [str(getattr(b, "text", "")) for b in (resultado.content or []) if getattr(b, "text", "")]
        if resultado.is_error:
            error_msg = recortar(" ".join(textos)) or "el servidor devolvió un error"
            if _es_error_de_argumentos(error_msg):
                error_msg += _pista_esquema(esquema, herramienta)
            raise ErrorMcp(error_msg)
        return "\n".join(textos).strip()

    def _por_defecto(self, herramienta: str) -> dict[str, Any]:
        valores = (self.definicion.get("argumentos_por_defecto") or {}).get(herramienta)
        return valores if isinstance(valores, dict) else {}


def abrir_servidor(nombre: str, definicion: dict[str, Any]):
    """El servidor que toque: proceso hijo por stdio, o dirección por HTTP."""
    if definicion.get("url"):
        return ServidorMcpRemoto(nombre, definicion)
    return ServidorMcp(nombre, definicion)


# -- El cliente -------------------------------------------------------------- #


class ServidorMcp:
    """Un proceso hijo hablando JSON-RPC por stdio, una línea por mensaje."""

    def __init__(self, nombre: str, definicion: dict[str, Any]) -> None:
        self.nombre = nombre
        #: Lo que el fichero dice de este servidor. Se guarda entero porque de
        #: aquí salen los argumentos por defecto y la lista de herramientas
        #: permitidas, que antes se miraban en una global del módulo de arriba —
        #: y una capa de abajo leyendo una variable de la de arriba es justo lo
        #: que la partición viene a quitar.
        self.definicion = definicion
        self.comando: list[str] = definicion["comando"]
        self.entorno = definicion["env"]
        self.tope = float(definicion["tope_segundos"])
        self.proceso: asyncio.subprocess.Process | None = None
        self.herramientas: list[dict[str, Any]] = []
        self._contador = 0
        # Una conversación a la vez. El stdin/stdout es un único canal de
        # líneas compartido: sin este cerrojo, dos llamadas entrelazadas se
        # leen la respuesta la una a la otra — cada `_pedir` salta las líneas
        # cuyo id no es el suyo, así que la respuesta ajena se pierde y la otra
        # llamada acaba a plazo muerto. No reentrante a propósito: quien ya lo
        # tiene llama a `_arrancar_bajo_cerrojo`, nunca a `arrancar`.
        self._cerrojo = asyncio.Lock()

    # -- ciclo de vida ------------------------------------------------------ #

    @property
    def vivo(self) -> bool:
        return self.proceso is not None and self.proceso.returncode is None

    async def arrancar(self) -> None:
        """Lanza el proceso y completa el saludo del protocolo."""
        async with self._cerrojo:
            await self._arrancar_bajo_cerrojo()

    async def _arrancar_bajo_cerrojo(self) -> None:
        # Windows otra vez: `npx`, `uvx` y compañía son `.cmd`, y sin shell el
        # ejecutable desnudo no se encuentra. Mismo remedio que en
        # `proyectos.py`: `shutil.which`, que es lo que los encuentra.
        ejecutable = shutil.which(self.comando[0]) or self.comando[0]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proceso = await asyncio.create_subprocess_exec(
            ejecutable,
            *self.comando[1:],
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=_entorno_hijo(self.entorno),
            creationflags=flags,
        )
        try:
            saludo = await self._pedir(
                "initialize",
                {
                    "protocolVersion": VERSION_PROTOCOLO,
                    "capabilities": {},
                    "clientInfo": {"name": "perseo", "version": "2.0"},
                },
            )
            version_del_servidor = str((saludo or {}).get("protocolVersion") or VERSION_PROTOCOLO)
            logger.info("MCP '%s': saludo aceptado (protocolo %s).", self.nombre, version_del_servidor)
            # Aviso obligatorio tras el initialize; sin él, el servidor no empieza.
            await self._avisar("notifications/initialized")
            listado = await self._pedir("tools/list", {})
        except BaseException:
            # Un saludo que falla deja un proceso vivo con sus tuberías abiertas
            # y nadie hablándole. Sin esta limpieza, cada intento fallido gotea
            # un proceso y unos transportes que nadie cierra.
            await self.detener()
            raise
        self.herramientas = [
            h for h in ((listado or {}).get("tools") or []) if isinstance(h, dict)
        ]
        logger.info("MCP '%s': %d herramienta(s).", self.nombre, len(self.herramientas))

    async def detener(self) -> None:
        proceso = self.proceso
        # Primero soltar la referencia: un detener llamado desde otro bucle
        # (las pruebas crean uno por llamada) no debe volver a tocar este.
        self.proceso = None
        self.herramientas = []
        if proceso is None or proceso.returncode is not None:
            return
        try:
            proceso.terminate()
        except ProcessLookupError:
            return
        # Cerrar las tuberías aquí y no dejarlo al destructor del transporte:
        # un proceso muerto con su bucle ya cerrado suelta avisos feos al salir.
        # El stdin es un StreamWriter (tiene close); el stdout, un StreamReader
        # que solo expone el transporte.
        for extremo in (proceso.stdin, proceso.stdout):
            if extremo is None:
                continue
            try:
                if hasattr(extremo, "close"):
                    extremo.close()
                elif getattr(extremo, "transport", None) is not None:
                    extremo.transport.close()
            except (OSError, RuntimeError):
                pass
        try:
            await asyncio.wait_for(proceso.wait(), timeout=5)
        except (ProcessLookupError, asyncio.TimeoutError, RuntimeError):
            # El RuntimeError es el cambio de bucle: el terminate ya salió y con
            # él basta — en Windows es TerminateProcess, no una petición educada.
            try:
                proceso.kill()
            except (ProcessLookupError, RuntimeError):
                pass

    # -- operaciones -------------------------------------------------------- #

    async def llamar(self, herramienta: str, argumentos: dict[str, Any]) -> str:
        """`tools/call`. Devuelve el texto que trae la respuesta.

        Va bajo el cerrojo del servidor: aunque hoy un solo trabajador ejecute
        los trabajos de `mcp` en fila, el día que haya dos carriles o una
        llamada y un trabajo a la vez, las dos conversaciones no pueden
        entrelazarse sobre las mismas tuberías.
        """
        async with self._cerrojo:
            if not self.vivo:
                # El servidor pudo morirse con el trabajo anterior. Respawn
                # perezoso y una sola vez por llamada: si vuelve a fallar, que
                # falle ruido.
                await self.detener()
                await self._arrancar_bajo_cerrojo()

            esquema, argumentos = _preparar_llamada(
                self.nombre,
                self.herramientas,
                self._permitidas(),
                herramienta,
                argumentos,
                self._por_defecto(herramienta),
            )

            try:
                resultado = await self._pedir(
                    "tools/call", {"name": herramienta, "arguments": argumentos}
                )
            except ErrorMcp as e:
                # Un rechazo por argumentos llega como error de JSON-RPC, no
                # dentro del resultado: sin esto, la pista del esquema no se
                # añadía nunca justo cuando más falta hace.
                if not _es_error_de_argumentos(str(e)):
                    raise
                raise ErrorMcp(str(e) + _pista_esquema(esquema, herramienta)) from None
        contenido = (resultado or {}).get("content") or []
        textos = [
            str(bloque.get("text", ""))
            for bloque in contenido
            if isinstance(bloque, dict) and bloque.get("type") == "text"
        ]
        if (resultado or {}).get("isError"):
            error_msg = recortar(" ".join(t for t in textos if t)) or "el servidor devolvió un error"
            if _es_error_de_argumentos(error_msg):
                error_msg += _pista_esquema(_esquema_de(self.herramientas, herramienta), herramienta)
            raise ErrorMcp(error_msg)
        return "\n".join(t for t in textos if t).strip()

    def _por_defecto(self, herramienta: str) -> dict[str, Any]:
        """Lo que el fichero ponga por esa herramienta cuando el modelo calle."""
        valores = (self.definicion.get("argumentos_por_defecto") or {}).get(herramienta)
        return valores if isinstance(valores, dict) else {}

    # -- JSON-RPC ------------------------------------------------------------ #

    async def _pedir(self, metodo: str, parametros: dict[str, Any]) -> Any | None:
        """Una petición con respuesta, con plazo. Relanza el proceso si murió."""
        assert self.proceso is not None and self.proceso.stdin and self.proceso.stdout
        self._contador += 1
        identificador = self._contador
        linea = json.dumps(
            {"jsonrpc": "2.0", "id": identificador, "method": metodo, "params": parametros},
            ensure_ascii=False,
        )
        try:
            self.proceso.stdin.write(linea.encode("utf-8") + b"\n")
            await self.proceso.stdin.drain()
            limite = asyncio.get_running_loop().time() + self.tope
            while True:
                restante = limite - asyncio.get_running_loop().time()
                if restante <= 0:
                    # Un servidor que no contesta queda tonto para siempre: se
                    # mata aquí para que la siguiente llamada lo arranque de
                    # nuevo en vez de hablarle a un proceso colgado.
                    await self.detener()
                    raise ErrorMcp(
                        f"'{self.nombre}' tardó más de {self.tope:.0f} s en contestar a {metodo}."
                    )
                try:
                    bruto = await asyncio.wait_for(
                        self.proceso.stdout.readline(), timeout=restante
                    )
                except asyncio.TimeoutError:
                    await self.detener()
                    raise ErrorMcp(
                        f"'{self.nombre}' tardó más de {self.tope:.0f} s en contestar a {metodo}."
                    ) from None
                if not bruto:
                    raise ErrorMcp(f"'{self.nombre}' cerró su salida durante {metodo}.")
                mensaje = json.loads(bruto.decode("utf-8", errors="replace"))
                if mensaje.get("id") != identificador:
                    continue  # notificación o petición ajena: se ignora aquí
                if "error" in mensaje:
                    detalle = (mensaje["error"] or {}).get("message", "sin detalle")
                    raise ErrorMcp(f"'{self.nombre}' rechazó {metodo}: {detalle}")
                return mensaje.get("result")
        except json.JSONDecodeError as e:
            raise ErrorMcp(f"'{self.nombre}' dijo algo que no es JSON ({e}).") from e

    async def _avisar(self, metodo: str) -> None:
        """Una notificación: no lleva id y nadie contesta."""
        assert self.proceso is not None and self.proceso.stdin
        linea = json.dumps({"jsonrpc": "2.0", "method": metodo}, ensure_ascii=False)
        self.proceso.stdin.write(linea.encode("utf-8") + b"\n")
        await self.proceso.stdin.drain()

    def _permitidas(self) -> set[str]:
        return set(self.definicion.get("herramientas") or [])
