"""Agente `mcp`: los servidores MCP como un puerto más.

MCP (Model Context Protocol) es una convención para que un programa le ofrezca
herramientas a otro: el cliente lanza al servidor como proceso hijo y le habla
por su entrada y salida estándar con **JSON-RPC 2.0, una línea por mensaje**.
Nada más. No hace falta SDK oficial — que arrastra httpx y compañía, cuando lo
que hay que hacer es escribir líneas de JSON y leerlas — y así el núcleo sigue
con su única dependencia.

Por qué esto entra en Perseo y cómo convive con lo que ya había:

1. **Es un puerto, igual que el buzón o el calendario.** Los agentes propios
   (`memoria`, `pc`, `web`) se quedan donde están: llevan reglas de seguridad
   que ningún servidor de terceros trae — memoria no borra, `pc` no tiene shell,
   `web` no alcanza la red de casa. MCP llega AL LADO para extender, y cada
   equivalente de terceros tendrá que ganarse el puesto usándose.
2. **La política manda aquí dentro también.** Cada servidor lleva su nivel en
   `<datos>/mcp.json` — `libre`, `reversible` o `irreversible` — y lo desconocido
   es irreversible, que desde el 2026-08-22 cuesta un «sí» hablado y no un clic.
   Además una lista de herramientas permitidas por servidor: si el día de mañana
   `server-filesystem` estrena `delete_everything`, no pasa nada hasta que alguien
   la escriba en esa lista.
3. **Sin shell, otra vez.** El comando del servidor va en lista de argumentos,
   nunca como cadena, y no se interpola nada del modelo en él. Lo que puede
   ejecutarse lo escribió una persona en el fichero del disco, como en
   `proyectos.json`.


"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..infra import politica
from ..infra.router import registrar
from ..infra.configuracion import Configuracion

logger = logging.getLogger(__name__)

#: Versión del protocolo que le ofrecemos al servidor. La más extendida: todos
#: los servidores oficiales la aceptan, y la respuesta manda — si el servidor
#: vive en otra versión, la suya es la buena.
VERSION_PROTOCOLO = "2024-11-05"

#: Segundos por defecto esperando a un servidor. Un `tools/call` de un
#: navegador headless puede tardar; un servidor muerto no debe colgar un trabajo
#: para siempre. Se ajusta por servidor con `tope_segundos`.
TOPE_POR_DEFECTO = 60.0

NOMBRE_FICHERO = "mcp.json"

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


def cargar_servidores(directorio_datos: Path) -> dict[str, dict[str, Any]]:
    """Lo que haya en `<datos>/mcp.json`, validado.

    Una entrada mala se descarta con aviso, como en `proyectos.py`: el fichero
    lo escribe una persona y perder los cinco servidores buenos por una coma es
    peor que perder el malo.
    """
    ruta = Path(directorio_datos) / NOMBRE_FICHERO
    if not ruta.exists():
        return {}
    try:
        crudo = json.loads(ruta.read_text(encoding="utf-8") or "{}")
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("No se pudo leer %s (%s); se sigue sin MCP.", ruta.name, e)
        return {}
    if not isinstance(crudo, dict):
        logger.warning("%s debería ser un objeto de servidores.", NOMBRE_FICHERO)
        return {}

    validos: dict[str, dict[str, Any]] = {}
    for nombre, entrada in crudo.items():
        if not isinstance(entrada, dict):
            logger.warning("Servidor %r mal escrito; se ignora.", nombre)
            continue
        url = str(entrada.get("url") or "").strip()
        comando = entrada.get("comando")
        if url:
            # Servidor REMOTO: no hay proceso hijo, hay una dirección. Se exige
            # HTTPS salvo en el bucle local, porque por ahí van a viajar los
            # argumentos que Perseo compone con lo que ve en pantalla.
            if not _url_aceptable(url):
                logger.warning(
                    "Servidor %r con url %r: solo https:// (o http:// en "
                    "127.0.0.1) ; se ignora.",
                    nombre,
                    url,
                )
                continue
            comando = []
        elif not isinstance(comando, list) or not all(isinstance(p, str) for p in comando):
            logger.warning(
                "Servidor %r sin 'comando' como lista de argumentos ni 'url'; se "
                "ignora. Una cadena nunca vale como comando: por ahí entran las "
                "comillas y los &&.",
                nombre,
            )
            continue
        nivel = str(entrada.get("nivel") or politica.IRREVERSIBLE).strip().lower()
        if nivel not in politica.NIVELES:
            logger.warning("Servidor %r con nivel %r raro; irreversible.", nombre, nivel)
            nivel = politica.IRREVERSIBLE
        herramientas = entrada.get("herramientas") or []
        if not isinstance(herramientas, list) or not all(isinstance(h, str) for h in herramientas):
            herramientas = []
        entorno = entrada.get("env") or {}
        if not isinstance(entorno, dict):
            entorno = {}
        try:
            tope = float(entrada.get("tope_segundos") or TOPE_POR_DEFECTO)
        except (TypeError, ValueError):
            tope = TOPE_POR_DEFECTO
        cabeceras = entrada.get("cabeceras") or {}
        if not isinstance(cabeceras, dict):
            cabeceras = {}
        # Lo que el modelo no puede saber y el servidor exige: la raíz del vault
        # en `search_files`, y lo que venga. Por herramienta, y solo rellena
        # hueco vacío — lo que el modelo diga, manda.
        por_defecto = entrada.get("argumentos_por_defecto") or {}
        if not isinstance(por_defecto, dict) or not all(
            isinstance(v, dict) for v in por_defecto.values()
        ):
            logger.warning(
                "Servidor %r: 'argumentos_por_defecto' debe ser herramienta -> objeto; se ignora.",
                nombre,
            )
            por_defecto = {}
        # El nivel fino: una herramienta que solo mira no tiene por qué heredar
        # el nivel del servidor que además toca el sistema. Lo raro se descarta
        # y esa herramienta vuelve al nivel del servidor, que es el prudente.
        niveles_herramienta = entrada.get("niveles_herramienta") or {}
        if not isinstance(niveles_herramienta, dict):
            niveles_herramienta = {}
        finos: dict[str, str] = {}
        for clave, valor in niveles_herramienta.items():
            fino = str(valor or "").strip().lower()
            if fino in politica.NIVELES:
                finos[str(clave)] = fino
            else:
                logger.warning(
                    "Servidor %r: nivel %r raro para %r; se queda con el del servidor.",
                    nombre,
                    valor,
                    clave,
                )
        validos[str(nombre)] = {
            "niveles_herramienta": finos,
            "argumentos_por_defecto": {str(k): dict(v) for k, v in por_defecto.items()},
            "comando": [str(p) for p in comando],
            "url": url,
            "cabeceras": {str(k): str(v) for k, v in cabeceras.items()},
            "nivel": nivel,
            "herramientas": herramientas,
            "env": {str(k): str(v) for k, v in entorno.items()},
            "tope_segundos": max(5.0, min(tope, 600.0)),
        }
    return validos


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
        definicion = definiciones.get(self.nombre) or {}
        valores = (definicion.get("argumentos_por_defecto") or {}).get(herramienta)
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
        definicion = definiciones.get(self.nombre) or {}
        return set(definicion.get("herramientas") or [])


# -- Los argumentos, antes de salir ------------------------------------------ #

#: Lo que el modelo escribe cuando no ha mirado el esquema. Un servidor MCP
#: nombra sus parámetros en inglés; Perseo piensa en español y ese idioma se le
#: cuela hasta la llamada — de ahí un `{"comando": ...}` contra un `PowerShell`
#: que espera `command`, y una llamada perdida por una palabra. La traducción
#: solo entra si el esquema tiene el nombre bueno y la llamada no lo traía ya:
#: nunca inventa un campo ni pisa lo que el modelo escribió bien.
_ALIAS_ARGUMENTOS: dict[str, tuple[str, ...]] = {
    "comando": ("command",),
    "orden": ("command",),
    "ruta": ("path",),
    "archivo": ("path",),
    "fichero": ("path",),
    "carpeta": ("path",),
    "directorio": ("path",),
    "destino": ("destination",),
    "patron": ("pattern",),
    "patrón": ("pattern",),
    "busqueda": ("pattern", "query"),
    "búsqueda": ("pattern", "query"),
    "consulta": ("query", "pattern"),
    "texto": ("text",),
    "contenido": ("content", "text"),
    "titulo": ("title",),
    "título": ("title",),
    "mensaje": ("message",),
    "modo": ("mode",),
    "atajo": ("shortcut",),
    "duracion": ("duration",),
    "duración": ("duration",),
    "zona_horaria": ("timezone",),
    "nombre": ("name",),
    "condicion": ("condition",),
    "condición": ("condition",),
}


def _es_error_de_argumentos(mensaje: str) -> bool:
    """¿El servidor rechazó la llamada por los parámetros, y no por otra cosa?"""
    bajo = mensaje.lower()
    return (
        "32602" in bajo
        or "invalid arguments" in bajo
        or "input validation" in bajo
        or "validation error" in bajo
        or "missing required argument" in bajo
    )


def _esquema_de(herramientas: list[dict[str, Any]], nombre: str) -> dict[str, Any]:
    """El `inputSchema` que el servidor publicó para esa herramienta."""
    for h in herramientas:
        if str(h.get("name")) == nombre:
            esquema = h.get("inputSchema")
            return esquema if isinstance(esquema, dict) else {}
    return {}


def _propiedades(esquema: dict[str, Any]) -> dict[str, Any]:
    props = (esquema or {}).get("properties")
    return props if isinstance(props, dict) else {}


def _requeridos(esquema: dict[str, Any]) -> list[str]:
    req = (esquema or {}).get("required")
    return [str(k) for k in req] if isinstance(req, list) else []


def _pista_esquema(esquema: dict[str, Any], herramienta: str) -> str:
    """Los parámetros de una herramienta, en prosa corta, para el modelo.

    Va pegada a cualquier error de argumentos: quien se equivocó de nombre lee
    ahí mismo cómo se llaman de verdad y reintenta bien, en vez de repetir el
    mismo fallo hasta que alguien se rinde.
    """
    props = _propiedades(esquema)
    if not props:
        return ""
    requeridos = _requeridos(esquema)
    lineas = []
    for clave, valor in props.items():
        detalle = valor if isinstance(valor, dict) else {}
        tipo = detalle.get("type") or "?"
        marca = "requerido" if clave in requeridos else "opcional"
        descripcion = recortar(str(detalle.get("description") or ""), 80)
        lineas.append(f"  - {clave}: {tipo} ({marca}){' — ' + descripcion if descripcion else ''}")
    return f"\n\nParámetros de '{herramienta}':\n" + "\n".join(lineas)


def _acomodar(
    argumentos: dict[str, Any],
    esquema: dict[str, Any],
    por_defecto: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Los argumentos del modelo, puestos en los nombres que el servidor espera.

    Dos arreglos y ninguno más: traducir el nombre español al del esquema, y
    poner lo que el fichero declare por defecto para esa herramienta (la ruta
    del vault, por ejemplo, que el modelo nunca sabe y el servidor exige).
    """
    salida = dict(argumentos)
    props = _propiedades(esquema)
    if props:
        for clave in list(salida):
            if clave in props:
                continue
            for candidato in _ALIAS_ARGUMENTOS.get(str(clave).lower(), ()):
                if candidato in props and candidato not in salida:
                    salida[candidato] = salida.pop(clave)
                    break
    for clave, valor in (por_defecto or {}).items():
        if salida.get(clave) in (None, ""):
            salida[clave] = valor
    return salida


def _faltan_requeridos(argumentos: dict[str, Any], esquema: dict[str, Any]) -> list[str]:
    """Los requeridos que no vienen. Mejor decirlo aquí que gastar un viaje."""
    if not _propiedades(esquema):
        return []
    return [k for k in _requeridos(esquema) if argumentos.get(k) in (None, "")]


def _normalizar_search_files(args: dict[str, Any]) -> dict[str, Any]:
    """
    Convierte lenguaje natural en glob pattern para search_files del vault.
    El servidor MCP server-filesystem espera glob patterns (ej: *música*.md),
    no texto libre. Si el modelo manda palabras sueltas, las envolvemos.
    """
    args = dict(args)  # copia
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern.strip():
        return args

    # Ya parece un glob (contiene *, ?, [, ], {, })
    if any(c in pattern for c in "*?[]{"):
        return args

    # Lenguaje natural: envolvemos en *...* y añadimos .md si no tiene extensión
    palabras = pattern.strip().split()
    if len(palabras) == 1:
        base = palabras[0]
    else:
        # Múltiples palabras: probamos la más larga (más específica)
        base = max(palabras, key=len)

    # Sin añadir extensión: un `*musica.md*` solo casa con quien lleve
    # «musica.md» dentro del nombre, que no es ningún fichero. `*musica*` casa
    # con «Musica.md» y con «lista de musica.txt», que es lo que se buscaba.
    args["pattern"] = f"*{base}*"
    return args


def _parametros(herramienta: dict[str, Any]) -> list[dict[str, Any]]:
    """Los parámetros de una herramienta, tal como el catálogo los enseña."""
    esquema = herramienta.get("inputSchema")
    esquema = esquema if isinstance(esquema, dict) else {}
    requeridos = _requeridos(esquema)
    salida = []
    for clave, valor in _propiedades(esquema).items():
        detalle = valor if isinstance(valor, dict) else {}
        salida.append(
            {
                "nombre": str(clave),
                "tipo": str(detalle.get("type") or "?"),
                "requerido": clave in requeridos,
                "descripcion": recortar(str(detalle.get("description") or ""), 120),
            }
        )
    return salida


def _firma(herramienta: dict[str, Any]) -> str:
    """`nombre(requerido, [opcional])`, que es lo que el modelo necesita leer."""
    partes = [
        p["nombre"] if p["requerido"] else f"[{p['nombre']}]" for p in _parametros(herramienta)
    ]
    nombre = str(herramienta.get("name"))
    descripcion = recortar(str(herramienta.get("description") or ""), 100)
    firma = f"{nombre}({', '.join(partes)})"
    return f"{firma} — {descripcion}" if descripcion else firma


def recortar(texto: str, tope: int = 300) -> str:
    limpio = " ".join(str(texto).split())
    return limpio if len(limpio) <= tope else limpio[: tope - 1].rstrip() + "…"


# -- Ciclo de vida del módulo ------------------------------------------------ #

#: Lo que hay en `mcp.json`, cargado al arrancar el núcleo.
definiciones: dict[str, dict[str, Any]] = {}
#: Los procesos vivos, uno por servidor configurado.
_activos: dict[str, ServidorMcp] = {}


async def iniciar(cfg: Configuracion) -> None:
    """Carga la configuración y registra el nivel de cada servidor en la política.

    No arranca ningún proceso aquí: los servidores se lanzan la primera vez que
    alguien los usa. Arrancar de más sería tener cinco procesos hijos vivos para
    nada — y un núcleo que arranca sin MCP no debe pagar ni un milisegundo.
    """
    global definiciones
    definiciones = cargar_servidores(cfg.directorio_datos)
    politica.registrar_niveles(nivel_de)
    if definiciones:
        logger.info("MCP: %d servidor(es) configurado(s): %s", len(definiciones), ", ".join(definiciones))


async def detener() -> None:
    global definiciones
    for activo in _activos.values():
        await activo.detener()
    _activos.clear()
    definiciones = {}
    politica.registrar_niveles(None)


def _servidor(nombre: str) -> ServidorMcp:
    """El servidor vivo, arrancándolo si es la primera vez."""
    activo = _activos.get(nombre)
    if activo is not None and activo.vivo:
        return activo
    definicion = definiciones.get(nombre)
    if definicion is None:
        raise ErrorMcp(
            f"No hay ningún servidor MCP llamado '{nombre}'. Configurados: "
            + (", ".join(sorted(definiciones)) or "ninguno")
        )
    nuevo = abrir_servidor(nombre, definicion)
    return nuevo  # lo arranca quien llama, que sabe esperar


# -- La política por servidor ------------------------------------------------ #


def nivel_de(agente: str, peticion: dict[str, Any] | None) -> str | None:
    """El nivel del servidor que pide esta petición, o `None` si no es cosa nuestra.

    Lo consulta `politica.nivel` cuando su tabla no dice nada. El nivel viene del
    fichero que escribió una persona; lo que no esté ahí es irreversible por el
    camino de siempre.
    """
    if agente != "mcp":
        return None
    nombre = str((peticion or {}).get("servidor") or "").strip()
    definicion = definiciones.get(nombre)
    if definicion is None:
        return None  # servidor desconocido: cae en el irreversible por defecto

    herramienta = str((peticion or {}).get("herramienta") or "").strip()
    # Un servidor entero no es un nivel: `windows` tiene un `Snapshot` que solo
    # mira y un `Registry` que toca el sistema, y ponerle un único nivel a los
    # dos obliga a elegir entre preguntar por mirar o no preguntar por escribir.
    por_herramienta = definicion.get("niveles_herramienta") or {}
    if herramienta in por_herramienta:
        return str(por_herramienta[herramienta])

    if nombre == "windows" and herramienta == "PowerShell":
        argumentos = (peticion or {}).get("argumentos")
        comando = (argumentos or {}).get("command") if isinstance(argumentos, dict) else ""
        if _powershell_solo_lee(str(comando or "")):
            # Listar el escritorio no es una acción irreversible, y pararla para
            # pedir un sí a quien acaba de pedirla de viva voz sobra. Escribir,
            # borrar o instalar sigue costando su sí.
            return politica.REVERSIBLE

    return str(definicion["nivel"])


#: Lo que en PowerShell solo mira. La convención del lenguaje ayuda —`Get-*` lee,
#: `Remove-*` no— pero no basta: aquí están además los alias de toda la vida y
#: los verbos que solo dan forma a lo que ya salió (`Select-Object`, `Format-*`).
_POWERSHELL_LECTURA = frozenset(
    {
        "dir", "ls", "gci", "cat", "gc", "type", "pwd", "gl", "echo", "cd",
        "test-path", "resolve-path", "split-path", "join-path", "convert-path",
        "select-object", "select", "sort-object", "sort", "where-object", "where",
        "measure-object", "measure", "format-table", "ft", "format-list", "fl",
        "out-string", "convertto-json", "convertfrom-json", "group-object", "group",
        "select-string", "compare-object", "write-output", "write-host", "foreach-object",
    }
)

#: Lo que parte un comando en dos: una tubería, un `;`, una subexpresión, una
#: redirección. Cada trozo se juzga por separado, porque `Get-ChildItem |
#: Remove-Item` empieza leyendo y acaba borrando.
_SEPARADORES = ("|", ";", "&&", "||", "\n", "\r")


def _powershell_solo_lee(comando: str) -> bool:
    """¿Este comando solo mira? Ante la duda, no.

    No es un analizador de PowerShell y no pretende serlo: es una lista de lo
    que se reconoce como lectura, y todo lo demás cae del lado que pregunta. Un
    falso negativo cuesta un «sí»; un falso positivo dejaría borrar sin avisar.
    """
    texto = comando.strip()
    if not texto:
        return False
    # Una redirección escribe un fichero, y una subexpresión o un acento grave
    # esconden otro comando dentro. Nada de eso pasa por aquí.
    if any(c in texto for c in ">`$"):
        return False

    trozos = [texto]
    for separador in _SEPARADORES:
        trozos = [parte for trozo in trozos for parte in trozo.split(separador)]

    for trozo in trozos:
        palabras = trozo.strip().split()
        if not palabras:
            continue
        verbo = palabras[0].strip("(").lower()
        if verbo in _POWERSHELL_LECTURA:
            continue
        # `Get-`, `Show-`, `Find-` y `Test-` son los verbos de lectura de
        # PowerShell; `Get-Credential` es la excepción y se queda fuera.
        if verbo.startswith(("get-", "show-", "find-", "test-")) and verbo != "get-credential":
            continue
        return False
    return True


# -- El agente --------------------------------------------------------------- #


@registrar("mcp")
async def _mcp(trabajo: dict[str, Any]) -> dict[str, Any]:
    """Lista los servidores y sus herramientas, o llama a una herramienta."""
    peticion = trabajo.get("peticion") or {}
    accion = str(peticion.get("accion", "")).strip().lower()

    if accion == "servidores":
        return await _listar()

    if accion == "llamar":
        return await _llamar(peticion)

    raise ValueError(f"Acción desconocida para mcp: {accion!r}. Válidas: servidores, llamar.")


async def _listar() -> dict[str, Any]:
    if not definiciones:
        return {
            "texto": "No hay servidores MCP configurados.",
            "titular": None,
            "servidores": [],
        }

    lineas: list[str] = []
    resumen: list[dict[str, Any]] = []
    for nombre in sorted(definiciones):
        try:
            servidor = _activos.get(nombre)
            if servidor is None or not servidor.vivo:
                servidor = abrir_servidor(nombre, definiciones[nombre])
                await servidor.arrancar()
                _activos[nombre] = servidor
        except (ErrorMcp, OSError) as e:
            logger.warning("MCP '%s' no arrancó: %s", nombre, e)
            lineas.append(f"- {nombre}: no arrancó ({recortar(str(e), 120)})")
            resumen.append({"nombre": nombre, "error": recortar(str(e), 160)})
            continue

        # En el catálogo solo se enseñan las permitidas: que el modelo vea
        # `write_file` detrás de una lista que lo veta es invitarle a probar.
        permitidas = servidor._permitidas()
        visibles = [h for h in servidor.herramientas if not permitidas or str(h.get("name")) in permitidas]
        if visibles:
            # Con la firma, y no solo el nombre. Sin esto el modelo tiene que
            # adivinar cómo se llaman los parámetros —y adivina en español,
            # que es como se pierde una llamada por escribir `comando` donde
            # ponía `command`.
            lineas.append(f"- {nombre}:")
            for h in visibles:
                lineas.append(f"    {_firma(h)}")
        else:
            lineas.append(f"- {nombre}: sin herramientas")
        resumen.append(
            {
                "nombre": nombre,
                "nivel": str(definiciones[nombre]["nivel"]),
                "herramientas": [
                    {
                        "nombre": str(h.get("name")),
                        "descripcion": recortar(str(h.get("description") or ""), 200),
                        "parametros": _parametros(h),
                    }
                    for h in visibles
                ],
            }
        )

    texto = "Servidores MCP disponibles:\n" + "\n".join(lineas)
    return {"texto": texto, "titular": f"{len(resumen)} servidor(es) MCP", "servidores": resumen}


async def _llamar(peticion: dict[str, Any]) -> dict[str, Any]:
    nombre = str(peticion.get("servidor") or "").strip()
    herramienta = str(peticion.get("herramienta") or "").strip()
    argumentos = peticion.get("argumentos")
    if not nombre or not herramienta:
        raise ValueError("Para llamar a una herramienta hacen falta 'servidor' y 'herramienta'.")
    if argumentos is None:
        argumentos = {}
    if isinstance(argumentos, str):
        # Un modelo de voz manda a veces el objeto ya escrito como texto. Es
        # JSON válido: leerlo cuesta una línea y salva la llamada entera.
        try:
            argumentos = json.loads(argumentos or "{}")
        except json.JSONDecodeError:
            raise ValueError(
                "'argumentos' vino como texto y no es JSON: manda un objeto con los parámetros."
            ) from None
    if not isinstance(argumentos, dict):
        raise ValueError("'argumentos' debe ser un objeto con los parámetros de la herramienta.")

    servidor = _servidor(nombre)
    if not servidor.vivo:
        await servidor.arrancar()
        _activos[nombre] = servidor

    respuesta = await servidor.llamar(herramienta, argumentos)
    logger.info("MCP %s.%s contestó %d carácter(es).", nombre, herramienta, len(respuesta))
    titular = recortar(f"{nombre}.{herramienta}()", 120)
    return {
        "texto": respuesta or "(la herramienta no devolvió texto)",
        "titular": titular,
        "servidor": nombre,
        "herramienta": herramienta,
    }
