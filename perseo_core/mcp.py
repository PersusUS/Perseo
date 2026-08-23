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

Ver bitacora/06_HANDOFF.md §13.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from . import almacen, politica
from .agentes import registrar

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
        comando = entrada.get("comando")
        if not isinstance(comando, list) or not all(isinstance(p, str) for p in comando):
            logger.warning(
                "Servidor %r sin 'comando' como lista de argumentos; se ignora. "
                "Una cadena nunca vale: por ahí entran las comillas y los &&.",
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
        validos[str(nombre)] = {
            "comando": [str(p) for p in comando],
            "nivel": nivel,
            "herramientas": herramientas,
            "env": {str(k): str(v) for k, v in entorno.items()},
            "tope_segundos": max(5.0, min(tope, 600.0)),
        }
    return validos


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

            permitidas = self._permitidas()
            if herramienta not in {h.get("name") for h in self.herramientas}:
                disponibles = ", ".join(sorted(str(h.get("name")) for h in self.herramientas))
                raise ErrorMcp(f"'{self.nombre}' no tiene ninguna herramienta '{herramienta}'. Tiene: {disponibles}")
            if permitidas and herramienta not in permitidas:
                raise ErrorMcp(
                    f"'{herramienta}' no está en la lista de '{self.nombre}' en {NOMBRE_FICHERO}."
                )

            resultado = await self._pedir(
                "tools/call", {"name": herramienta, "arguments": argumentos}
            )
        contenido = (resultado or {}).get("content") or []
        textos = [
            str(bloque.get("text", ""))
            for bloque in contenido
            if isinstance(bloque, dict) and bloque.get("type") == "text"
        ]
        if (resultado or {}).get("isError"):
            raise ErrorMcp(recortar(" ".join(t for t in textos if t)) or "el servidor devolvió un error")
        return "\n".join(t for t in textos if t).strip()

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


def recortar(texto: str, tope: int = 300) -> str:
    limpio = " ".join(str(texto).split())
    return limpio if len(limpio) <= tope else limpio[: tope - 1].rstrip() + "…"


# -- Ciclo de vida del módulo ------------------------------------------------ #

#: Lo que hay en `mcp.json`, cargado al arrancar el núcleo.
definiciones: dict[str, dict[str, Any]] = {}
#: Los procesos vivos, uno por servidor configurado.
_activos: dict[str, ServidorMcp] = {}


async def iniciar(cfg: almacen.Configuracion) -> None:
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
    nuevo = ServidorMcp(nombre, definicion)
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
    return str(definicion["nivel"])


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
                servidor = ServidorMcp(nombre, definiciones[nombre])
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
        lineas.append(f"- {nombre}: {', '.join(str(h.get('name')) for h in visibles) or 'sin herramientas'}")
        resumen.append(
            {
                "nombre": nombre,
                "nivel": str(definiciones[nombre]["nivel"]),
                "herramientas": [
                    {
                        "nombre": str(h.get("name")),
                        "descripcion": recortar(str(h.get("description") or ""), 200),
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
