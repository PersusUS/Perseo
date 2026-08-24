"""Agente `dev`: escribir, programar y probar, con el Claude Agent SDK.

Este agente no se escribe, se envuelve. El bucle de agente, las herramientas, el
manejo de contexto y los subagentes ya existen en Claude Code y en su SDK;
reimplementarlos sería tirar meses para tener algo peor. Lo que hay aquí es el
puente entre la cola del núcleo y ese bucle, con la parte que sí es decisión
nuestra: **qué puede tocar y qué no**.

Sobre el coste: el Agent SDK y `claude -p` entran en la suscripción (§3 del
plan), así que este agente no gasta dinero nuevo. La restricción que sí aplica es
que la autenticación por suscripción está prohibida en productos de terceros —
Perseo es una herramienta personal y queda dentro.

**El motor es un puerto**, como el buzón y el vault. Hoy detrás está la línea de
comandos de Claude Code, que es lo que hay instalado; el día que el paquete
`claude-agent-sdk` esté en el entorno, es escribir otra clase con el mismo
método. El agente no cambia.

Tres decisiones que gobiernan el módulo:

1. **Los encargos no salen de las raíces permitidas.** La raíz configurada
   (`PERSEO_DEV_RAIZ`, por defecto este repositorio) más el **Escritorio** —que
   es lo que las instrucciones le prometen al modelo, y lo que el servidor MCP
   'subagentes' ya permite—. Se pueden añadir más por entorno sin tocar código:
   `PERSEO_DEV_RAICES_EXTRA="C:\\una\\ruta;C:\\otra"`. Una ruta que se sale de
   todas se rechaza antes de arrancar nada. El motivo es el de siempre: lo que
   Perseo lee viene de correos y de pantallas, y desde la Fase D un correo puede
   acabar convertido en trabajo.
2. **Editar sí, destruir no.** Se aceptan las ediciones sin preguntar —son
   reversibles, están en git, y es lo que dice §7 del plan— pero hay una lista de
   lo que no se ejecuta ni preguntando: `git push`, borrados, formateos. Y una
   lista corta de comandos que sí puede correr sin pedir permiso, que es la que
   le permite comprobar su propio trabajo.
3. **`dev` tiene su propio carril.** Un encargo de código tarda minutos; el
   correo y la memoria tardan milisegundos. Los atiende un trabajador aparte para
   que uno no deje al otro esperando. Ver `Trabajador` en `agentes.py`.

Ver bitacora/05_PLAN_PERSEO_V2.md §3, §7 y §9 (Fase E).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any, Protocol

from . import almacen, proyectos
from .agentes import registrar

logger = logging.getLogger(__name__)

#: Lo que `dev` puede ejecutar sin preguntar. Es corta a propósito: lo justo para
#: que compruebe lo que acaba de escribir. Todo lo demás se deniega.
HERRAMIENTAS_PERMITIDAS = (
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
    "TodoWrite",
    "Bash(python -m pytest*)",
    "Bash(python -m perseo_core*)",
    "Bash(python perseo_core/verificar*)",
    "Bash(npx tsc*)",
    "Bash(cargo check*)",
    "Bash(git status*)",
    "Bash(git diff*)",
    "Bash(git log*)",
)

#: Lo que no se ejecuta ni aunque se permitiera por otro lado. Publicar es del
#: usuario, y borrar no tiene vuelta atrás: las dos cosas están fuera del nivel
#: "reversible" de §7.
HERRAMIENTAS_DENEGADAS = (
    "Bash(git push*)",
    "Bash(git reset --hard*)",
    "Bash(git clean*)",
    "Bash(rm *)",
    "Bash(rmdir *)",
    "Bash(del *)",
    "Bash(format*)",
    "WebFetch",
    "WebSearch",
)

#: Tope de vueltas del bucle de agente. Sin él, un encargo mal entendido puede
#: dar vueltas sin fin contra la cuota de la suscripción.
MAX_VUELTAS = 40


@dataclass(frozen=True)
class Resultado:
    """Lo que devuelve un motor. `sesion` permite continuar el encargo después."""

    texto: str
    ok: bool = True
    vueltas: int = 0
    sesion: str = ""


#: Cómo avisa un motor de por dónde va. Solo el motor sobre el SDK sabe
#: rellenarlo —los que hablan por línea de órdenes no dicen nada hasta el
#: final—, y quien no lo use no paga nada por tenerlo.
Aviso = Callable[[str], None]


class Motor(Protocol):
    """Quién ejecuta de verdad el encargo."""

    async def ejecutar(
        self,
        instruccion: str,
        raiz: Path,
        tope: float,
        sesion: str = "",
        avisar: Aviso | None = None,
    ) -> Resultado: ...


class MotorClaude:
    """Claude Code en modo no interactivo (`claude -p`).

    Se habla con él por línea de comandos y no por biblioteca porque es lo que
    está instalado. La salida se pide en JSON, que trae el texto final, el número
    de vueltas y el identificador de sesión para poder retomar el encargo.
    """

    def __init__(self, ejecutable: str) -> None:
        self._ejecutable = ejecutable

    async def ejecutar(
        self,
        instruccion: str,
        raiz: Path,
        tope: float,
        sesion: str = "",
        avisar: Aviso | None = None,
    ) -> Resultado:
        argumentos = [
            self._ejecutable,
            "-p",
            instruccion,
            "--output-format",
            "json",
            # Las ediciones se aceptan solas; lo que no está en la lista de
            # permitidas se deniega en vez de quedarse esperando una respuesta
            # que aquí no puede dar nadie.
            "--permission-mode",
            "acceptEdits",
            "--max-turns",
            str(MAX_VUELTAS),
            "--allowed-tools",
            *HERRAMIENTAS_PERMITIDAS,
            "--disallowed-tools",
            *HERRAMIENTAS_DENEGADAS,
        ]
        if sesion:
            argumentos += ["--resume", sesion]

        proceso = await asyncio.create_subprocess_exec(
            *argumentos,
            cwd=str(raiz),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            salida, error = await asyncio.wait_for(proceso.communicate(), timeout=tope)
        except asyncio.TimeoutError:
            proceso.kill()
            await proceso.wait()
            raise TimeoutError(f"El encargo pasó de {tope:.0f} s y se cortó.") from None

        texto = salida.decode("utf-8", "replace").strip()
        if proceso.returncode != 0:
            detalle = error.decode("utf-8", "replace").strip()[:500] or texto[:500]
            return Resultado(texto=detalle or "Claude terminó con error.", ok=False)

        try:
            datos = json.loads(texto)
        except json.JSONDecodeError:
            # Sin JSON no hay metadatos, pero el texto sigue valiendo.
            return Resultado(texto=texto[:4000])

        return Resultado(
            texto=str(datos.get("result", ""))[:4000],
            ok=not datos.get("is_error", False),
            vueltas=int(datos.get("num_turns", 0) or 0),
            sesion=str(datos.get("session_id", "")),
        )


#: Lo que un CLI de agente escribe cuando ha fracasado PERO sale con código 0.
#: Dos casos vistos el 2026-08-24: opencode denegándose a sí mismo el permiso de
#: escribir (sin `--auto`) y el proveedor del modelo gratuito cayéndose a media
#: petición. Los dos daban el encargo por bueno con el disco intacto.
SENALES_DE_FRACASO = (
    "auto-rejecting",
    "rejected permission",
    "error from provider",
    "endpoint is unavailable",
    "no such model",
)


def fracaso_encubierto(texto: str) -> str:
    """El motivo, si la salida delata un fracaso con código de éxito. O ''."""
    bajo = (texto or "").lower()
    for senal in SENALES_DE_FRACASO:
        if senal in bajo:
            for linea in reversed((texto or "").splitlines()):
                if senal in linea.lower():
                    return linea.strip()[:400]
            return senal
    return ""


class MotorOpencode:
    """opencode en modo no interactivo (`opencode run`).

    El segundo motor, para que el señor Persus ELIJA con quién trabaja cada
    encargo (2026-08-24): Claude de suscripción u opencode gratuito. Los
    permisos van en `--auto`, porque sin él este programa se deniega a sí mismo
    lo que necesita para trabajar y sale con código 0 igualmente. El modelo, en
    `PERSEO_DEV_MODELO` si se quiere uno concreto. La salida llega formateada y
    con controles ANSI; se limpian aquí una vez, en un sitio.

    No es el motor por defecto: su endpoint gratuito se cae a ratos, y un
    encargo que muere así no siempre lo dice.
    """

    _ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

    def __init__(self, ejecutable: str) -> None:
        self._ejecutable = ejecutable

    async def ejecutar(
        self,
        instruccion: str,
        raiz: Path,
        tope: float,
        sesion: str = "",
        avisar: Aviso | None = None,
    ) -> Resultado:
        # `--auto` no es una comodidad: sin él, `opencode run` pide permiso para
        # escribir, nadie contesta porque esto no es interactivo, y el propio
        # programa se lo deniega («auto-rejecting») saliendo con código 0. El
        # encargo se apuntaba como hecho sin haber tocado un fichero (2026-08-24).
        argumentos = [self._ejecutable, "run", "--auto"]
        modelo = os.environ.get("PERSEO_DEV_MODELO", "").strip()
        if modelo:
            argumentos += ["-m", modelo]
        if sesion:
            argumentos += ["--session", sesion]
        argumentos.append(instruccion)

        proceso = await asyncio.create_subprocess_exec(
            *argumentos,
            cwd=str(raiz),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            salida, error = await asyncio.wait_for(proceso.communicate(), timeout=tope)
        except asyncio.TimeoutError:
            proceso.kill()
            await proceso.wait()
            raise TimeoutError(f"El encargo pasó de {tope:.0f} s y se cortó.") from None

        texto = self._ANSI.sub("", salida.decode("utf-8", "replace")).strip()
        if proceso.returncode != 0:
            detalle = error.decode("utf-8", "replace").strip()[:500] or texto[:500]
            return Resultado(texto=detalle or "opencode terminó con error.", ok=False)

        # Código 0 tampoco basta aquí: el modelo gratuito devuelve «Endpoint is
        # unavailable» y sale bien. Un encargo que no hizo nada tiene que
        # contarse como fallo, o el panel enseña éxitos que no existieron.
        motivo = fracaso_encubierto(texto)
        if motivo:
            return Resultado(texto=motivo, ok=False)

        return Resultado(texto=texto[:4000])


class MotorSdk:
    """Claude por el **Agent SDK oficial** (`claude-agent-sdk`), no por la consola.

    Es el mismo bucle de agente que `MotorClaude`, con la diferencia que se
    nota usándolo: los mensajes llegan **según pasan**, así que se puede contar
    por dónde va el encargo en vez de enseñar una barra girando durante seis
    minutos. Lo que se cuenta es la herramienta que acaba de usar —«Editando
    api.py», «Ejecutando pytest»—, que es la pregunta que uno se hace mirando.

    Lo demás es lo mismo y a propósito: las mismas listas de herramientas
    permitidas y denegadas, el mismo tope de vueltas y el mismo cerco de
    directorios resuelto antes de arrancar. El SDK no relaja ninguna decisión
    de seguridad; solo cambia por dónde se habla con el agente.

    `setting_sources=["project"]`: el encargo hereda el `CLAUDE.md` del
    proyecto donde trabaja —que es contexto útil— pero **no** los ajustes ni
    los hooks globales del usuario. Un agente que corre solo, de madrugada y
    sin nadie mirando no debe arrastrar la configuración de una sesión humana.
    """

    def __init__(self) -> None:
        # Se importa aquí y no arriba: el paquete es opcional (§18) y sin él
        # el núcleo tiene que arrancar igual, con el motor de consola.
        from claude_agent_sdk import ClaudeAgentOptions, query

        self._query = query
        self._Opciones = ClaudeAgentOptions

    async def ejecutar(
        self,
        instruccion: str,
        raiz: Path,
        tope: float,
        sesion: str = "",
        avisar: Aviso | None = None,
    ) -> Resultado:
        from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock

        opciones = self._Opciones(
            cwd=str(raiz),
            permission_mode="acceptEdits",
            max_turns=MAX_VUELTAS,
            allowed_tools=list(HERRAMIENTAS_PERMITIDAS),
            disallowed_tools=list(HERRAMIENTAS_DENEGADAS),
            setting_sources=["project"],
            resume=sesion or None,
            # NO es un adorno. Sin el preset, el SDK arranca al agente SIN el
            # preámbulo de Claude Code —el que le dice en qué directorio está
            # trabajando— y el modelo se inventa rutas absolutas: el mismo
            # encargo escribió en `C:\Users\<usuario>`, en `C:\Users\jp` y en la
            # raíz del repositorio, tres veces seguidas y ninguna donde tocaba.
            # Con el preset, el fichero cae exactamente en `cwd` (H-67).
            system_prompt={"type": "preset", "preset": "claude_code"},
        )

        texto_suelto: list[str] = []
        final: Any = None
        try:
            async with asyncio.timeout(tope):
                async for mensaje in self._query(prompt=instruccion, options=opciones):
                    if isinstance(mensaje, AssistantMessage):
                        for bloque in mensaje.content:
                            if isinstance(bloque, ToolUseBlock) and avisar is not None:
                                avisar(_contar_herramienta(bloque.name, bloque.input))
                            elif isinstance(bloque, TextBlock):
                                texto_suelto.append(bloque.text)
                    elif isinstance(mensaje, ResultMessage):
                        final = mensaje
        except TimeoutError:
            raise TimeoutError(f"El encargo pasó de {tope:.0f} s y se cortó.") from None

        if final is None:
            # El SDK terminó sin dar resultado: pasa si el proceso muere solo.
            unido = "\n".join(texto_suelto).strip()
            return Resultado(texto=unido or "El agente terminó sin decir nada.", ok=bool(unido))

        return Resultado(
            texto=str(final.result or "\n".join(texto_suelto))[:4000],
            ok=not final.is_error,
            vueltas=int(final.num_turns or 0),
            sesion=str(final.session_id or ""),
        )


#: Cómo se cuenta cada herramienta mientras el encargo corre. Se dice qué está
#: pasando, no el JSON de la llamada: quien mira quiere saber si avanza.
_COMO_SE_CUENTA = {
    "Read": "Leyendo",
    "Write": "Escribiendo",
    "Edit": "Editando",
    "Glob": "Buscando ficheros",
    "Grep": "Buscando",
    "Bash": "Ejecutando",
    "TodoWrite": "Ordenando el trabajo",
    "Task": "Repartiendo a un subagente",
}


def _contar_herramienta(nombre: str, entrada: dict[str, Any]) -> str:
    """Una línea corta y en cristiano de lo que el agente acaba de hacer."""
    verbo = _COMO_SE_CUENTA.get(nombre, nombre)
    detalle = ""
    for clave in ("file_path", "path", "pattern", "command", "description"):
        valor = entrada.get(clave) if isinstance(entrada, dict) else None
        if valor:
            detalle = Path(str(valor)).name if clave in ("file_path", "path") else str(valor)
            break
    return f"{verbo} {detalle}".strip()[:120]


class MotorFalso:
    """Motor de mentira, para verificar el circuito sin gastar suscripción.

    Existe por lo mismo que `BuzonFalso`: el camino que va de la cola al agente y
    vuelta —incluido lo que pasa cuando un encargo tarda— se puede comprobar sin
    depender de un servicio de fuera.
    """

    def __init__(self, tardanza: float = 0.0) -> None:
        self.tardanza = tardanza
        self.encargos: list[str] = []

    async def ejecutar(
        self,
        instruccion: str,
        raiz: Path,
        tope: float,
        sesion: str = "",
        avisar: Aviso | None = None,
    ) -> Resultado:
        self.encargos.append(instruccion)
        if self.tardanza:
            await asyncio.sleep(min(self.tardanza, tope))
        return Resultado(texto=f"(simulado) {instruccion}", vueltas=1, sesion="falsa")


def hay_sdk() -> bool:
    """¿Está instalado el Agent SDK? Es opcional: sin él se habla por consola."""
    return importlib.util.find_spec("claude_agent_sdk") is not None


def abrir_motor(cfg: almacen.Configuracion) -> Motor | None:
    """Devuelve el motor configurado, o `None` si no hay ninguno utilizable.

    Sin `PERSEO_DEV_MOTOR` manda el **SDK** si está instalado, porque es el
    único que sabe contar por dónde va el encargo mientras corre; si no está,
    la consola de Claude, que hace lo mismo callada.
    """
    if cfg.dev_motor == "falso":
        return MotorFalso(tardanza=float(cfg.dev_tardanza_falsa))

    if cfg.dev_motor == "sdk":
        if not hay_sdk():
            logger.warning(
                "PERSEO_DEV_MOTOR=sdk pero no está `claude-agent-sdk` en el "
                "entorno; se sigue con la consola de Claude. `pip install "
                "claude-agent-sdk`."
            )
        else:
            return MotorSdk()

    if not cfg.dev_motor and hay_sdk():
        return MotorSdk()

    if cfg.dev_motor == "opencode":
        ejecutable = shutil.which("opencode")
        if ejecutable is None:
            logger.warning(
                "PERSEO_DEV_MOTOR=opencode pero no se encuentra %r en el PATH; "
                "el agente `dev` fallará hasta que esté.",
                "opencode",
            )
            return None
        return MotorOpencode(ejecutable)

    ejecutable = shutil.which(cfg.dev_ejecutable)
    if ejecutable is None:
        logger.warning(
            "No se encuentra %r en el PATH; el agente `dev` fallará hasta que esté.",
            cfg.dev_ejecutable,
        )
        return None
    return MotorClaude(ejecutable)


def _escritorio() -> Path | None:
    """La carpeta de Escritorio DE VERDAD, preguntándoselo a Windows.

    «~/Desktop» se queda corto: un Windows español con OneDrive redirige la
    carpeta y la llama «Escritorio», y lo que vale no es la adivinanza sino la
    API de carpetas conocidas. Si algo falla, `None` y punto: es un extra del
    cerco, nunca una pieza de la que dependa nada.
    """
    if os.name != "nt":
        return None
    try:
        import ctypes

        # FOLDERID_Desktop {B4BFCC3A-DB2C-424C-B029-7FE99A87C641}. A mano
        # porque `ctypes.wintypes` no trae GUID hasta Python 3.12.
        class _GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", ctypes.c_ulong),
                ("Data2", ctypes.c_ushort),
                ("Data3", ctypes.c_ushort),
                ("Data4", ctypes.c_ubyte * 8),
            ]

        identificador = _GUID(
            0xB4BFCC3A,
            0xDB2C,
            0x424C,
            (ctypes.c_ubyte * 8)(0xB0, 0x29, 0x7F, 0xE9, 0x9A, 0x87, 0xC6, 0x41),
        )
        salida = ctypes.c_wchar_p()
        shell32 = ctypes.WinDLL("shell32", use_last_error=True)
        shell32.SHGetKnownFolderPath.argtypes = (
            ctypes.POINTER(_GUID),
            ctypes.c_ulong,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_wchar_p),
        )
        if shell32.SHGetKnownFolderPath(
            ctypes.byref(identificador), 0, None, ctypes.byref(salida)
        ) != 0:
            return None
        ruta = Path(str(salida.value))
        memoria = ctypes.cast(salida, ctypes.c_void_p)
        if memoria.value:
            ctypes.WinDLL("ole32").CoTaskMemFree(memoria)
        return ruta
    except Exception:  # noqa: BLE001 — sin Escritorio conocido, se vive sin él
        return None


def raices_permitidas(raiz: Path) -> tuple[Path, ...]:
    """El cerco entero: la raíz configurada, el PERFIL DEL USUARIO y los extras.

    El perfil entero (`C:\\Users\\<quien>`) lo mandó el señor Persus el
    2026-08-24 —*«que tenga permiso para trabajar en todo usuario»—: sus
    proyectos viven repartidos por la carpeta personal (MAGI, armario…) y
    andar listándolos a mano era el impuesto de cada encargo. El Escritorio
    sigue añadiéndose aparte porque OneDrive puede redirigirlo FUERA del
    perfil. Los extras siguen por entorno (`PERSEO_DEV_RAICES_EXTRA`,
    separada por `;`) para lo que esté fuera de `C:\\Users`.
    """
    rutas: list[Path] = [raiz, Path.home()]
    for crudo in os.environ.get("PERSEO_DEV_RAICES_EXTRA", "").split(";"):
        if crudo.strip():
            rutas.append(Path(crudo.strip()).expanduser())
    escritorio = _escritorio()
    if escritorio is not None:
        rutas.append(escritorio)

    resueltas: list[Path] = []
    for ruta in rutas:
        resuelta = ruta.resolve()
        if resuelta not in resueltas:
            resueltas.append(resuelta)
    return tuple(resueltas)


# --------------------------------------------------------------------------- #
# El encargo en lenguaje natural
# --------------------------------------------------------------------------- #

#: El motor y el proyecto se DICEN en el propio encargo — «en Armario, añade
#: un README, con opencode» — y el núcleo los entiende aquí. La cara manda el
#: texto tal cual: las caras no piensan, que para eso está el núcleo.
_MOTOR_DEL_TEXTO = re.compile(
    r"\b(?:con|usando|usa|vía|via)\s+(opencode|claude)\b", re.IGNORECASE
)


def _motor_del_texto(texto: str) -> str:
    """El motor pedido en el encargo, o vacío si no se dijo ninguno."""
    coincidencia = _MOTOR_DEL_TEXTO.search(texto)
    return coincidencia.group(1).lower() if coincidencia else ""


def _carpeta_del_proyecto(proyecto: proyectos.Proyecto) -> str:
    """Dónde trabaja un agente en cada modo: la carpeta, nunca la URL."""
    if proyecto.modo == "arranque":
        return proyecto.carpeta
    if proyecto.modo == "servicio":
        if proyecto.servidores:
            return str(proyecto.servidores[0].get("carpeta", ""))
        return ""
    return proyecto.destino


def _proyecto_del_texto(texto: str, datos: Path) -> str:
    """La carpeta del proyecto nombrado en el encargo, o vacío.

    Se busca el nombre de ficha («Armario», «CVScraper») como palabra entera
    — que «perseo» no salte dentro de «perseverar». El primero que aparezca
    en la lista gana; nombrar dos proyectos en un encargo es dos encargos.
    """
    minusculas = texto.lower()
    for proyecto in proyectos.listar(datos):
        ficha = proyecto.nombre.lower()
        token = re.split(r"[·—–-]", ficha, maxsplit=1)[0].strip()
        if token and re.search(rf"(?<!\w){re.escape(token)}(?!\w)", minusculas):
            return _carpeta_del_proyecto(proyecto)
    return ""


# --------------------------------------------------------------------------- #
# El agente
# --------------------------------------------------------------------------- #

_motor: Motor | None = None
_raiz: Path | None = None
_raices: tuple[Path, ...] = ()
_tope: float = 900.0
#: Dónde viven `proyectos.json` y compañía: hace falta para entender el
#: proyecto nombrado en un encago en lenguaje natural.
_datos: Path | None = None
#: El ejecutable de Claude, por si el entorno le cambió el nombre. Se fija en
#: `iniciar` para que la elección POR ENCARGO (peticion.motor) lo respete.
_ejecutable_claude: str = "claude"

#: Por dónde va cada encargo vivo, para que el panel enseñe algo mejor que una
#: barra girando. En memoria a propósito: un encargo en curso no sobrevive a un
#: reinicio del núcleo —`recuperar_huerfanos` lo devuelve a la cola—, así que
#: guardar esto en disco sería conservar una frase que ya no es verdad.
_progreso: dict[int, str] = {}


def progreso_de(id_trabajo: int) -> str:
    """Lo último que se sabe de un encargo en curso. Vacío si no hay nada."""
    return _progreso.get(int(id_trabajo), "")


def iniciar(cfg: almacen.Configuracion) -> Motor | None:
    global _motor, _raiz, _raices, _tope, _datos, _ejecutable_claude
    _raiz = Path(cfg.dev_raiz).resolve()
    _raices = raices_permitidas(_raiz)
    _tope = float(cfg.dev_tope)
    _datos = Path(cfg.directorio_datos)
    _ejecutable_claude = str(cfg.dev_ejecutable or "claude")
    if _motor is None:
        _motor = abrir_motor(cfg)
        if _motor is not None:
            logger.info("Agente dev listo sobre %s (tope %.0f s).", list(map(str, _raices)), _tope)
    return _motor


def detener() -> None:
    global _motor, _raices, _datos
    _motor = None
    _raices = ()
    _datos = None
    _progreso.clear()


def _motor_de(nombre: str) -> Motor | None:
    """Un motor para UN encargo, por nombre. `None` si no está instalado.

    Los nombres son los de `PERSEO_DEV_MOTOR`: `claude`, `opencode` y `falso`.
    Vacío no llega aquí — lo filtra quien llama, que usa el motor por defecto.
    """
    nombre = nombre.strip().lower()
    if nombre == "falso":
        return MotorFalso()
    if nombre == "sdk":
        return MotorSdk() if hay_sdk() else None
    if nombre == "opencode":
        ejecutable = shutil.which("opencode")
        return MotorOpencode(ejecutable) if ejecutable else None
    if nombre in ("claude", "anthropic"):
        ejecutable = shutil.which(_ejecutable_claude)
        return MotorClaude(ejecutable) if ejecutable else None
    return None


def resolver_raiz(pedida: str) -> Path:
    """Comprueba que el encargo se queda dentro de las raíces permitidas.

    Se resuelve antes de comparar: comparar cadenas sin resolver es exactamente
    como se cuela un `..`. Es la misma regla que en `memoria.py`, y por el mismo
    motivo — lo que llega puede venir de un correo.
    """
    if _raiz is None or not _raices:
        raise RuntimeError("El agente dev no está iniciado; falta llamar a dev.iniciar().")
    if not pedida:
        return _raiz

    # Unir una ruta absoluta a `_raiz` da la absoluta tal cual (pathlib), así que
    # la misma comparación vale para relativas —caen dentro o fuera— y para el
    # Escritorio u otra raíz que llegue ya absoluta.
    destino = (_raiz / pedida).expanduser().resolve()
    if not any(destino == raiz or raiz in destino.parents for raiz in _raices):
        listadas = ", ".join(str(raiz) for raiz in _raices)
        raise ValueError(f"{pedida!r} cae fuera de las raíces permitidas ({listadas}).")
    if not destino.is_dir():
        raise ValueError(f"{pedida!r} no es un directorio.")
    return destino


@registrar("dev")
async def _dev(trabajo: dict[str, Any]) -> dict[str, Any]:
    """Le encarga a un agente de código una tarea y devuelve lo que contestó.

    El motor se elige por encargo (`peticion.motor`: `claude` u `opencode`);
    sin elegir, manda el configurado al arrancar. No pide confirmación: editar
    código es reversible y está en git, que es lo que dice §7 del plan. Lo
    irreversible —publicar, borrar— no está en la lista de lo que puede
    ejecutar.
    """
    peticion = trabajo.get("peticion") or {}
    instruccion = str(peticion.get("texto") or peticion.get("instruccion") or "").strip()
    if not instruccion:
        raise ValueError("Un encargo de `dev` necesita `texto`.")

    # El motor: explícito en la petición, dicho en el texto («con opencode»),
    # o el configurado al arrancar. En ese orden.
    pedido = str(peticion.get("motor") or "").strip().lower() or _motor_del_texto(instruccion)
    if pedido:
        motor = _motor_de(pedido)
        if motor is None:
            raise ValueError(
                f"El motor {pedido!r} no está disponible ahora mismo: o no está "
                "instalado su ejecutable en el PATH, o el nombre no es ninguno "
                "de los conocidos (claude, opencode)."
            )
    elif _motor is not None:
        motor = _motor
    else:
        raise RuntimeError(
            "No hay motor para `dev`. Instala Claude Code u opencode y "
            "comprueba su ejecutable en el PATH, pon PERSEO_DEV_MOTOR=falso "
            "para probar el circuito, o elige motor en cada encargo."
        )

    # El directorio: explícito, nombrado en el texto («en Armario…»), o la raíz.
    directorio_pedido = str(peticion.get("directorio", "")).strip()
    if not directorio_pedido and _datos is not None:
        directorio_pedido = _proyecto_del_texto(instruccion, _datos)
    raiz = resolver_raiz(directorio_pedido)
    sesion = str(peticion.get("sesion", ""))

    logger.info(
        "Encargo de dev (%s) en %s: %.120s",
        type(motor).__name__.replace("Motor", "").lower() or pedido or "configurado",
        raiz,
        instruccion,
    )
    id_trabajo = int(trabajo.get("id") or 0)

    def contar(paso: str) -> None:
        if id_trabajo:
            _progreso[id_trabajo] = paso

    try:
        resultado = await motor.ejecutar(instruccion, raiz, _tope, sesion, avisar=contar)
    finally:
        _progreso.pop(id_trabajo, None)

    if not resultado.ok:
        raise RuntimeError(resultado.texto or "El encargo falló.")

    return {
        "texto": resultado.texto,
        "vueltas": resultado.vueltas,
        # Con esto, un encargo siguiente puede continuar donde se quedó en vez de
        # empezar de cero: `peticion.sesion` con este valor.
        "sesion": resultado.sesion,
        "directorio": str(raiz),
        # Por Telegram, el titular; el texto entero se lee por el tailnet.
        "titular": (
            f"Encargo de código terminado "
            f"({resultado.vueltas} vuelta{'s' if resultado.vueltas != 1 else ''})"
        ),
    }
