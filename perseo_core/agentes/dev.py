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

1. **Los encargos trabajan desde la carpeta del usuario, y de ahí no salen.** La
   raíz configurada (`PERSEO_DEV_RAIZ`, por defecto `C:\\Users\\<quien>`) más el
   **Escritorio** —que puede estar redirigido fuera del perfil por OneDrive—.
   Se pueden añadir más por entorno sin tocar código:
   `PERSEO_DEV_RAICES_EXTRA="C:\\una\\ruta;C:\\otra"`. Una ruta que se sale de
   todas se rechaza antes de arrancar nada. El motivo es el de siempre: lo que
   Perseo lee viene de correos y de pantallas, y desde la Fase D un correo puede
   acabar convertido en trabajo.

   Hasta el 2026-08-26 la raíz era este repositorio, y el proyecto nombrado en
   el encargo —«en Armario, arregla X»— **encerraba** al agente en la carpeta de
   ese proyecto. El señor Persus lo tachó: sus proyectos se llaman unos a otros
   y un agente encerrado en uno no puede leer el de al lado. Ahora el proyecto
   nombrado se le **cuenta** (`_contexto_del_encargo`) en vez de servirle de
   jaula, y la carpeta solo se estrecha si el encargo la pide expresamente.
2. **Editar sí, destruir no.** Se aceptan las ediciones sin preguntar —son
   reversibles, están en git, y es lo que dice §7 del plan— pero hay una lista de
   lo que no se ejecuta ni preguntando: `git push`, borrados, formateos.

   Lo que sí puede ejecutar depende de **quién pidió el encargo**: el señor
   Persus con el dedo (`origen` `texto` o `voz`) trabaja con
   `HERRAMIENTAS_PERMITIDAS_AMPLIAS` —`Bash` y subagentes incluidos, porque un
   agente que no puede arrancar un proceso no puede «abrir la app» y decía que
   sí—; lo que nace solo, de un correo o un disparador, se queda con la
   lista corta.
3. **`dev` tiene su propio carril.** Un encargo de código tarda minutos; el
   correo y la memoria tardan milisegundos. Los atiende un trabajador aparte para
   que uno no deje al otro esperando. Ver `Trabajador` en `agentes.py`.
4. **Todo lo que hace queda apuntado.** Cada herramienta, cada resultado y cada
   subagente van a la bitácora (`Paso`, `actividad_de`), en memoria y en
   `<datos>/actividad/<id>.jsonl`. No se borra al terminar el encargo: la
   pregunta «dice HECHO, ¿pero qué hizo?» solo se hace después.


"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import logging
import os
import re
import shutil
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from ..infra import almacen
from ..servicios import proyectos
from ..infra.router import registrar
from .dev_motores import (
    HERRAMIENTAS_PERMITIDAS,
    HERRAMIENTAS_PERMITIDAS_AMPLIAS,
    ORIGENES_DE_CONFIANZA,
    Encargo,
    Motor,
    MotorClaude,
    MotorFalso,
    MotorOpencode,
    Paso,
)
from .dev_sdk import MotorSdk

logger = logging.getLogger(__name__)

def hay_sdk() -> bool:
    """¿Está instalado el Agent SDK? Es opcional: sin él se habla por consola."""
    return importlib.util.find_spec("claude_agent_sdk") is not None


def abrir_motor(cfg: almacen.Configuracion) -> Motor | None:
    """Devuelve el motor configurado, o `None` si no hay ninguno utilizable.

    Sin `PERSEO_DEV_MOTOR` manda **opencode**: es el que no gasta suscripción,
    y el señor Persus lo pidió con todas las letras el 2026-08-26 —«que de una
    puñetera vez se use opencode de principal»—. Sabe contar por dónde va el
    encargo igual que el SDK (`--format json`), así que la razón por la que
    antes ganaba el SDK ya no existe.

    Si opencode no está instalado, el SDK; y si tampoco, la consola de Claude.
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

    if not cfg.dev_motor:
        ejecutable = shutil.which("opencode")
        if ejecutable is not None:
            return MotorOpencode(ejecutable)
        if hay_sdk():
            return MotorSdk()

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


def _parece_carpeta(candidato: str) -> bool:
    """¿Esto es una carpeta, o una dirección web disfrazada?

    Existe por lo que le pasó al señor Persus el 2026-08-25 estando fuera de
    casa: la lista de proyectos del móvil ofrecía «Armario · App» y mandaba su
    `destino`, que en modo servicio es `http://127.0.0.1:8000`. El encargo
    moría con «'http://127.0.0.1:8000' no es un directorio» antes de arrancar
    nada, y en la pantalla solo se veía FALLIDO sin decir por qué.
    """
    texto = (candidato or "").strip()
    return bool(texto) and "://" not in texto


def _carpeta_del_proyecto(proyecto: proyectos.Proyecto) -> str:
    """Dónde trabaja un agente en cada modo: la carpeta, nunca la URL.

    Se prueban los candidatos en orden y se devuelve el primero que sea una
    ruta de verdad. Un proyecto que solo sabe decir dónde MIRARSE —una URL— no
    tiene carpeta, y eso es un vacío, no un error.
    """
    if proyecto.modo == "arranque":
        candidatos = [proyecto.carpeta]
    elif proyecto.modo == "servicio":
        candidatos = [str(s.get("carpeta", "")) for s in proyecto.servidores]
        candidatos.append(proyecto.carpeta)
    else:
        candidatos = [proyecto.destino, proyecto.carpeta]
    for candidato in candidatos:
        if _parece_carpeta(candidato):
            return candidato
    return ""


def _contexto_del_encargo(instruccion: str, raiz: Path) -> str:
    """Lo que el agente tiene que saber antes de leer el encargo.

    Tres cosas, y las tres salieron de fallos reales:

    1. **Desde dónde trabaja y que puede moverse.** La carpeta del usuario
       entera, porque los proyectos del señor Persus se llaman unos a otros y
       encerrar al agente en uno era el impuesto de cada encargo (2026-08-26).
    2. **Dónde vive el proyecto que se nombra**, si se nombra alguno. Es la
       ruta absoluta, que es lo que evita que el modelo se la invente.
    3. **Que no dé por hecho lo que no hizo.** «Abre la app de armario» acabó
       en verde dos veces sin abrir nada (2026-08-25): un encargo que no se
       puede cumplir se dice, no se aprueba.
    """
    lineas = [
        f"Trabajas desde {raiz}, la carpeta personal del señor Persus. Sus "
        "proyectos cuelgan de ahí y se llaman unos a otros, así que puedes "
        "entrar en cualquiera de ellos; usa rutas absolutas.",
    ]
    if _datos is not None:
        carpeta = _proyecto_del_texto(instruccion, _datos)
        if carpeta:
            lineas.append(f"El proyecto que nombra el encargo está en {carpeta}.")
    lineas.append(
        "Si no puedes cumplir el encargo con las herramientas que tienes, dilo "
        "claramente y explica qué te falta. No lo des por hecho."
    )
    return " ".join(lineas)


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

#: Cuántos pasos se guardan por encargo en memoria, y cuántos encargos se
#: recuerdan. El tope existe porque la salida de un `pytest` largo cabe entera
#: en un paso, y cuarenta encargos con seiscientos pasos ya son memoria del
#: núcleo que no se recupera hasta reiniciar.
TOPE_PASOS = 600
TOPE_ENCARGOS_RECORDADOS = 40

#: Cuántos ficheros de bitácora se dejan en disco. Es un registro de depuración,
#: no un archivo histórico.
TOPE_BITACORAS_EN_DISCO = 200

#: La bitácora de cada encargo: todo lo que hicieron el agente y sus subagentes.
#: A diferencia de `_progreso`, esto NO se borra al terminar: la pregunta que
#: hubo que contestar el 2026-08-25 —«dice HECHO, pero ¿qué hizo?»— solo se
#: contesta después, y con el paso a paso delante.
_bitacoras: dict[int, list[Paso]] = {}


def _carpeta_bitacoras() -> Path | None:
    """Dónde se escriben las bitácoras, o `None` si no hay dónde.

    En disco además de en memoria porque el núcleo se reinicia y la pregunta
    «¿qué hizo el encargo de anoche?» sigue en pie a la mañana siguiente.
    """
    if _datos is None:
        return None
    carpeta = Path(_datos) / "actividad"
    try:
        carpeta.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return carpeta


def _anotar(id_trabajo: int, paso: Paso) -> None:
    """Apunta un paso: en memoria para el panel, en disco para mañana.

    Un fallo escribiendo no puede tumbar el encargo: la bitácora es para mirar,
    no una pieza de la que dependa el trabajo.
    """
    id_trabajo = int(id_trabajo)
    if not id_trabajo:
        return
    sellado = replace(paso, momento=datetime.now().isoformat(timespec="seconds"))

    if paso.titulo:
        _progreso[id_trabajo] = paso.titulo

    pasos = _bitacoras.setdefault(id_trabajo, [])
    pasos.append(sellado)
    if len(pasos) > TOPE_PASOS:
        del pasos[: len(pasos) - TOPE_PASOS]
    while len(_bitacoras) > TOPE_ENCARGOS_RECORDADOS:
        _bitacoras.pop(next(iter(_bitacoras)))

    carpeta = _carpeta_bitacoras()
    if carpeta is None:
        return
    try:
        with (carpeta / f"{id_trabajo}.jsonl").open("a", encoding="utf-8") as fichero:
            fichero.write(json.dumps(sellado.a_dict(), ensure_ascii=False) + "\n")
    except OSError as e:  # noqa: PERF203 — apuntar no puede romper el encargo
        logger.debug("No se pudo escribir la bitácora de #%d: %s", id_trabajo, e)


def _leer_bitacora(id_trabajo: int) -> list[Paso]:
    """La bitácora de un encargo que ya no está en memoria, leída del disco."""
    carpeta = _carpeta_bitacoras()
    if carpeta is None:
        return []
    fichero = carpeta / f"{int(id_trabajo)}.jsonl"
    if not fichero.is_file():
        return []
    pasos: list[Paso] = []
    try:
        for linea in fichero.read_text(encoding="utf-8").splitlines():
            if not linea.strip():
                continue
            try:
                crudo = json.loads(linea)
            except json.JSONDecodeError:
                continue
            pasos.append(
                Paso(
                    tipo=str(crudo.get("tipo", "")),
                    titulo=str(crudo.get("titulo", "")),
                    detalle=str(crudo.get("detalle", "")),
                    agente=str(crudo.get("agente", "principal")),
                    ok=bool(crudo.get("ok", True)),
                    momento=str(crudo.get("momento", "")),
                )
            )
    except OSError:
        return []
    return pasos[-TOPE_PASOS:]


def _limpiar_bitacoras_viejas() -> None:
    """Deja en disco solo las últimas. Se llama al arrancar, una vez."""
    carpeta = _carpeta_bitacoras()
    if carpeta is None:
        return
    try:
        ficheros = sorted(carpeta.glob("*.jsonl"), key=lambda f: f.stat().st_mtime)
    except OSError:
        return
    for viejo in ficheros[:-TOPE_BITACORAS_EN_DISCO]:
        with contextlib.suppress(OSError):
            viejo.unlink()


def progreso_de(id_trabajo: int) -> str:
    """Lo último que se sabe de un encargo en curso. Vacío si no hay nada."""
    return _progreso.get(int(id_trabajo), "")


def actividad_de(id_trabajo: int) -> dict[str, Any]:
    """Todo lo que hizo un encargo, suyo y de sus subagentes.

    Devuelve los pasos en orden y, aparte, la lista de quiénes trabajaron —el
    principal y cada subagente— con su nombre y cuántos pasos dio cada uno.
    Con eso la pantalla puede enseñar el encargo entero o meterse dentro de un
    subagente concreto, que es de lo que se trata para depurar.
    """
    id_trabajo = int(id_trabajo)
    pasos = _bitacoras.get(id_trabajo)
    if pasos is None:
        pasos = _leer_bitacora(id_trabajo)

    agentes: dict[str, dict[str, Any]] = {}
    for paso in pasos:
        ficha = agentes.setdefault(
            paso.agente,
            {
                "id": paso.agente,
                "titulo": "Agente principal" if paso.agente == "principal" else "",
                "pasos": 0,
                "fallos": 0,
            },
        )
        ficha["pasos"] += 1
        if not paso.ok:
            ficha["fallos"] += 1
        if paso.tipo == "subagente" and not ficha["titulo"]:
            ficha["titulo"] = paso.titulo
    for identificador, ficha in agentes.items():
        if not ficha["titulo"]:
            ficha["titulo"] = f"Subagente {identificador[:8]}"

    return {
        "id": id_trabajo,
        "vivo": id_trabajo in _progreso,
        "pasos": [p.a_dict() for p in pasos],
        "agentes": list(agentes.values()),
    }


def iniciar(cfg: almacen.Configuracion) -> Motor | None:
    global _motor, _raiz, _raices, _tope, _datos, _ejecutable_claude
    _raiz = Path(cfg.dev_raiz).resolve()
    _raices = raices_permitidas(_raiz)
    _tope = float(cfg.dev_tope)
    _datos = Path(cfg.directorio_datos)
    _ejecutable_claude = str(cfg.dev_ejecutable or "claude")
    _limpiar_bitacoras_viejas()
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
                "de los conocidos (sdk, claude, opencode, falso)."
            )
    elif _motor is not None:
        motor = _motor
    else:
        raise RuntimeError(
            "No hay motor para `dev`. Instala Claude Code u opencode y "
            "comprueba su ejecutable en el PATH, pon PERSEO_DEV_MOTOR=falso "
            "para probar el circuito, o elige motor en cada encargo."
        )

    # El directorio: la RAÍZ, salvo que el encargo pida otra cosa a propósito.
    #
    # Antes se adivinaba del texto —«en Armario…» encerraba al agente en la
    # carpeta de Armario— y el señor Persus lo tachó el 2026-08-26: sus
    # proyectos se llaman unos a otros, y un agente encerrado en uno no puede
    # mirar el de al lado. Ahora el proyecto nombrado se le CUENTA al agente
    # (ver `_contexto_del_encargo`) en vez de servirle de jaula.
    directorio_pedido = str(peticion.get("directorio", "")).strip()
    if not _parece_carpeta(directorio_pedido):
        # Una URL en `directorio` es lo que mandaba la lista del móvil: se
        # ignora en vez de tumbar el encargo con «no es un directorio».
        if directorio_pedido:
            logger.info("Se ignora %r como directorio: no es una carpeta.", directorio_pedido)
        directorio_pedido = ""
    raiz = resolver_raiz(directorio_pedido)
    sesion = str(peticion.get("sesion", ""))
    modelo = str(peticion.get("modelo", "")).strip()

    # Las manos que se le dan dependen de quién lo pidió: el señor Persus con
    # el dedo, o algo que Perseo leyó. Ver `HERRAMIENTAS_PERMITIDAS_AMPLIAS`.
    origen = str(trabajo.get("origen") or "texto")
    permitidas = (
        HERRAMIENTAS_PERMITIDAS_AMPLIAS
        if origen in ORIGENES_DE_CONFIANZA
        else HERRAMIENTAS_PERMITIDAS
    )

    logger.info(
        "Encargo de dev (%s%s) en %s: %.120s",
        type(motor).__name__.replace("Motor", "").lower() or pedido or "configurado",
        f", {modelo}" if modelo else "",
        raiz,
        instruccion,
    )
    id_trabajo = int(trabajo.get("id") or 0)

    def contar(paso: Paso) -> None:
        _anotar(id_trabajo, paso)

    encargo = Encargo(
        instruccion=instruccion,
        raiz=raiz,
        tope=_tope,
        sesion=sesion,
        modelo=modelo,
        contexto=_contexto_del_encargo(instruccion, raiz),
        permitidas=permitidas,
        carpetas_extra=tuple(r for r in _raices if r != raiz),
    )

    try:
        resultado = await motor.ejecutar(encargo, avisar=contar)
    except Exception as e:
        _anotar(id_trabajo, Paso(tipo="error", titulo=str(e)[:120], detalle=str(e), ok=False))
        raise
    finally:
        _progreso.pop(id_trabajo, None)

    if not resultado.ok:
        raise RuntimeError(resultado.texto or "El encargo falló.")

    return {
        "texto": resultado.texto,
        "vueltas": resultado.vueltas,
        # Cuántos pasos quedaron apuntados: es la pista de que hay una bitácora
        # que mirar, y de si el encargo hizo algo o solo habló.
        "pasos": len(_bitacoras.get(id_trabajo, ())),
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
