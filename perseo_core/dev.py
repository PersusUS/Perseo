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

1. **Los encargos no salen de la raíz permitida.** `PERSEO_DEV_RAIZ`, que por
   defecto es este repositorio. Una ruta que se sale se rechaza antes de arrancar
   nada. El motivo es el de siempre: lo que Perseo lee viene de correos y de
   pantallas, y desde la Fase D un correo puede acabar convertido en trabajo.
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
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from . import almacen
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


class Motor(Protocol):
    """Quién ejecuta de verdad el encargo."""

    async def ejecutar(
        self, instruccion: str, raiz: Path, tope: float, sesion: str = ""
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
        self, instruccion: str, raiz: Path, tope: float, sesion: str = ""
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
        self, instruccion: str, raiz: Path, tope: float, sesion: str = ""
    ) -> Resultado:
        self.encargos.append(instruccion)
        if self.tardanza:
            await asyncio.sleep(min(self.tardanza, tope))
        return Resultado(texto=f"(simulado) {instruccion}", vueltas=1, sesion="falsa")


def abrir_motor(cfg: almacen.Configuracion) -> Motor | None:
    """Devuelve el motor configurado, o `None` si no hay ninguno utilizable."""
    if cfg.dev_motor == "falso":
        return MotorFalso(tardanza=float(cfg.dev_tardanza_falsa))

    ejecutable = shutil.which(cfg.dev_ejecutable)
    if ejecutable is None:
        logger.warning(
            "No se encuentra %r en el PATH; el agente `dev` fallará hasta que esté.",
            cfg.dev_ejecutable,
        )
        return None
    return MotorClaude(ejecutable)


# --------------------------------------------------------------------------- #
# El agente
# --------------------------------------------------------------------------- #

_motor: Motor | None = None
_raiz: Path | None = None
_tope: float = 900.0


def iniciar(cfg: almacen.Configuracion) -> Motor | None:
    global _motor, _raiz, _tope
    _raiz = Path(cfg.dev_raiz).resolve()
    _tope = float(cfg.dev_tope)
    if _motor is None:
        _motor = abrir_motor(cfg)
        if _motor is not None:
            logger.info("Agente dev listo sobre %s (tope %.0f s).", _raiz, _tope)
    return _motor


def detener() -> None:
    global _motor
    _motor = None


def resolver_raiz(pedida: str) -> Path:
    """Comprueba que el encargo se queda dentro de la raíz permitida.

    Se resuelve antes de comparar: comparar cadenas sin resolver es exactamente
    como se cuela un `..`. Es la misma regla que en `memoria.py`, y por el mismo
    motivo — lo que llega puede venir de un correo.
    """
    if _raiz is None:
        raise RuntimeError("El agente dev no está iniciado; falta llamar a dev.iniciar().")
    if not pedida:
        return _raiz

    destino = (_raiz / pedida).resolve()
    if destino != _raiz and _raiz not in destino.parents:
        raise ValueError(f"{pedida!r} cae fuera de la raíz permitida ({_raiz}).")
    if not destino.is_dir():
        raise ValueError(f"{pedida!r} no es un directorio.")
    return destino


@registrar("dev")
async def _dev(trabajo: dict[str, Any]) -> dict[str, Any]:
    """Le encarga a Claude una tarea de código y devuelve lo que contestó.

    No pide confirmación: editar código es reversible y está en git, que es lo
    que dice §7 del plan. Lo irreversible —publicar, borrar— no está en la lista
    de lo que puede ejecutar.
    """
    peticion = trabajo.get("peticion") or {}
    instruccion = str(peticion.get("texto") or peticion.get("instruccion") or "").strip()
    if not instruccion:
        raise ValueError("Un encargo de `dev` necesita `texto`.")

    if _motor is None:
        raise RuntimeError(
            "No hay motor para `dev`. Instala Claude Code y comprueba que `claude` "
            "está en el PATH, o pon PERSEO_DEV_MOTOR=falso para probar el circuito."
        )

    raiz = resolver_raiz(str(peticion.get("directorio", "")))
    sesion = str(peticion.get("sesion", ""))

    logger.info("Encargo de dev en %s: %.120s", raiz, instruccion)
    resultado = await _motor.ejecutar(instruccion, raiz, _tope, sesion)

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
