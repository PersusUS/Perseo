"""Router local, registro de agentes y trabajador de la cola.

El router es la primera pieza donde se aplica el reparto del plan (§3): un
modelo pequeño en la GPU decide **a dónde va** cada petición, y solo lo que
necesita razonar de verdad sale a la nube. No es una optimización — con 250
peticiones diarias en el plan gratuito de Gemini, es lo que hace que el sistema
aguante un día entero.

La fiabilidad del router no viene del tamaño del modelo sino de la gramática:
Ollama acepta un esquema JSON en `format` y aplica decodificación restringida,
así que la respuesta **siempre** valida. Un 4B puede rellenar mal un campo, pero
no puede devolver algo que no sea del esquema. Por eso una de las salidas
posibles es "no estoy seguro": el modelo tiene dónde escalar en vez de inventar.

Si Ollama no está levantado, el router no rompe nada: cae a "encolar", que es la
opción segura — el trabajo queda registrado y visible en vez de perderse.

Ver bitacora/05_PLAN_PERSEO_V2.md §3 y §9 (Fase A).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import aiohttp

from . import almacen
from .bus import Bus

logger = logging.getLogger(__name__)

#: Un agente recibe **el trabajo entero**, no solo la petición: necesita ver la
#: confirmación para saber si ya le han dicho que sí, y el identificador para
#: poder informar de su progreso más adelante.
Manejador = Callable[[dict[str, Any]], Awaitable[Any]]

#: Agentes disponibles, por nombre. En la Fase A solo hay uno de prueba; las
#: fases D y E añaden `correo`, `agenda`, `memoria`, `dev`, `web` y `pc`.
REGISTRO: dict[str, Manejador] = {}


class NecesitaConfirmacion(Exception):
    """La lanza un agente cuando lo siguiente que haría es irreversible.

    No es un error: el trabajo se para, queda en `esperando` con la pregunta
    guardada, y cuando alguien contesta vuelve a la cola. El agente se ejecuta
    entonces desde el principio, así que **todo lo que haga antes de lanzarla
    tiene que poder repetirse sin consecuencias**.
    """

    def __init__(self, resumen: str, detalle: str = "") -> None:
        super().__init__(resumen)
        self.resumen = resumen
        self.detalle = detalle


def aprobado(trabajo: dict[str, Any]) -> bool:
    """Si este trabajo ya trae un sí. Los agentes lo consultan al empezar."""
    confirmacion = trabajo.get("confirmacion") or {}
    return isinstance(confirmacion, dict) and confirmacion.get("decision") == "aprobado"


def registrar(nombre: str) -> Callable[[Manejador], Manejador]:
    """Decorador que da de alta un agente en el registro."""

    def decorador(funcion: Manejador) -> Manejador:
        if nombre in REGISTRO:
            raise ValueError(f"El agente {nombre!r} ya está registrado.")
        REGISTRO[nombre] = funcion
        return funcion

    return decorador


@registrar("eco")
async def _eco(trabajo: dict[str, Any]) -> dict[str, Any]:
    """Agente de prueba de la Fase A: devuelve lo que recibe, con una pausa.

    Existe para validar el circuito completo —encolar, reclamar, ejecutar,
    completar, notificar— sin depender todavía de ninguna integración. La pausa
    hace que el trabajo sea visible en la cola en vez de completarse antes de
    que llegue el primer sondeo.
    """
    await asyncio.sleep(1)
    return {"texto": trabajo["peticion"].get("texto", "")}


@registrar("simulacro")
async def _simulacro(trabajo: dict[str, Any]) -> dict[str, Any]:
    """Agente de prueba del camino de confirmación de la Fase B.

    Es a las aprobaciones lo que `eco` es a la cola: no hace nada de verdad,
    pero recorre el circuito entero —parar, preguntar, esperar la respuesta,
    seguir— para que esté probado antes de que lo use un agente que sí borra
    cosas. Se jubila cuando la Fase E traiga la política de confirmación real.
    """
    peticion = trabajo["peticion"]
    accion = str(peticion.get("accion", "una acción irreversible"))

    if not aprobado(trabajo):
        raise NecesitaConfirmacion(
            resumen=f"¿Confirmas: {accion}?",
            detalle=peticion.get("detalle", ""),
        )

    await asyncio.sleep(0.2)
    return {"texto": f"Hecho: {accion}"}


# --------------------------------------------------------------------------- #
# Router local
# --------------------------------------------------------------------------- #

#: Esquema de la decisión del router. La gramática garantiza la *forma*; el
#: contenido sigue siendo responsabilidad del modelo, de ahí `no_seguro`.
ESQUEMA_RUTA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "destino": {
            "type": "string",
            "enum": ["responder", "encolar", "no_seguro"],
            "description": (
                "responder: se contesta al momento sin trabajo de fondo. "
                "encolar: hace falta un agente y puede tardar. "
                "no_seguro: no lo tienes claro."
            ),
        },
        "agente": {"type": "string"},
        "respuesta": {"type": "string"},
        "motivo": {"type": "string"},
    },
    "required": ["destino", "agente", "motivo"],
}

_INSTRUCCIONES_ROUTER = """\
Eres el enrutador de un asistente personal. No resuelves la petición: decides a dónde va.

Agentes disponibles: {agentes}

Elige `destino`:
- "responder" si es un saludo, una confirmación o algo trivial que no necesita ningún agente.
  En ese caso rellena `respuesta` con lo que hay que contestar.
- "encolar" si hace falta trabajo real. Rellena `agente` con el más adecuado de la lista.
- "no_seguro" si dudas. Es una respuesta válida y preferible a inventarte un agente.

`motivo`: una frase corta explicando la decisión.
"""


@dataclass(frozen=True)
class Ruta:
    destino: str
    agente: str
    motivo: str
    respuesta: str = ""

    @property
    def hay_que_encolar(self) -> bool:
        # `no_seguro` se trata como encolar: ante la duda, que quede registrado
        # y visible en la cola en lugar de contestar cualquier cosa.
        return self.destino in ("encolar", "no_seguro")


class Router:
    """Decide el destino de cada petición usando el modelo local."""

    def __init__(self, cfg: almacen.Configuracion, agente_por_defecto: str = "eco") -> None:
        self._cfg = cfg
        self._agente_por_defecto = agente_por_defecto
        self._sesion: aiohttp.ClientSession | None = None
        self.disponible: bool | None = None  # None = todavía sin comprobar

    async def abrir(self) -> None:
        if self._sesion is None:
            self._sesion = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))

    async def cerrar(self) -> None:
        if self._sesion is not None:
            await self._sesion.close()
            self._sesion = None

    def _respaldo(self, motivo: str) -> Ruta:
        return Ruta(destino="encolar", agente=self._agente_por_defecto, motivo=motivo)

    async def decidir(self, texto: str) -> Ruta:
        """Devuelve la ruta para una petición. Nunca lanza: ante fallo, encola."""
        if self._sesion is None:
            await self.abrir()
        assert self._sesion is not None

        agentes = ", ".join(sorted(REGISTRO)) or self._agente_por_defecto
        cuerpo = {
            "model": self._cfg.modelo_router,
            "stream": False,
            "format": ESQUEMA_RUTA,
            # Sin esto, un modelo híbrido de razonamiento (la familia Qwen3 lo
            # es) intenta pensar antes de responder, y el razonamiento choca con
            # la gramática: la petición se queda colgada minutos y `content`
            # llega vacío. El router no necesita pensar — clasifica.
            "think": False,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": _INSTRUCCIONES_ROUTER.format(agentes=agentes)},
                {"role": "user", "content": texto},
            ],
        }

        try:
            async with self._sesion.post(
                f"{self._cfg.url_ollama}/api/chat", json=cuerpo
            ) as respuesta:
                if respuesta.status != 200:
                    detalle = (await respuesta.text())[:200]
                    self.disponible = False
                    return self._respaldo(f"Ollama respondió {respuesta.status}: {detalle}")
                datos = await respuesta.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            # Caso normal en un portátil: Ollama no está arrancado. No es un
            # error del sistema, es una capacidad que hoy no está.
            if self.disponible is not False:
                logger.warning("Router local no disponible (%s). Se encolará todo.", e)
            self.disponible = False
            return self._respaldo(f"Router local no disponible: {e}")

        self.disponible = True
        crudo = (datos.get("message") or {}).get("content", "")
        try:
            decision = json.loads(crudo)
        except json.JSONDecodeError:
            # No debería ocurrir con `format` puesto — si ocurre, la versión de
            # Ollama no está aplicando la gramática y conviene saberlo.
            logger.error("El router devolvió algo que no es JSON: %.200s", crudo)
            return self._respaldo("El router no devolvió JSON válido")

        agente = decision.get("agente") or self._agente_por_defecto
        if agente not in REGISTRO:
            # Esquema válido, contenido equivocado: exactamente el fallo que un
            # modelo pequeño comete. Se corrige aquí, no se propaga.
            logger.info("El router eligió un agente inexistente (%r); se usa el de respaldo.", agente)
            agente = self._agente_por_defecto

        return Ruta(
            destino=decision.get("destino", "encolar"),
            agente=agente,
            motivo=decision.get("motivo", ""),
            respuesta=decision.get("respuesta", ""),
        )


# --------------------------------------------------------------------------- #
# Trabajador
# --------------------------------------------------------------------------- #


class Trabajador:
    """Reclama trabajos de la cola y los ejecuta, uno detrás de otro.

    Las llamadas a `almacen` son de SQLite y bloquean; van por `to_thread` para
    no parar el bucle de eventos mientras la API sigue atendiendo peticiones.
    """

    def __init__(
        self,
        bus: Bus,
        intervalo: float = 0.5,
        agentes: tuple[str, ...] | None = None,
        excluir: tuple[str, ...] = (),
        nombre: str = "trabajador",
    ) -> None:
        self._bus = bus
        self._intervalo = intervalo
        # Dos carriles desde la Fase E: uno para lo corto y otro para `dev`, que
        # puede tardar minutos. Con un solo trabajador, un encargo de código
        # dejaba el correo sin triar mientras durase.
        self._agentes = agentes
        self._excluir = excluir
        self._nombre = nombre
        self._parar = asyncio.Event()

    def detener(self) -> None:
        self._parar.set()

    async def ejecutar(self) -> None:
        atendidos = self._agentes or tuple(a for a in sorted(REGISTRO) if a not in self._excluir)
        logger.info("Trabajador %s en marcha (agentes: %s).", self._nombre, ", ".join(atendidos))
        while not self._parar.is_set():
            trabajo = await asyncio.to_thread(almacen.reclamar, self._agentes, self._excluir)
            if trabajo is None:
                # Sondeo simple. Cuando la Fase D traiga disparadores reales, el
                # bus podrá despertar al trabajador y esto será solo el respaldo.
                try:
                    await asyncio.wait_for(self._parar.wait(), timeout=self._intervalo)
                except asyncio.TimeoutError:
                    pass
                continue

            await self._ejecutar_uno(trabajo)

        logger.info("Trabajador %s detenido.", self._nombre)

    async def _ejecutar_uno(self, trabajo: dict[str, Any]) -> None:
        id_trabajo = int(trabajo["id"])
        nombre = str(trabajo["agente"])
        self._bus.publicar("trabajo.reclamado", trabajo=trabajo)

        manejador = REGISTRO.get(nombre)
        if manejador is None:
            error = f"Agente desconocido: {nombre}"
            logger.error("%s (trabajo %d)", error, id_trabajo)
            fallido = await asyncio.to_thread(almacen.fallar, id_trabajo, error)
            self._bus.publicar("trabajo.fallido", trabajo=fallido)
            return

        try:
            resultado = await manejador(trabajo)
        except NecesitaConfirmacion as pregunta:
            # Ni hecho ni fallido: el trabajo se queda esperando un sí. Quien lo
            # dé puede ser la web o, más adelante, Telegram.
            esperando = await asyncio.to_thread(
                almacen.pedir_confirmacion, id_trabajo, pregunta.resumen, pregunta.detalle
            )
            self._bus.publicar("trabajo.espera_confirmacion", trabajo=esperando)
            logger.info("Trabajo %d esperando confirmación: %s", id_trabajo, pregunta.resumen)
            return
        except asyncio.CancelledError:
            # Cierre del proceso: devolver el trabajo a la cola para que el
            # próximo arranque lo recoja, en vez de marcarlo como fallido.
            await asyncio.to_thread(almacen.recuperar_huerfanos)
            raise
        except Exception as e:
            logger.exception("El agente %s falló en el trabajo %d", nombre, id_trabajo)
            fallido = await asyncio.to_thread(almacen.fallar, id_trabajo, f"{type(e).__name__}: {e}")
            self._bus.publicar("trabajo.fallido", trabajo=fallido)
            return

        hecho = await asyncio.to_thread(almacen.completar, id_trabajo, resultado)
        self._bus.publicar("trabajo.hecho", trabajo=hecho)
        logger.info("Trabajo %d completado por %s.", id_trabajo, nombre)
