"""Agente `correo`: el primero que trabaja solo.

Hasta la Fase C todo lo que hacía Perseo empezaba porque alguien se lo pedía. El
agente `correo` es el primero que no: un disparador mira el buzón cada tanto,
encola un trabajo con lo que ha encontrado, el triaje local lo clasifica y solo
si hay algo relevante sale un aviso. Ese es el criterio de la Fase D.

**El buzón es un puerto, no una integración.** El agente no sabe si detrás hay
Gmail, un IMAP o un fichero: pide mensajes y clasifica. Eso es lo que permite
verificar el circuito entero sin credenciales de Google —`BuzonFalso` lee un JSON
del disco— y lo que hará que enchufar Gmail más adelante sea escribir una clase,
no tocar el agente.

Dos reglas que gobiernan el módulo:

1. **Titular por Telegram, detalle por Tailscale.** El resultado del trabajo
   lleva remitentes y asuntos, y se lee en la web. Lo que sale por Telegram es
   `titular()`, que solo cuenta: "5 correos, 2 requieren acción". Ningún asunto
   viaja por un tercero.
2. **El agente no manda nada.** Clasificar es reversible; contestar no. En este
   corte de la Fase D no hay ninguna acción irreversible, así que no hace falta
   `NecesitaConfirmacion` todavía — llegará con los borradores. Cuando llegue,
   hay que recordar que **tras aprobar el agente se ejecuta desde el principio**,
   así que releer el buzón tiene que poder repetirse sin consecuencias (lo es:
   leer no cambia nada).

Ver bitacora/05_PLAN_PERSEO_V2.md §5 y §9 (Fase D).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from . import almacen, disparadores, triaje
from .agentes import registrar

logger = logging.getLogger(__name__)

#: Cuántos mensajes entran como mucho en un trabajo. Con un buzón que lleva
#: semanas sin mirarse, sin tope el primer arranque encolaría un trabajo de
#: cientos de clasificaciones que tardaría media hora en terminar.
TOPE_LOTE = 20


@dataclass(frozen=True)
class Mensaje:
    """Lo mínimo que hace falta para triar. Deliberadamente no es el correo entero."""

    id: str
    remitente: str
    asunto: str
    extracto: str = ""
    fecha: str = ""

    def a_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def desde_dict(cls, crudo: dict[str, Any]) -> "Mensaje":
        return cls(
            id=str(crudo.get("id", "")),
            remitente=str(crudo.get("remitente", "")),
            asunto=str(crudo.get("asunto", "")),
            extracto=str(crudo.get("extracto", "")),
            fecha=str(crudo.get("fecha", "")),
        )


class Buzon(Protocol):
    """De dónde salen los correos. Lo implementa cada proveedor."""

    async def nuevos(self) -> list[Mensaje]:
        """Mensajes que el buzón considera sin leer. Puede repetir: el disparador filtra."""
        ...


class BuzonFalso:
    """Buzón de fichero, para verificar el circuito sin credenciales.

    Lee el JSON en cada vuelta en vez de al arrancar, para que se le puedan
    añadir mensajes con el núcleo levantado — que es justo lo que hace falta para
    comprobar que un correo que llega **después** también dispara el aviso.
    """

    def __init__(self, ruta: Path) -> None:
        self._ruta = ruta

    async def nuevos(self) -> list[Mensaje]:
        return await asyncio.to_thread(self._leer)

    def _leer(self) -> list[Mensaje]:
        if not self._ruta.exists():
            return []
        try:
            crudo = json.loads(self._ruta.read_text(encoding="utf-8") or "[]")
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Buzón falso ilegible (%s).", e)
            return []
        if not isinstance(crudo, list):
            logger.warning("El buzón falso debería ser una lista de mensajes.")
            return []
        return [Mensaje.desde_dict(m) for m in crudo if isinstance(m, dict)]


def abrir_buzon(cfg: almacen.Configuracion) -> Buzon | None:
    """Devuelve el buzón configurado, o `None` si no hay ninguno.

    `None` no es un error: es el estado por defecto. Igual que con Telegram, el
    núcleo arranca sin correo y lo dice por registro — un canal que no está
    configurado no debe impedir que funcione todo lo demás.
    """
    if cfg.correo_buzon == "falso":
        return BuzonFalso(Path(cfg.correo_falso))
    if cfg.correo_buzon:
        # Aquí entrará `BuzonGmail` cuando haya credenciales OAuth. El agente y
        # el disparador no cambian: solo esta línea.
        logger.error("Buzón %r desconocido; se sigue sin correo.", cfg.correo_buzon)
    return None


# --------------------------------------------------------------------------- #
# El agente
# --------------------------------------------------------------------------- #

#: El triaje se comparte entre trabajos: abre una sesión HTTP contra Ollama y no
#: tiene sentido montarla y tirarla en cada correo. Lo pone en pie el arranque.
_triaje: triaje.Triaje | None = None


def iniciar(cfg: almacen.Configuracion) -> triaje.Triaje:
    global _triaje
    if _triaje is None:
        _triaje = triaje.Triaje(cfg)
    return _triaje


async def detener() -> None:
    global _triaje
    if _triaje is not None:
        await _triaje.cerrar()
        _triaje = None


@registrar("correo")
async def _correo(trabajo: dict[str, Any]) -> dict[str, Any]:
    """Tría un lote de mensajes con el modelo local.

    Devuelve el recuento y una línea por mensaje. El detalle se queda aquí, en la
    cola, que se lee por el tailnet; lo que sale por Telegram es `titular()`.
    """
    peticion = trabajo.get("peticion") or {}
    crudos = peticion.get("mensajes") or []
    mensajes = [Mensaje.desde_dict(m) for m in crudos if isinstance(m, dict)]
    if not mensajes:
        vacio = triaje.recontar([])
        return {"recuento": vacio, "clasificados": [], "titular": titular(vacio)}

    if _triaje is None:
        raise RuntimeError("El triaje no está iniciado; falta llamar a correo.iniciar().")

    clasificados: list[dict[str, Any]] = []
    clasificaciones: list[triaje.Clasificacion] = []
    for mensaje in mensajes:
        clasificacion = await _triaje.clasificar(mensaje.a_dict())
        clasificaciones.append(clasificacion)
        clasificados.append(
            {
                "id": mensaje.id,
                "remitente": mensaje.remitente,
                "asunto": mensaje.asunto,
                "clase": clasificacion.clase,
                "motivo": clasificacion.motivo,
                "del_modelo": clasificacion.del_modelo,
            }
        )

    recuento = triaje.recontar(clasificaciones)
    logger.info(
        "Triados %d correo(s): %d requieren acción, %d interesantes, %d ignorables, %d sin decidir.",
        recuento["total"],
        recuento[triaje.REQUIERE_ACCION],
        recuento[triaje.INTERESANTE],
        recuento[triaje.IGNORAR],
        recuento[triaje.NO_SEGURO],
    )
    # El titular viaja por Telegram y los clasificados no. Se compone aquí, que
    # es donde se sabe qué es contenido del correo y qué es un recuento.
    return {"recuento": recuento, "clasificados": clasificados, "titular": titular(recuento)}


# --------------------------------------------------------------------------- #
# El disparador
# --------------------------------------------------------------------------- #

_buzon: Buzon | None = None
_vistos: disparadores.Vistos | None = None


@disparadores.registrar("correo", intervalo=300)
async def _vigilar_buzon(ctx: disparadores.Contexto) -> None:
    """Mira el buzón y encola lo que no se haya triado todavía.

    La primera vuelta sobre un buzón nuevo **no clasifica nada**: apunta lo que
    hay y se queda callada. Sin eso, estrenar el disparador con un buzón de
    semanas encolaría veinte clasificaciones y mandaría un aviso de correos
    viejos, que es la mejor forma de que se desactive el día uno.
    """
    global _buzon, _vistos

    if _buzon is None:
        _buzon = abrir_buzon(ctx.cfg)
        if _buzon is None:
            raise disparadores.Retirarse("no hay buzón configurado")
        _vistos = disparadores.Vistos(
            ruta=Path(ctx.cfg.directorio_datos) / "correo_vistos.json"
        ).cargar()
    assert _vistos is not None

    mensajes = await _buzon.nuevos()
    pendientes = set(_vistos.sin_ver([m.id for m in mensajes]))
    nuevos = [m for m in mensajes if m.id in pendientes]
    if not nuevos:
        return

    if _vistos.estrenando:
        _vistos.anotar([m.id for m in nuevos])
        logger.info(
            "Buzón estrenado con %d correo(s) ya en él; se apuntan sin triar.", len(nuevos)
        )
        return

    lote = nuevos[:TOPE_LOTE]
    trabajo = await ctx.encolar("correo", {"accion": "triar", "mensajes": [m.a_dict() for m in lote]})
    # Se anotan al encolar y no al terminar: si el triaje falla, repetirlo en
    # bucle cada cinco minutos no lo arreglaría, y el trabajo fallido queda
    # visible en la cola para mirarlo.
    _vistos.anotar([m.id for m in lote])
    logger.info("Encolado el trabajo %s con %d correo(s) nuevos.", trabajo["id"], len(lote))


def titular(recuento: dict[str, Any]) -> str | None:
    """El aviso que sale por Telegram, o `None` si no merece molestar.

    Solo cuenta. Ni remitentes ni asuntos: eso es contenido del correo y viaja
    por el tailnet, no por un tercero. Y si no hay nada relevante no se manda
    nada — llenar el móvil de "0 correos interesantes" es la forma más rápida de
    que se silencie el canal.
    """
    accion = int(recuento.get(triaje.REQUIERE_ACCION, 0))
    interesantes = int(recuento.get(triaje.INTERESANTE, 0))
    dudosos = int(recuento.get(triaje.NO_SEGURO, 0))
    if accion + interesantes + dudosos == 0:
        return None

    total = int(recuento.get("total", 0))
    partes = []
    if accion:
        partes.append(f"{accion} requiere{'n' if accion != 1 else ''} acción")
    if interesantes:
        partes.append(f"{interesantes} interesante{'s' if interesantes != 1 else ''}")
    if dudosos:
        partes.append(f"{dudosos} sin decidir")
    return f"{total} correo{'s' if total != 1 else ''}: " + ", ".join(partes)
