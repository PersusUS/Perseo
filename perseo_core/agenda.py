"""Agente `agenda`: lo que viene, avisado antes de que llegue.

Mismo patrón que el correo, y a propósito: **el calendario es un puerto**. El
agente pregunta qué hay cerca y no sabe si detrás está Google Calendar o un
fichero. Hoy solo hay `CalendarioFalso`, que lee un JSON; cuando haya
credenciales OAuth se escribe `CalendarioGoogle` con el mismo método y no cambia
nada más.

Dos cosas que lo diferencian del correo:

1. **No hace falta el modelo local.** Un evento que empieza dentro de una hora es
   relevante por definición: no hay nada que clasificar. Esto no gasta ni cuota
   de Gemini ni GPU.
2. **Se avisa una vez por evento.** La marca de agua guarda los identificadores ya
   avisados, así que mover el reloj hacia delante no repite el aviso cada diez
   minutos. Un canal que repite se silencia, y entonces tampoco avisa de lo que
   importa.

La regla del canal se respeta igual: por Telegram sale **cuántos y a qué hora**,
nunca el título del evento — eso es contenido, y se lee por el tailnet.


"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

from . import almacen, disparadores, google_api
from .agentes import registrar
from .dominio.evento import Evento

logger = logging.getLogger(__name__)

#: Cuántos eventos entran como mucho en un aviso.
TOPE_LOTE = 20


class Calendario(Protocol):
    """De dónde salen los eventos. Lo implementa cada proveedor."""

    async def proximos(self, horizonte: timedelta) -> list[Evento]:
        """Eventos que empiezan de aquí a `horizonte`. Puede repetir: el disparador filtra."""
        ...


class CalendarioFalso:
    """Calendario de fichero, para verificar el circuito sin credenciales."""

    def __init__(self, ruta: Path) -> None:
        self._ruta = ruta

    async def proximos(self, horizonte: timedelta) -> list[Evento]:
        return await asyncio.to_thread(self._leer, horizonte)

    def _leer(self, horizonte: timedelta) -> list[Evento]:
        if not self._ruta.exists():
            return []
        try:
            crudo = json.loads(self._ruta.read_text(encoding="utf-8") or "[]")
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Calendario falso ilegible (%s).", e)
            return []
        if not isinstance(crudo, list):
            logger.warning("El calendario falso debería ser una lista de eventos.")
            return []

        ahora = datetime.now(timezone.utc)
        limite = ahora + horizonte
        eventos = [Evento.desde_dict(e) for e in crudo if isinstance(e, dict)]
        # Los que ya han empezado no se avisan: avisar de algo a lo que llegas
        # tarde es ruido, no información.
        dentro = [e for e in eventos if e.momento is not None and ahora <= e.momento <= limite]
        return sorted(dentro, key=lambda e: e.momento or ahora)


def abrir_calendario(cfg: almacen.Configuracion) -> Calendario | None:
    """Devuelve el calendario configurado, o `None` si no hay ninguno."""
    if cfg.agenda_origen == "falso":
        return CalendarioFalso(Path(cfg.agenda_falsa))

    if cfg.agenda_origen == "google":
        try:
            return google_api.CalendarioGoogle(google_api.credenciales(cfg))
        except google_api.SinCredenciales as e:
            logger.error("Google Calendar pedido pero sin credenciales: %s", e)
            return None

    if cfg.agenda_origen:
        logger.error("Calendario %r desconocido; se sigue sin agenda.", cfg.agenda_origen)
    return None


# --------------------------------------------------------------------------- #
# El agente
# --------------------------------------------------------------------------- #

#: Horizonte por defecto de «qué hay próximo»: un día. Y el techo: una semana,
#: porque «¿qué tengo este mes?» no cabe en una respuesta hablada.
HORAS_POR_DEFECTO = 24
HORAS_MAXIMAS = 24 * 7

_cfg: almacen.Configuracion | None = None
_calendario: Calendario | None = None


def iniciar(cfg: almacen.Configuracion) -> None:
    """Guarda la configuración para que el agente pueda abrir el calendario.

    Lo mismo que hace `correo.iniciar` con su buzón: la cara que ejecuta
    trabajos necesita saber de dónde tirar aunque el disparador no haya corrido
    nunca — con `PERSEO_DISPARADORES` vacío, el agente sigue atendiendo.
    """
    global _cfg
    _cfg = cfg


async def detener() -> None:
    global _cfg, _calendario
    _cfg = None
    _calendario = None


def calendario() -> Calendario | None:
    """El calendario configurado, abierto la primera vez y cacheado después.

    Compartido por el agente y el disparador: dos conexiones al mismo sitio
    sería otra cosa que mantener viva.
    """
    global _calendario
    if _calendario is None:
        if _cfg is None:
            raise RuntimeError("El agente agenda no está iniciado; falta agenda.iniciar(cfg).")
        _calendario = abrir_calendario(_cfg)
    return _calendario


@registrar("agenda")
async def _agenda(trabajo: dict[str, Any]) -> dict[str, Any]:
    """Prepara el aviso de lo que viene, o lee lo próximo si se le pide.

    Solo lee y ordena: no crea ni mueve nada. Crear y mover son de la Fase E y
    pasarán por `NecesitaConfirmacion`, porque mover el evento equivocado en el
    calendario de alguien no tiene deshacer cómodo.
    """
    peticion = trabajo.get("peticion") or {}

    if str(peticion.get("accion", "avisar")).strip().lower() == "proximos":
        return await _proximos(peticion)

    crudos = peticion.get("eventos") or []
    eventos = [Evento.desde_dict(e) for e in crudos if isinstance(e, dict)]
    return {
        "accion": "avisar",
        "eventos": [e.a_dict() for e in eventos],
        "titular": titular(eventos),
    }


async def _proximos(peticion: dict[str, Any]) -> dict[str, Any]:
    """Lo que empieza de aquí a `horas`, leído del calendario de verdad."""
    calendario_activo = calendario()
    if calendario_activo is None:
        raise RuntimeError(
            "No hay calendario configurado: hace falta PERSEO_AGENDA=google (o falso)."
        )

    try:
        horas = float(peticion.get("horas") or HORAS_POR_DEFECTO)
    except (TypeError, ValueError):
        horas = HORAS_POR_DEFECTO
    horas = max(1.0, min(horas, HORAS_MAXIMAS))

    # Tope de lote como el disparador: una lista de veinte eventos tampoco se
    # lee en voz alta.
    eventos = (await calendario_activo.proximos(timedelta(hours=horas)))[:TOPE_LOTE]
    return {
        "accion": "proximos",
        "horas": horas,
        "eventos": [e.a_dict() for e in eventos],
        "titular": titular(eventos),
    }


def titular(eventos: list[Evento]) -> str | None:
    """El aviso que sale por Telegram, o `None` si no hay nada que decir.

    Cuántos y a qué hora empieza el primero. **El título no sale**: es contenido
    del calendario, y para eso está el enlace a la web por el tailnet.
    """
    if not eventos:
        return None

    primero = eventos[0].momento
    cuando = primero.astimezone().strftime("%H:%M") if primero else "pronto"
    if len(eventos) == 1:
        return f"1 evento a las {cuando}"
    return f"{len(eventos)} eventos; el primero a las {cuando}"


# --------------------------------------------------------------------------- #
# El disparador
# --------------------------------------------------------------------------- #

_avisados: disparadores.Vistos | None = None


@disparadores.registrar("agenda", intervalo=600)
async def _vigilar_calendario(ctx: disparadores.Contexto) -> None:
    """Encola un aviso cuando algo se acerca, y solo una vez por evento."""
    global _avisados

    iniciar(ctx.cfg)
    if calendario() is None:
        raise disparadores.Retirarse("no hay calendario configurado")
    if _avisados is None:
        _avisados = disparadores.Vistos(
            ruta=Path(ctx.cfg.directorio_datos) / "agenda_avisados.json"
        ).cargar()
    assert _avisados is not None

    eventos = await _calendario.proximos(timedelta(minutes=ctx.cfg.agenda_antelacion))
    pendientes = set(_avisados.sin_ver([e.id for e in eventos]))
    nuevos = [e for e in eventos if e.id in pendientes][:TOPE_LOTE]
    if not nuevos:
        return

    # Aquí no hay estreno que valga, al revés que en el correo: un evento que
    # empieza dentro de una hora hay que avisarlo aunque el núcleo acabe de
    # arrancar. Lo viejo ya lo descarta el propio calendario.
    trabajo = await ctx.encolar("agenda", {"accion": "avisar", "eventos": [e.a_dict() for e in nuevos]})
    _avisados.anotar([e.id for e in nuevos])
    logger.info("Encolado el trabajo %s con %d evento(s) próximos.", trabajo["id"], len(nuevos))
