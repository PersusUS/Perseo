"""Bus de eventos en proceso.

Quien produce un evento no sabe quién lo escucha. Eso es lo que permite que en
la Fase D los disparadores (correo nuevo, evento de calendario, cron) publiquen
sin conocer las caras, y que en la Fase B la web del móvil y la app Tauri lean
el mismo flujo sin que el trabajador sepa que existen.

Decisión: las colas de los suscriptores están **acotadas**, y al llenarse se
descarta el evento más viejo en vez de bloquear al productor. Un móvil con la
pantalla apagada deja de consumir su cola; sin el tope, esa cola crecería sin
límite y acabaría comiéndose la memoria del proceso. Perder un evento de
progreso en un cliente dormido es aceptable — la cola de trabajos en SQLite
sigue siendo la fuente de verdad, y al reconectar se lee de ahí.

Ver bitacora/05_PLAN_PERSEO_V2.md §2.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)

# Cuántos eventos aguanta un suscriptor lento antes de empezar a perder los
# viejos. 256 da margen de sobra para una reconexión corta.
TOPE_COLA = 256


def _ahora() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class Evento:
    tipo: str
    datos: dict[str, Any] = field(default_factory=dict)
    momento: str = field(default_factory=_ahora)

    def a_dict(self) -> dict[str, Any]:
        return {"tipo": self.tipo, "datos": self.datos, "momento": self.momento}


class Bus:
    """Publicación y suscripción entre las piezas del núcleo."""

    def __init__(self, tope_cola: int = TOPE_COLA) -> None:
        self._colas: set[asyncio.Queue[Evento]] = set()
        self._tope = tope_cola

    @property
    def suscriptores(self) -> int:
        return len(self._colas)

    def publicar(self, tipo: str, **datos: Any) -> Evento:
        """Publica un evento. No bloquea nunca, ni siquiera con suscriptores atascados."""
        evento = Evento(tipo=tipo, datos=datos)
        for cola in self._colas:
            if cola.full():
                # Descartar el más viejo: para un cliente que vuelve, los
                # eventos recientes valen más que los rancios.
                with contextlib.suppress(asyncio.QueueEmpty):
                    cola.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                cola.put_nowait(evento)
        return evento

    @contextlib.asynccontextmanager
    async def suscribir(self) -> AsyncIterator[AsyncIterator[Evento]]:
        """Contexto que entrega un flujo de eventos y se da de baja al salir.

        Uso:
            async with bus.suscribir() as eventos:
                async for evento in eventos:
                    ...
        """
        cola: asyncio.Queue[Evento] = asyncio.Queue(maxsize=self._tope)
        self._colas.add(cola)
        logger.debug("Suscriptor añadido (%d en total).", len(self._colas))
        try:
            yield self._flujo(cola)
        finally:
            self._colas.discard(cola)
            logger.debug("Suscriptor retirado (%d en total).", len(self._colas))

    async def _flujo(self, cola: asyncio.Queue[Evento]) -> AsyncIterator[Evento]:
        while True:
            yield await cola.get()
