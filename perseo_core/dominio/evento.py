"""Un evento del calendario, sin saber de dónde sale.

Vive aquí y no en el agente `agenda` por un motivo que se pagaba caro: quien
trae los eventos de Google necesita este tipo, y el agente necesita a Google.
Con el tipo dentro del agente eso era un ciclo, y el ciclo se esquivaba
importando el módulo de Google **dentro de una función**. El tipo es del
dominio, no del agente: aquí abajo nadie tiene que esquivar nada.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class Evento:
    """Un evento del calendario. `inicio` es ISO-8601; con zona, mejor."""

    id: str
    titulo: str
    inicio: str
    fin: str = ""
    lugar: str = ""

    def a_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def desde_dict(cls, crudo: dict[str, Any]) -> "Evento":
        return cls(
            id=str(crudo.get("id", "")),
            titulo=str(crudo.get("titulo", "")),
            inicio=str(crudo.get("inicio", "")),
            fin=str(crudo.get("fin", "")),
            lugar=str(crudo.get("lugar", "")),
        )

    @property
    def momento(self) -> datetime | None:
        """El inicio como fecha, o `None` si no se puede leer.

        Se normaliza a UTC: el sistema acabará repartido entre varias máquinas y
        comparar horas locales entre ellas es una fuente de errores gratuita.
        """
        try:
            leido = datetime.fromisoformat(self.inicio.replace("Z", "+00:00"))
        except ValueError:
            return None
        if leido.tzinfo is None:
            leido = leido.astimezone()
        return leido.astimezone(timezone.utc)
