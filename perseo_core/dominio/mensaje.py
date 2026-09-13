"""Un correo reducido a lo que hace falta para decidir qué hacer con él.

Aquí abajo por lo mismo que `Evento`: lo construye quien habla con Gmail y lo
consume el agente `correo`, y tenerlo dentro del agente era el otro medio ciclo.
Sigue siendo deliberadamente pobre —remitente, asunto, un extracto— porque lo
que viaja al clasificador es esto y no el correo entero.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Mensaje:
    """Lo mínimo que hace falta para triar. Deliberadamente no es el correo entero."""

    id: str
    remitente: str
    asunto: str
    extracto: str = ""
    fecha: str = ""
    #: El hilo al que pertenece, cuando el buzón lo sabe. Lo usa el borrador para
    #: que la respuesta cuelgue de la conversación en vez de nacer suelta.
    hilo: str = ""

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
            hilo=str(crudo.get("hilo", "")),
        )
