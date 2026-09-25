"""Lo que un recado sabe de sí mismo, y lo que deja escrito para seguir otro día.

Aparte de `recado.py` porque tiene su propia costura: el bucle decide qué hacer,
y esto es solo la foto de por dónde va —la conversación con el modelo, la página,
los síes ya dados— y el fichero donde se guarda cuando hay que parar a esperar
uno. Un recado que espera puede esperar horas, y en medio reiniciarse el núcleo.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..servicios import navegacion as nav
from .recado_puertos import compactar

#: Un punto de reanudación que nadie reclama en este tiempo es de un recado
#: rechazado o abandonado.
DIAS_PUNTO = 2


@dataclass
class Final:
    """Cómo acaba un recado: hecho, cumplida, sin novedad, atascado o sin pasos."""

    estado: str
    texto: str


@dataclass
class Estado:
    """Todo lo que hace falta para seguir un recado en otro momento."""

    contents: list[dict[str, Any]]
    pasos: int = 0
    url: str = ""
    #: El ref de la última instantánea y lo que es. De aquí sale el nombre de lo
    #: que se pulsa, no de lo que escribe el modelo.
    elementos: dict[str, nav.Elemento] = field(default_factory=dict)
    #: Huellas de lo exterior que el dueño ya aprobó en este recado.
    aprobadas: list[str] = field(default_factory=list)
    tarjeta_usada: bool = False
    #: El menor tope de las tarjetas metidas en este recado, si alguna lo tiene.
    limite: float | None = None
    #: Si el recado comprueba una vigilancia: su id, y si ya dijo que se cumple.
    vigilancia: str = ""
    cumplida: bool = False

    def a_json(self) -> dict[str, Any]:
        """Sin `elementos`: son de la página de entonces, y al seguir se miran de nuevo."""
        return {
            "contents": self.contents,
            "pasos": self.pasos,
            "url": self.url,
            "aprobadas": self.aprobadas,
            "tarjeta_usada": self.tarjeta_usada,
            "limite": self.limite,
            "vigilancia": self.vigilancia,
            "cumplida": self.cumplida,
        }

    @classmethod
    def de_json(cls, datos: dict[str, Any]) -> "Estado":
        return cls(
            contents=list(datos.get("contents") or []),
            pasos=int(datos.get("pasos") or 0),
            url=str(datos.get("url") or ""),
            aprobadas=list(datos.get("aprobadas") or []),
            tarjeta_usada=bool(datos.get("tarjeta_usada")),
            limite=datos.get("limite"),
            vigilancia=str(datos.get("vigilancia") or ""),
            cumplida=bool(datos.get("cumplida")),
        )


class Puntos:
    """Los puntos de reanudación, uno por trabajo, en `<datos>/recados/<id>.json`."""

    def __init__(self, carpeta: Path) -> None:
        self.carpeta = Path(carpeta)
        self.carpeta.mkdir(parents=True, exist_ok=True)

    def _ruta(self, id_trabajo: int) -> Path:
        return self.carpeta / f"{int(id_trabajo)}.json"

    def guardar(self, id_trabajo: int, estado: Estado, turno: dict[str, Any]) -> None:
        compactar(estado.contents)
        ruta = self._ruta(id_trabajo)
        temporal = ruta.with_suffix(".tmp")
        temporal.write_text(json.dumps({"estado": estado.a_json(), "turno": turno}, ensure_ascii=False), encoding="utf-8")
        os.replace(temporal, ruta)

    def cargar(self, id_trabajo: int) -> dict[str, Any] | None:
        try:
            return json.loads(self._ruta(id_trabajo).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

    def borrar(self, id_trabajo: int) -> None:
        try:
            self._ruta(id_trabajo).unlink()
        except OSError:
            pass

    def limpiar_viejos(self) -> None:
        corte = time.time() - DIAS_PUNTO * 86400
        for fichero in self.carpeta.glob("*.json"):
            try:
                if fichero.stat().st_mtime < corte:
                    fichero.unlink()
            except OSError:
                pass
