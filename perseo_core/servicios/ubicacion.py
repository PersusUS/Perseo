"""Dónde está él, si él lo dice: la última ubicación compartida, y solo esa.

Instinct lee la ubicación del teléfono todo el rato. Aquí no: la ubicación llega
**cuando él la comparte** —un mensaje de ubicación por Telegram o WhatsApp, o un
atajo del iPhone que llama a `POST /ubicacion`— y se guarda **solo la última**,
sobrescribiendo la anterior, en `<datos>/ubicacion.json`. No hay historial de
dónde ha estado, porque un historial de sitios es lo que no se quiere tener el
día que alguien lo pida.

Sirve para lo que un asistente necesita saber: «resérvame algo cerca», «cuánto
tardo en llegar», «qué tiempo hace aquí». La herramienta `mi_ubicacion` se la da
al modelo con su antigüedad, porque una ubicación de ayer no es «aquí».
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

NOMBRE_FICHERO = "ubicacion.json"

#: Desde cuándo una ubicación deja de valer como «aquí».
HORAS_FRESCA = 2


def _ruta(directorio_datos: Path | str) -> Path:
    return Path(directorio_datos) / NOMBRE_FICHERO


def guardar(
    directorio_datos: Path | str,
    latitud: Any,
    longitud: Any,
    ahora: datetime,
    precision: Any = None,
    fuente: str = "",
) -> dict[str, Any]:
    try:
        lat, lon = float(latitud), float(longitud)
    except (TypeError, ValueError):
        raise ValueError("latitud y longitud tienen que ser números") from None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("esas coordenadas no están en la Tierra")
    dato = {
        "latitud": round(lat, 6),
        "longitud": round(lon, 6),
        "precision_m": round(float(precision)) if precision not in (None, "") else None,
        "fuente": str(fuente or "")[:40],
        "momento": ahora.isoformat(timespec="seconds"),
    }
    ruta = _ruta(directorio_datos)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(".tmp")
    temporal.write_text(json.dumps(dato), encoding="utf-8")
    temporal.replace(ruta)
    logger.info("Ubicación actualizada desde %s.", dato["fuente"] or "fuera")
    return dato


def ultima(directorio_datos: Path | str) -> dict[str, Any] | None:
    try:
        dato = json.loads(_ruta(directorio_datos).read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return dato if isinstance(dato, dict) and "latitud" in dato else None


def borrar(directorio_datos: Path | str) -> bool:
    try:
        _ruta(directorio_datos).unlink()
        return True
    except FileNotFoundError:
        return False


def describir(dato: dict[str, Any] | None, ahora: datetime) -> str:
    """Lo que ve el modelo: dónde, cuándo y un enlace al mapa."""
    if not dato:
        return (
            "No sé dónde está: no ha compartido su ubicación. Puede mandarla por Telegram o "
            "WhatsApp (adjuntar → ubicación)."
        )
    try:
        minutos = int((ahora - datetime.fromisoformat(dato["momento"])).total_seconds() // 60)
    except (KeyError, ValueError):
        minutos = -1
    hace = "hace un momento" if 0 <= minutos < 2 else (f"hace {minutos} min" if minutos < 120 else f"hace {minutos // 60} h")
    vieja = "" if 0 <= minutos < HORAS_FRESCA * 60 else " (ya no es de ahora: pregúntale si sigue ahí)"
    precision = f", ±{dato['precision_m']} m" if dato.get("precision_m") else ""
    fuente = f", por {dato['fuente']}" if dato.get("fuente") else ""
    lat, lon = dato["latitud"], dato["longitud"]
    return (
        f"Última ubicación compartida {hace}{fuente}{vieja}: {lat}, {lon}{precision}. "
        f"Mapa: https://www.google.com/maps?q={lat},{lon}"
    )
