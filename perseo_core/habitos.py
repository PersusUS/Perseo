"""El espejo del seguimiento de hábitos, para que el núcleo también lo sepa.

El seguimiento vive en el `localStorage` de la ventana de la app —es la
excepción consciente a «las caras no piensan»: no encola trabajo, no habla con
nadie y marcar una casilla no puede depender de que el núcleo esté encendido—.
Eso resolvía la pantalla y dejaba fuera a media casa: el Perseo de la llamada
podía leerlo porque corre EN esa ventana, pero el chat escrito, el triaje y
cualquier agente son Python y no tienen forma de mirar dentro de un navegador.

Así que la ventana manda aquí una copia cada vez que algo cambia, y esto la
guarda. Tres decisiones que conviene no deshacer sin pensarlo:

1. **Esto es un buzón, no una segunda contabilidad.** No cuenta casillas ni
   calcula rachas: recibe el texto YA redactado por `RealTime/src/lib/habitos.ts`
   y lo devuelve. Contar en los dos lados es la forma segura de que un día el
   chat diga 14 y la pantalla diga 15, y entonces las dos cifras dejan de valer.
2. **La copia se fecha y la fecha se cuenta.** Si la app lleva tres días
   cerrada, lo que hay aquí es de hace tres días y decirlo sin más sería
   mentir con datos viejos. `resumen()` lo antepone.
3. **Solo lectura hacia fuera.** Nadie marca hábitos desde el núcleo. Quien
   marca es el señor Persus en su pantalla; si desde aquí se pudiera escribir,
   habría dos escritores sobre el mismo dato y ningún árbitro.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

NOMBRE_FICHERO = "habitos.json"

#: A partir de cuántas horas la copia deja de darse por buena sin avisar. Un
#: día entero: por debajo de eso, lo normal es que la app haya estado abierta
#: hoy y el aviso sería ruido en cada respuesta.
HORAS_FRESCA = 24


def ruta(directorio_datos: Path) -> Path:
    return Path(directorio_datos) / NOMBRE_FICHERO


def guardar(directorio_datos: Path, texto: str, foto: dict[str, Any] | None = None) -> dict[str, Any]:
    """Deja la copia en disco, fechada, y devuelve lo que quedó guardado.

    Escritura atómica por el mismo motivo que en `biometria`: la ventana manda
    esto en cada cambio —o sea, mientras se pinta arrastrando, varias veces por
    segundo— y un corte a mitad de escritura dejaría un JSON partido que la
    siguiente lectura tendría que tirar.
    """
    copia = {
        "texto": str(texto or "").strip(),
        "foto": foto if isinstance(foto, dict) else None,
        "sellado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    destino = ruta(directorio_datos)
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_suffix(".tmp")
    temporal.write_text(json.dumps(copia, ensure_ascii=False, indent=1), encoding="utf-8")
    temporal.replace(destino)
    return copia


def cargar(directorio_datos: Path) -> dict[str, Any] | None:
    """Lo último que mandó la ventana, o nada.

    Un fichero roto vale lo mismo que uno que no está: nada. Devolver medio
    JSON haría que Perseo contase hábitos a medias, que es peor que decir que
    no los tiene.
    """
    try:
        copia = json.loads(ruta(directorio_datos).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError):
        logger.warning("La copia de hábitos está ilegible; se ignora.")
        return None
    if not isinstance(copia, dict) or not copia.get("texto"):
        return None
    return copia


def _antiguedad(sellado: str) -> str | None:
    """Cuánto hace de la copia, dicho en palabras, o nada si está fresca."""
    try:
        cuando = datetime.fromisoformat(sellado)
    except (TypeError, ValueError):
        return "de fecha desconocida"
    if cuando.tzinfo is None:
        cuando = cuando.replace(tzinfo=timezone.utc)

    horas = (datetime.now(timezone.utc) - cuando).total_seconds() / 3600
    if horas < HORAS_FRESCA:
        return None
    dias = int(horas // 24)
    if dias <= 1:
        return "de ayer"
    return f"de hace {dias} días"


def resumen(directorio_datos: Path) -> str:
    """Los hábitos para quien pregunta desde el núcleo, en castellano.

    Si la copia está pasada, lo dice DELANTE y no al final: un modelo que lee
    doce líneas de cifras y encuentra la salvedad abajo ya ha decidido cómo
    contestar. Puesta delante, la salvedad forma parte de la respuesta.
    """
    copia = cargar(directorio_datos)
    if copia is None:
        return (
            "No hay copia del seguimiento de hábitos en el núcleo. El señor Persus "
            "lo lleva en la pantalla de hábitos de la app, y la copia se manda desde "
            "ahí: si no ha abierto la app en este ordenador, aquí no consta nada. "
            "Dilo así en vez de suponer cómo va."
        )

    viejo = _antiguedad(str(copia.get("sellado", "")))
    if viejo:
        return (
            f"Atención: esta copia del seguimiento de hábitos es {viejo} —la app no se "
            f"ha abierto desde entonces—, así que puede que haya marcado cosas que aquí "
            f"no salen. Con esa salvedad:\n\n{copia['texto']}"
        )
    return str(copia["texto"])
