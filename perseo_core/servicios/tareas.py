"""El espejo del tablero de tareas, para que el núcleo también lo sepa.

Hermano de `habitos.py`, y por la misma razón. El corcho vive en el
`localStorage` de la ventana de la app —es la excepción consciente a «las caras
no piensan»: no encola trabajo, no habla con nadie y arrastrar una nota no
puede depender de que el núcleo esté encendido—. Eso resolvía la pantalla y
dejaba fuera a media casa: el Perseo de la llamada podía leer el tablero porque
corre EN esa ventana, pero el chat escrito, el triaje y cualquier agente son
Python y no tienen forma de mirar dentro de un navegador.

Así que la ventana manda aquí una copia cada vez que algo cambia, y esto la
guarda. Las tres decisiones son las de los hábitos, y tampoco conviene
deshacerlas sin pensarlo:

1. **Esto es un buzón, no un segundo tablero.** No cuenta notas ni decide qué
   está atascado: recibe el texto YA redactado por `RealTime/src/lib/tareas.ts`
   y lo devuelve. Contar en los dos lados es la forma segura de que un día el
   chat diga cuatro pendientes y la pantalla enseñe cinco, y entonces las dos
   cifras dejan de valer.
2. **La copia se fecha y la fecha se cuenta.** Si la app lleva tres días
   cerrada, lo que hay aquí es de hace tres días; decir «tienes esto pendiente»
   con eso sería mentir con datos viejos. `resumen()` lo antepone.
3. **El núcleo no escribe el tablero: lo pide.** El señor Persus quiso
   (2026-09-03) que Perseo también clavara y moviera notas, y eso choca de
   frente con lo anterior: el tablero está en una ventana que puede estar
   cerrada, y dos escritores sobre el mismo almacén sin árbitro es la receta de
   perder notas. La salida es que aquí no se escriba nada del tablero — se
   **encola una orden**, y la ventana, que sigue siendo el único escritor, la
   recoge y la aplica cuando está abierta (`encolar` y `recoger`).

   Lo que se paga por esto: una orden dada con la app cerrada tarda en cumplirse
   lo que tarde él en abrirla. Es el precio correcto. La alternativa —escribir
   una copia en el núcleo y fundirla luego con la de la ventana— es inventarse
   un segundo tablero y un árbitro para cuando discrepen, y eso acaba en notas
   duplicadas o desaparecidas, que es exactamente de lo que va todo este
   fichero.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

NOMBRE_FICHERO = "tareas.json"

#: Las órdenes que esperan a que la ventana las recoja. Fichero aparte del
#: espejo a propósito: uno es lo que hay y el otro lo que se ha pedido, y
#: mezclarlos haría que borrar una cosa se llevara la otra.
NOMBRE_ORDENES = "tareas_ordenes.json"

#: Cuántas órdenes se guardan sin recoger. Con la app cerrada una semana, lo que
#: sobra son las viejas: se tira la más antigua. Un buzón que crece sin tope
#: acaba aplicando de golpe cuarenta notas el día que se abre la app.
TOPE_ORDENES = 50

#: Las columnas del tablero. Repetidas aquí y no importadas de ningún sitio
#: porque el que manda es `RealTime/src/lib/tareas.ts` y esto es un portero: si
#: algún día se añade una columna, esta lista se queda corta y la orden se
#: rechaza, que es lo que tiene que pasar.
COLUMNAS = ("sin_hacer", "en_proceso", "completadas", "papelera")

#: A partir de cuántas horas la copia deja de darse por buena sin avisar. Un
#: día entero, como en los hábitos: por debajo de eso lo normal es que la app
#: haya estado abierta hoy y el aviso sería ruido en cada respuesta.
HORAS_FRESCA = 24


def ruta(directorio_datos: Path) -> Path:
    return Path(directorio_datos) / NOMBRE_FICHERO


def ruta_ordenes(directorio_datos: Path) -> Path:
    return Path(directorio_datos) / NOMBRE_ORDENES


def guardar(directorio_datos: Path, texto: str, foto: dict[str, Any] | None = None) -> dict[str, Any]:
    """Deja la copia en disco, fechada, y devuelve lo que quedó guardado.

    Escritura atómica por lo mismo que en `habitos`: la ventana manda esto en
    cada cambio del tablero y un corte a mitad de escritura dejaría un JSON
    partido que la siguiente lectura tendría que tirar.
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
    JSON haría que Perseo leyese medio tablero, que es peor que decir que no lo
    tiene: media lista de pendientes parece una lista entera.
    """
    try:
        copia = json.loads(ruta(directorio_datos).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError):
        logger.warning("La copia del tablero de tareas está ilegible; se ignora.")
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
    """El tablero para quien pregunta desde el núcleo, en castellano.

    Si la copia está pasada, lo dice DELANTE y no al final: un modelo que lee
    una lista de pendientes y encuentra la salvedad abajo ya ha decidido cómo
    contestar. Puesta delante, la salvedad forma parte de la respuesta — y aquí
    importa más que en los hábitos, porque una tarea que él ya cerró ayer es lo
    peor que se le puede recordar.
    """
    copia = cargar(directorio_datos)
    if copia is None:
        return (
            "No hay copia del tablero de tareas en el núcleo. El señor Persus lo "
            "lleva en la pantalla de tareas de la app, y la copia se manda desde "
            "ahí: si no ha abierto la app en este ordenador, aquí no consta nada. "
            "Dilo así en vez de suponer qué tiene pendiente."
        )

    viejo = _antiguedad(str(copia.get("sellado", "")))
    if viejo:
        return (
            f"Atención: esta copia del tablero de tareas es {viejo} —la app no se ha "
            f"abierto desde entonces—, así que puede haber movido o cerrado notas que "
            f"aquí siguen abiertas. Con esa salvedad:\n\n{copia['texto']}"
        )
    return str(copia["texto"])


# --------------------------------------------------------------------------- #
# Lo que se le pide a la ventana
#
# El núcleo no toca el tablero: deja aquí lo que Perseo quiere hacer con él y la
# ventana —el único escritor— lo recoge y lo aplica. Ver el punto 3 de la
# cabecera para por qué no se escribe directamente.
# --------------------------------------------------------------------------- #


def _guardar_ordenes(directorio_datos: Path, ordenes: list[dict[str, Any]]) -> None:
    """Deja la cola en disco, atómica como el espejo."""
    destino = ruta_ordenes(directorio_datos)
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_suffix(".tmp")
    temporal.write_text(json.dumps(ordenes, ensure_ascii=False, indent=1), encoding="utf-8")
    temporal.replace(destino)


def ordenes(directorio_datos: Path) -> list[dict[str, Any]]:
    """Lo que hay pendiente, sin tocarlo. Un fichero roto es una cola vacía."""
    try:
        cola = json.loads(ruta_ordenes(directorio_datos).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (json.JSONDecodeError, OSError):
        logger.warning("La cola de órdenes del tablero está ilegible; se ignora.")
        return []
    return [o for o in cola if isinstance(o, dict)] if isinstance(cola, list) else []


def encolar(directorio_datos: Path, accion: str, titulo: str, **extra: Any) -> dict[str, Any]:
    """Apunta una orden para la ventana y devuelve la que quedó apuntada.

    Se valida aquí y no en la ventana porque aquí es donde se sabe decir que no:
    una orden mal formada rechazada en el núcleo se le puede contar al modelo en
    el acto, mientras que una que viaja y muere al otro lado se pierde en
    silencio y Perseo se queda diciendo que lo hizo.
    """
    accion = str(accion or "").strip()
    titulo = str(titulo or "").strip()
    if accion not in ("crear", "mover"):
        raise ValueError(f"Acción desconocida: «{accion}»")
    if not titulo:
        raise ValueError("Una nota sin título no es una nota")

    orden: dict[str, Any] = {
        "accion": accion,
        "titulo": titulo,
        "pedida": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    columna = str(extra.get("columna") or "").strip()
    if columna:
        if columna not in COLUMNAS:
            raise ValueError(f"Columna desconocida: «{columna}»")
        orden["columna"] = columna
    elif accion == "mover":
        raise ValueError("Mover una nota necesita columna de destino")
    detalle = str(extra.get("detalle") or "").strip()
    if detalle:
        orden["detalle"] = detalle

    cola = ordenes(directorio_datos)
    cola.append(orden)
    _guardar_ordenes(directorio_datos, cola[-TOPE_ORDENES:])
    return orden


def recoger(directorio_datos: Path) -> list[dict[str, Any]]:
    """Entrega lo pendiente y vacía la cola. Lo llama la ventana.

    Se entrega y se borra en el mismo gesto, sin acuse de recibo. Es una
    decisión consciente: el acuse pediría un segundo viaje y un estado «en
    vuelo» que habría que caducar, y lo que se protege son dos o tres notas al
    día. Si la ventana se cierra en el medio segundo entre recoger y aplicar, la
    orden se pierde — y perder una nota que él acaba de dictar se nota en el
    acto y se vuelve a dictar, que es una salida que un tablero sí tiene.
    """
    cola = ordenes(directorio_datos)
    if cola:
        _guardar_ordenes(directorio_datos, [])
    return cola
