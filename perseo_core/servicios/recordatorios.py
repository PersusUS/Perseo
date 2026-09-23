"""Recordatorios: «avísame en veinte minutos», «recuérdame mañana a las nueve».

Hasta el 2026-09-23 Perseo no sabía hacerlo. Tenía el tablero de tareas, que es
del señor Persus y no avisa de nada, y el calendario de Google, que sí avisa
pero es de solo lectura. Lo más básico que se le pide a un asistente —que te
recuerde algo a una hora— no tenía dónde vivir.

**Dónde viven.** En `<datos>/recordatorios.json`, junto al resto de lo que
Perseo escribe mientras trabaja. No en el vault, que es memoria; no en el
calendario, que no es suyo; y no en la cola, porque un recordatorio no es un
trabajo hasta que vence.

**Cómo avisan.** Un disparador mira cada treinta segundos si alguno ha vencido,
y si lo hay encola un trabajo `recordatorios.avisar`. Así el aviso pasa por el
mismo camino que el de la agenda: sale en la cola del panel y de la web, y
Telegram lleva el titular al móvil.

**Si el núcleo estaba apagado a la hora.** El aviso sale en cuanto vuelve,
diciendo cuánto llega tarde. Un recordatorio que se pierde en silencio es peor
que uno que llega tarde avisando de que llega tarde.

**Las horas son locales y con zona.** Se guardan en ISO con su desfase, para
que el cambio de hora de octubre no mueva un aviso de sitio sin decir nada.
"""

from __future__ import annotations

import json
import logging
import threading
import unicodedata
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

NOMBRE_FICHERO = "recordatorios.json"

#: Cómo se repite, si se repite.
REPETICIONES = ("nunca", "diario", "laborables", "semanal")

#: Un recordatorio para dentro de más de esto casi seguro es un error del
#: modelo al calcular la fecha, y apuntarlo es crear algo que nadie mirará.
MAXIMO_ADELANTO = timedelta(days=366)

#: Cuántos pendientes se leen de una vez: una lista de treinta no se dice en
#: voz alta.
TOPE_LISTA = 15

_bloqueo = threading.Lock()

_DIAS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")


def ahora_local() -> datetime:
    return datetime.now().astimezone()


def ruta(directorio_datos: Path | str) -> Path:
    return Path(directorio_datos) / NOMBRE_FICHERO


def _cargar(directorio_datos: Path | str) -> list[dict[str, Any]]:
    try:
        datos = json.loads(ruta(directorio_datos).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError) as e:
        # Un fichero roto no tumba los avisos futuros: se avisa y se sigue. Lo
        # que había se pierde, y por eso se dice alto en el registro.
        logger.error("No se pueden leer los recordatorios (%s); se empieza de cero.", e)
        return []
    return [r for r in datos if isinstance(r, dict)] if isinstance(datos, list) else []


def _guardar(directorio_datos: Path | str, recordatorios: list[dict[str, Any]]) -> None:
    destino = ruta(directorio_datos)
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_suffix(".tmp")
    temporal.write_text(json.dumps(recordatorios, ensure_ascii=False, indent=1), encoding="utf-8")
    temporal.replace(destino)


# --------------------------------------------------------------------------- #
# Fechas
# --------------------------------------------------------------------------- #


def calcular_cuando(
    ahora: datetime,
    en_minutos: Any = None,
    fecha_hora: Any = None,
    hora: Any = None,
    dias: Any = None,
    dia_semana: Any = None,
) -> datetime:
    """La hora del aviso, a partir de lo que diga el modelo.

    Tres formas, y **dos de ellas no necesitan saber qué día es hoy**, que es
    justo lo que el modelo no sabe —ningún prompt lleva la fecha—:

      · `en_minutos`: «en veinte minutos»;
      · `hora` y, si hace falta, `dias` o `dia_semana`: «mañana a las nueve» es
        hora 09:00 y dias 1; «el jueves a las cinco», hora 17:00 y jueves. Una
        hora sola es la próxima vez que llega: «a las nueve» dicho a las diez es
        mañana;
      · `fecha_hora` en ISO sin zona, hora local: para una fecha concreta.

    Los errores se dicen en palabras, porque llegan al modelo y es él quien
    tiene que corregirse.
    """
    if hora not in (None, ""):
        cuando = _por_hora(ahora, str(hora), dias, dia_semana)
    elif en_minutos not in (None, ""):
        try:
            minutos = float(en_minutos)
        except (TypeError, ValueError):
            raise ValueError(f"«{en_minutos}» no es un número de minutos") from None
        if minutos <= 0:
            raise ValueError("los minutos tienen que ser más de cero")
        cuando = ahora + timedelta(minutes=minutos)
    elif fecha_hora not in (None, ""):
        try:
            cuando = datetime.fromisoformat(str(fecha_hora).strip())
        except ValueError:
            raise ValueError(
                f"«{fecha_hora}» no es una fecha: hace falta AAAA-MM-DDTHH:MM, en hora local"
            ) from None
        cuando = cuando.astimezone() if cuando.tzinfo is None else cuando
        if cuando <= ahora:
            raise ValueError(f"las {cuando:%H:%M} del {cuando:%d/%m} ya han pasado")
    else:
        raise ValueError("falta cuándo: en_minutos, hora (con dias o dia_semana) o fecha_hora")

    if cuando - ahora > MAXIMO_ADELANTO:
        raise ValueError("eso es dentro de más de un año; revisa la fecha")
    return cuando.replace(microsecond=0)


def _por_hora(ahora: datetime, hora: str, dias: Any, dia_semana: Any) -> datetime:
    try:
        horas, minutos = (int(x) for x in hora.strip().replace(".", ":").split(":")[:2])
        base = ahora.replace(hour=horas, minute=minutos, second=0, microsecond=0)
    except (TypeError, ValueError):
        raise ValueError(f"«{hora}» no es una hora: hace falta HH:MM") from None

    if dia_semana not in (None, ""):
        nombre = _normalizar(str(dia_semana))
        semana = [_normalizar(d) for d in _DIAS]
        if nombre not in semana:
            raise ValueError(f"«{dia_semana}» no es un día de la semana")
        adelante = (semana.index(nombre) - ahora.weekday()) % 7
        # «El jueves» dicho un jueves es el de la semana que viene.
        return base + timedelta(days=adelante or 7)
    if dias not in (None, ""):
        try:
            cuantos = int(float(dias))
        except (TypeError, ValueError):
            raise ValueError(f"«{dias}» no es un número de días") from None
        if cuantos < 0:
            raise ValueError("los días tienen que ser cero o más")
        cuando = base + timedelta(days=cuantos)
        if cuando <= ahora:
            raise ValueError(f"hoy a las {base:%H:%M} ya ha pasado")
        return cuando
    # Una hora sola es la próxima vez que llega.
    return base if base > ahora else base + timedelta(days=1)


def siguiente(cuando: datetime, repetir: str, ahora: datetime) -> datetime | None:
    """La próxima vez que toca, estrictamente después de `ahora`. `None` si no se repite."""
    if repetir == "nunca":
        return None
    paso = timedelta(weeks=1) if repetir == "semanal" else timedelta(days=1)
    proxima = cuando
    while proxima <= ahora or (repetir == "laborables" and proxima.weekday() >= 5):
        proxima += paso
    return proxima


def describir(cuando: datetime, ahora: datetime) -> str:
    """«hoy a las 09:00», «mañana a las 09:00», «el jueves 25 a las 09:00»."""
    cuando = cuando.astimezone(ahora.tzinfo)
    dias = (cuando.date() - ahora.date()).days
    hora = f"a las {cuando:%H:%M}"
    if dias == 0:
        return f"hoy {hora}"
    if dias == 1:
        return f"mañana {hora}"
    if 1 < dias < 7:
        return f"el {_DIAS[cuando.weekday()]} {cuando.day} {hora}"
    return f"el {cuando.day}/{cuando.month}/{cuando.year} {hora}"


def _normalizar(texto: str) -> str:
    sin_tildes = unicodedata.normalize("NFD", texto.lower())
    return " ".join("".join(c for c in sin_tildes if unicodedata.category(c) != "Mn").split())


# --------------------------------------------------------------------------- #
# Lo que se hace con ellos
# --------------------------------------------------------------------------- #


def crear(
    directorio_datos: Path | str,
    texto: str,
    cuando: datetime,
    repetir: str = "nunca",
    quien: str | None = None,
) -> dict[str, Any]:
    texto = " ".join(str(texto or "").split())
    if not texto:
        raise ValueError("falta qué hay que recordar")
    repetir = (repetir or "nunca").strip().lower()
    if repetir not in REPETICIONES:
        raise ValueError(f"«{repetir}» no es una repetición: {', '.join(REPETICIONES)}")
    recordatorio = {
        "id": uuid.uuid4().hex[:8],
        "texto": texto[:300],
        "cuando": cuando.isoformat(),
        "repetir": repetir,
        "creado": ahora_local().isoformat(timespec="seconds"),
        "quien": quien,
    }
    with _bloqueo:
        lista = _cargar(directorio_datos)
        lista.append(recordatorio)
        _guardar(directorio_datos, lista)
    return recordatorio


def pendientes(directorio_datos: Path | str) -> list[dict[str, Any]]:
    with _bloqueo:
        lista = _cargar(directorio_datos)
    return sorted(lista, key=lambda r: str(r.get("cuando", "")))


def cancelar(directorio_datos: Path | str, clave: str) -> dict[str, Any]:
    """Quita un recordatorio por su id o por el principio de su texto.

    Si encajan dos, no se quita ninguno: tirar el equivocado es peor que
    preguntar cuál.
    """
    buscado = _normalizar(str(clave or ""))
    if not buscado:
        raise ValueError("falta cuál: su texto o su número")
    with _bloqueo:
        lista = _cargar(directorio_datos)
        encajan = [
            r
            for r in lista
            if r.get("id") == clave.strip() or _normalizar(str(r.get("texto", ""))).startswith(buscado)
        ]
        if not encajan:
            encajan = [r for r in lista if buscado in _normalizar(str(r.get("texto", "")))]
        if not encajan:
            raise ValueError(f"no hay ningún recordatorio que sea «{clave}»")
        if len(encajan) > 1:
            textos = "», «".join(str(r.get("texto")) for r in encajan[:4])
            raise ValueError(f"encajan {len(encajan)}: «{textos}». Di cuál")
        quitado = encajan[0]
        _guardar(directorio_datos, [r for r in lista if r is not quitado])
    return quitado


def vencidos(directorio_datos: Path | str, ahora: datetime) -> list[dict[str, Any]]:
    """Los que ya tocan. Los que no se repiten se quitan; los que sí, se mueven.

    Lo que se devuelve es la foto de cada uno **al vencer**, con cuánto llega
    tarde, para que el aviso pueda decirlo.
    """
    with _bloqueo:
        lista = _cargar(directorio_datos)
        tocan: list[dict[str, Any]] = []
        quedan: list[dict[str, Any]] = []
        for recordatorio in lista:
            try:
                cuando = datetime.fromisoformat(str(recordatorio.get("cuando")))
            except ValueError:
                logger.warning("Recordatorio con fecha ilegible, se quita: %r", recordatorio)
                continue
            if cuando > ahora:
                quedan.append(recordatorio)
                continue
            retraso = int((ahora - cuando).total_seconds() // 60)
            tocan.append({**recordatorio, "retraso_min": retraso})
            proxima = siguiente(cuando, str(recordatorio.get("repetir", "nunca")), ahora)
            if proxima is not None:
                quedan.append({**recordatorio, "cuando": proxima.isoformat()})
        if tocan:
            _guardar(directorio_datos, quedan)
    return tocan
