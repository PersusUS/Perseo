"""Vigilancias: «avísame cuando haya entradas», «resérvalo si baja de 80 €».

Lo que Instinct hace con las entradas agotadas y las mesas sin hueco: mirar
cada tanto, sin que nadie se lo vuelva a pedir, y avisar —o hacerlo— cuando
cambia. Aquí una vigilancia es un recado con una condición, que un disparador
relanza cada pocas horas hasta que la condición se cumple o caduca.

**Dónde viven.** En `<datos>/vigilancias.json`, como los recordatorios: no son
trabajos hasta que toca comprobarlas, y un reinicio no las puede perder.

**Por qué hay topes, y por qué son estos.** Cada comprobación es un recado, y
un recado son varias peticiones a Gemini —abrir la página, leerla, decir si se
cumple—: tres o cuatro de media. El chat vive del mismo cubo, 500 al día en
Flash Lite. Cinco vigilancias cada media hora serían 240 comprobaciones al día
y se comerían la cuota entera antes de comer. Por eso: cada una se mira como
mucho una vez por hora, no hay más de cinco a la vez, y entre todas no pasan de
24 comprobaciones al día —unas cien peticiones, la quinta parte del cubo—.
Lo que no cabe en el día espera al siguiente, no se pierde.

**Qué pasa al cumplirse.** O se avisa (la llamada de siempre, y el titular por
Telegram), o se hace el recado entero. En ese caso lo que sale de casa se para
igual que en cualquier recado: la vigilancia encuentra la mesa, pero pagarla
espera su sí (ADR 0007).
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

NOMBRE_FICHERO = "vigilancias.json"

#: Qué se hace cuando la condición se cumple.
AL_CUMPLIRSE = ("avisar", "hacer")

HORAS_MINIMO = 1
HORAS_POR_DEFECTO = 3
DIAS_POR_DEFECTO = 7
DIAS_MAXIMO = 30
MAXIMO_ACTIVAS = 5

#: Comprobaciones al día entre todas. Ver la cabecera.
TOPE_DIARIO = 24

_bloqueo = threading.Lock()


def ruta(directorio_datos: Path | str) -> Path:
    return Path(directorio_datos) / NOMBRE_FICHERO


def _cargar(directorio_datos: Path | str) -> dict[str, Any]:
    try:
        datos = json.loads(ruta(directorio_datos).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"vigilancias": [], "cuenta": {}}
    except (OSError, json.JSONDecodeError) as e:
        logger.error("No se pueden leer las vigilancias (%s); se empieza de cero.", e)
        return {"vigilancias": [], "cuenta": {}}
    if not isinstance(datos, dict):
        return {"vigilancias": [], "cuenta": {}}
    datos.setdefault("vigilancias", [])
    datos.setdefault("cuenta", {})
    return datos


def _guardar(directorio_datos: Path | str, datos: dict[str, Any]) -> None:
    destino = ruta(directorio_datos)
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_suffix(".tmp")
    temporal.write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding="utf-8")
    temporal.replace(destino)


def _normalizar(texto: str) -> str:
    sin_tildes = unicodedata.normalize("NFD", texto.lower())
    return " ".join("".join(c for c in sin_tildes if unicodedata.category(c) != "Mn").split())


def _limpio(texto: Any, tope: int) -> str:
    return " ".join(str(texto or "").split())[:tope]


# --------------------------------------------------------------------------- #
# Lo que pide él
# --------------------------------------------------------------------------- #


def crear(
    directorio_datos: Path | str,
    objetivo: str,
    condicion: str,
    ahora: datetime,
    cada_horas: Any = None,
    dias: Any = None,
    al_cumplirse: str = "avisar",
    quien: str | None = None,
) -> dict[str, Any]:
    objetivo = _limpio(objetivo, 400)
    condicion = _limpio(condicion, 300)
    if not objetivo:
        raise ValueError("falta qué hay que mirar: la web o lo que se busca")
    if not condicion:
        raise ValueError("falta cuándo avisar: qué tiene que pasar")
    al_cumplirse = (al_cumplirse or "avisar").strip().lower()
    if al_cumplirse not in AL_CUMPLIRSE:
        raise ValueError(f"«{al_cumplirse}» no vale: {', '.join(AL_CUMPLIRSE)}")
    try:
        horas = max(HORAS_MINIMO, float(cada_horas)) if cada_horas not in (None, "") else HORAS_POR_DEFECTO
        plazo = min(DIAS_MAXIMO, max(1, int(float(dias)))) if dias not in (None, "") else DIAS_POR_DEFECTO
    except (TypeError, ValueError):
        raise ValueError("cada_horas y dias tienen que ser números") from None

    vigilancia = {
        "id": uuid.uuid4().hex[:8],
        "objetivo": objetivo,
        "condicion": condicion,
        "al_cumplirse": al_cumplirse,
        "cada_horas": horas,
        "hasta": (ahora + timedelta(days=plazo)).isoformat(timespec="seconds"),
        # La primera, enseguida: quien pide vigilar algo quiere saber si ya
        # se cumple, no enterarse dentro de tres horas.
        "proxima": ahora.isoformat(timespec="seconds"),
        "estado": "activa",
        "comprobaciones": 0,
        "ultima": None,
        "creada": ahora.isoformat(timespec="seconds"),
        "quien": quien,
    }
    with _bloqueo:
        datos = _cargar(directorio_datos)
        activas = [v for v in datos["vigilancias"] if v.get("estado") == "activa"]
        if len(activas) >= MAXIMO_ACTIVAS:
            raise ValueError(
                f"ya hay {MAXIMO_ACTIVAS} vigilancias activas, que es el tope: quita alguna antes"
            )
        datos["vigilancias"].append(vigilancia)
        _guardar(directorio_datos, datos)
    return vigilancia


def activas(directorio_datos: Path | str) -> list[dict[str, Any]]:
    with _bloqueo:
        datos = _cargar(directorio_datos)
    return [v for v in datos["vigilancias"] if v.get("estado") == "activa"]


def obtener(directorio_datos: Path | str, id_vigilancia: str) -> dict[str, Any] | None:
    with _bloqueo:
        datos = _cargar(directorio_datos)
    return next((v for v in datos["vigilancias"] if v.get("id") == id_vigilancia), None)


def cancelar(directorio_datos: Path | str, clave: str) -> dict[str, Any]:
    """Quita una activa por su id o por el principio de lo que vigila.

    Si encajan dos no se quita ninguna, como con los recordatorios.
    """
    buscado = _normalizar(str(clave or ""))
    if not buscado:
        raise ValueError("falta cuál: lo que vigila o su número")
    with _bloqueo:
        datos = _cargar(directorio_datos)
        vivas = [v for v in datos["vigilancias"] if v.get("estado") == "activa"]
        encajan = [
            v for v in vivas
            if v.get("id") == clave.strip() or _normalizar(str(v.get("objetivo", ""))).startswith(buscado)
        ] or [v for v in vivas if buscado in _normalizar(str(v.get("objetivo", "")))]
        if not encajan:
            raise ValueError(f"no hay ninguna vigilancia activa que sea «{clave}»")
        if len(encajan) > 1:
            textos = "», «".join(str(v.get("objetivo")) for v in encajan[:4])
            raise ValueError(f"encajan {len(encajan)}: «{textos}». Di cuál")
        encajan[0]["estado"] = "cancelada"
        _guardar(directorio_datos, datos)
    return encajan[0]


# --------------------------------------------------------------------------- #
# Lo que hace el disparador
# --------------------------------------------------------------------------- #


def para_comprobar(
    directorio_datos: Path | str, ahora: datetime
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Las que toca mirar ya, y las que acaban de caducar sin cumplirse.

    A las que tocan se les adelanta la próxima **aquí**, al encolarlas, y no al
    acabar: si la comprobación tarda o falla, la vuelta siguiente del disparador
    no debe encolar otra encima.
    """
    dia = ahora.date().isoformat()
    with _bloqueo:
        datos = _cargar(directorio_datos)
        cuenta = datos["cuenta"] if datos["cuenta"].get("dia") == dia else {"dia": dia, "n": 0}
        tocan: list[dict[str, Any]] = []
        caducadas: list[dict[str, Any]] = []
        for v in datos["vigilancias"]:
            if v.get("estado") != "activa":
                continue
            try:
                hasta = datetime.fromisoformat(str(v["hasta"]))
                proxima = datetime.fromisoformat(str(v["proxima"]))
            except (KeyError, ValueError):
                logger.warning("Vigilancia con fechas ilegibles, se cierra: %r", v)
                v["estado"] = "caducada"
                continue
            if hasta <= ahora:
                v["estado"] = "caducada"
                caducadas.append(dict(v))
                continue
            if proxima > ahora or cuenta["n"] >= TOPE_DIARIO:
                continue
            cuenta["n"] += 1
            v["comprobaciones"] = int(v.get("comprobaciones") or 0) + 1
            v["proxima"] = (ahora + timedelta(hours=float(v.get("cada_horas") or HORAS_POR_DEFECTO))).isoformat(
                timespec="seconds"
            )
            tocan.append(dict(v))
        datos["cuenta"] = cuenta
        if tocan or caducadas:
            _guardar(directorio_datos, datos)
    return tocan, caducadas


def apuntar(
    directorio_datos: Path | str, id_vigilancia: str, cumple: bool, detalle: str, ahora: datetime
) -> dict[str, Any] | None:
    """Lo que dijo la última comprobación. Si se cumple, la vigilancia se cierra."""
    with _bloqueo:
        datos = _cargar(directorio_datos)
        v = next((x for x in datos["vigilancias"] if x.get("id") == id_vigilancia), None)
        if v is None:
            return None
        v["ultima"] = {"cuando": ahora.isoformat(timespec="seconds"), "cumple": cumple, "detalle": _limpio(detalle, 300)}
        if cumple and v.get("estado") == "activa":
            v["estado"] = "cumplida"
        _guardar(directorio_datos, datos)
    return v


def enunciado(v: dict[str, Any]) -> str:
    """El encargo que recibe el recado que la comprueba."""
    if v.get("al_cumplirse") == "hacer":
        despues = (
            "Si SÍ se cumple, llama a informar con cumple=true y luego haz el recado entero; "
            "acaba con terminar."
        )
    else:
        despues = "Si SÍ se cumple, llama a informar con cumple=true y lo que viste; no hagas nada más."
    return (
        f"Esto es una vigilancia, no un recado normal. Lo que se vigila: {v.get('objetivo')}\n"
        f"La condición: {v.get('condicion')}\n"
        "Mira si la condición se cumple AHORA, con los pasos justos. "
        "Si NO se cumple, llama a informar con cumple=false y un detalle corto de lo que viste, "
        f"y nada más. {despues}"
    )
