"""Lo que la biometría deja en disco: los perfiles y el registro de quién se aprendió.

Dos ficheros en `<datos>/`, los dos fuera de git como todo ese directorio:

  · `perfiles.json`: los números de cada persona con nombre. Escritura atómica
    (tmp + replace): un corte de luz a mitad no deja un JSON a medias.
  · `personas.log`: **cada vez que Perseo aprende algo de alguien, una línea**.
    Quién, cuándo, qué se guardó y por qué camino. Lo pidió el señor Persus el
    2026-09-23 junto con la regla de que nada se aprende sin que alguien diga
    el nombre: si el ordenador guarda voces y caras, que se pueda leer cuándo
    y de quién sin abrir un JSON de vectores. Ahí no van números, solo hechos.

Formato de un perfil: `{"voz": [...]|null, "voces": [[...]], "caras": [[...]],
"muestras": int, "creado": iso}`. `voces` es la galería de voz; `voz` es su
centro, que se sigue escribiendo porque es lo único que tenían los perfiles de
antes y lo que lee el resumen.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from . import biometria_galeria as galeria

logger = logging.getLogger(__name__)

NOMBRE_FICHERO = "perfiles.json"
NOMBRE_REGISTRO = "personas.log"


def ruta_perfiles(directorio_datos: Path) -> Path:
    return Path(directorio_datos) / NOMBRE_FICHERO


def cargar(directorio_datos: Path) -> dict[str, dict[str, Any]]:
    ruta = ruta_perfiles(directorio_datos)
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError):
        # Un fichero roto no tumba el reconocimiento: se empieza de cero y la
        # siguiente escritura atómica lo repone. Perder unos perfiles es malo;
        # que el núcleo no arranque por ello, peor.
        return {}
    if not isinstance(datos, dict):
        return {}
    return datos


def guardar(directorio_datos: Path, perfiles: dict[str, dict[str, Any]]) -> None:
    ruta = ruta_perfiles(directorio_datos)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(".tmp")
    temporal.write_text(json.dumps(perfiles, ensure_ascii=False, indent=1), encoding="utf-8")
    temporal.replace(ruta)


def voz_antigua(perfil: dict[str, Any]) -> bool:
    """Un perfil de voz de antes del 2026-09-23: un solo vector, y hecho mal.

    Se construía con un «refuerzo» que arrastraba medio perfil hacia el último
    trozo oído, así que puede llevar dentro la voz de quien pasara por delante.
    Se sigue usando para reconocer —es lo que hay— pero la primera muestra
    nueva lo sustituye en vez de sumarse a él.
    """
    return bool(perfil.get("voz")) and not perfil.get("voces")


def resumen(perfiles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Lo que enseña la pantalla: nombres y tamaños, nunca los vectores."""
    salida = []
    for nombre, perfil in sorted(perfiles.items()):
        salida.append(
            {
                "nombre": nombre,
                "voz": bool(perfil.get("voz")),
                "voces": len(galeria.voces(perfil)),
                "voz_antigua": voz_antigua(perfil),
                "caras": len(perfil.get("caras") or []),
                "muestras": perfil.get("muestras", 0),
                "creado": perfil.get("creado"),
            }
        )
    return salida


def ahora_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def perfil_de(perfiles: dict[str, dict[str, Any]], nombre: str) -> dict[str, Any]:
    """El perfil con ese nombre, creándolo vacío si no existe. Nunca pisa uno."""
    perfil = perfiles.get(nombre)
    if perfil is None:
        perfil = {"voz": None, "voces": [], "caras": [], "muestras": 0, "creado": ahora_iso()}
        perfiles[nombre] = perfil
    return perfil


def sumar_voces(perfil: dict[str, Any], vectores: list[list[float]], maximo: int) -> int:
    """Mete en la galería de voz lo que aporte. Devuelve cuántos entraron.

    Se parte de `voces` y no de `galeria.voces()`: el vector de un perfil
    antiguo no se hereda, se sustituye (ver `voz_antigua`).
    """
    lista = list(perfil.get("voces") or [])
    entraron = sum(1 for v in vectores if galeria.incorporar(lista, v, maximo))
    if entraron:
        galeria.poner_voces(perfil, lista)
        perfil["muestras"] = perfil.get("muestras", 0) + entraron
    return entraron


def sumar_caras(perfil: dict[str, Any], vectores: list[list[float]], maximo: int) -> int:
    caras = perfil.setdefault("caras", [])
    return sum(1 for v in vectores if galeria.incorporar(caras, v, maximo))


# --------------------------------------------------------------------------- #
# El registro
# --------------------------------------------------------------------------- #


def anotar(directorio_datos: Path, evento: str, nombre: str, detalle: str = "") -> None:
    """Una línea en `personas.log`. Que no se pueda escribir no tumba nada."""
    linea = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {evento:<10} {nombre}"
    if detalle:
        linea += f" — {detalle}"
    try:
        ruta = Path(directorio_datos) / NOMBRE_REGISTRO
        ruta.parent.mkdir(parents=True, exist_ok=True)
        with ruta.open("a", encoding="utf-8") as fichero:
            fichero.write(linea + "\n")
    except OSError as e:
        logger.warning("No se pudo escribir en %s: %s", NOMBRE_REGISTRO, e)
    logger.info("Personas: %s", linea)


def ultimas(directorio_datos: Path, cuantas: int = 20) -> list[str]:
    """Las últimas líneas del registro, las más recientes al final."""
    try:
        lineas = (Path(directorio_datos) / NOMBRE_REGISTRO).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return lineas[-cuantas:]
