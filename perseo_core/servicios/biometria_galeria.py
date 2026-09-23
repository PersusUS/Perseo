"""Los vectores de una persona, guardados como galería y no como uno solo.

**Por qué galería.** Hasta el 2026-09-23 cada perfil de voz era UN vector, y se
«reforzaba» con `_media(a, b)`: el punto medio entre lo guardado y lo de hoy.
Eso no es una media, es una media móvil de peso 0,5 —cada acierto arrastraba
el perfil la mitad del camino hacia el último trozo—, así que el perfil
acababa siendo las dos o tres últimas muestras y no la persona. Un trozo malo
que pasara el umbral (una tele de fondo, otra voz parecida) se quedaba con la
mitad del perfil de golpe.

Ahora cada persona guarda hasta `MAX_VECTORES` vectores distintos entre sí, y
se la puntúa con la media de los tres que mejor casan. Una muestra nueva solo
entra si **aporta algo**: si se parece mucho a una que ya está, sobra; si la
galería está llena, sustituye a la más redundante, y solo si ella es más
distinta que esa. Así la galería gana ángulos y registros de voz sin olvidar
los buenos y sin dejarse arrastrar por el último.

Las caras ya se guardaban como lista (cinco ángulos, los cinco primeros que
llegaran). Usan la misma galería, que además sabe qué hacer cuando se llena.

Sin numpy, como el resto de la biometría.
"""

from __future__ import annotations

import math
from typing import Any

#: Vectores por persona y canal. Ocho ángulos o registros de voz ya cubren lo
#: que da una cámara de escritorio o un micrófono de portátil; más solo
#: engordaría `perfiles.json` sin mejorar el acierto.
MAX_VECTORES = 8

#: Cuántos de los mejores se promedian para puntuar. Uno solo haría que un
#: vector raro de la galería decidiera; todos, que los ángulos que no tocan
#: hoy hundieran la nota del que sí.
MEJORES = 3

#: A partir de este parecido, una muestra nueva no aporta nada.
REDUNDANTE = 0.95


def coseno(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return -1.0
    punto = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return -1.0
    return punto / (na * nb)


def normalizar(vector: list[float]) -> list[float]:
    norma = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norma for x in vector]


def centroide(galeria: list[list[float]]) -> list[float] | None:
    """La media de verdad, con todos pesando lo mismo, y normalizada."""
    if not galeria:
        return None
    dimension = len(galeria[0])
    suma = [0.0] * dimension
    for vector in galeria:
        for i, x in enumerate(normalizar(vector)):
            suma[i] += x
    return normalizar(suma)


def puntuar(galeria: list[list[float]], vector: list[float]) -> float:
    """Media de los `MEJORES` cosenos más altos. -1 si no hay con qué comparar."""
    parecidos = sorted((coseno(vector, g) for g in galeria), reverse=True)
    if not parecidos:
        return -1.0
    mejores = parecidos[:MEJORES]
    return sum(mejores) / len(mejores)


def incorporar(galeria: list[list[float]], vector: list[float], maximo: int = MAX_VECTORES) -> bool:
    """Añade el vector si aporta variedad. Devuelve si la galería cambió."""
    if not galeria:
        galeria.append(vector)
        return True
    parecido = max(coseno(vector, g) for g in galeria)
    if parecido >= REDUNDANTE:
        return False
    if len(galeria) < maximo:
        galeria.append(vector)
        return True

    # Llena: la más redundante es la que más se parece, de media, a las demás.
    # Solo se va si la nueva es más distinta de la galería que ella.
    def redundancia(i: int) -> float:
        otros = [coseno(galeria[i], g) for j, g in enumerate(galeria) if j != i]
        return sum(otros) / len(otros)

    peor = max(range(len(galeria)), key=redundancia)
    if parecido < redundancia(peor):
        galeria[peor] = vector
        return True
    return False


# --------------------------------------------------------------------------- #
# Racimos: los desconocidos a medio aprender
# --------------------------------------------------------------------------- #


def racimo_nuevo(vector: list[float], peso: float) -> dict[str, Any]:
    racimo: dict[str, Any] = {"suma": [0.0] * len(vector), "peso": 0.0, "vectores": []}
    acumular(racimo, vector, peso)
    return racimo


def acumular(racimo: dict[str, Any], vector: list[float], peso: float) -> None:
    """Suma ponderada de verdad: un trozo de tres segundos pesa más que uno de uno.

    `vector` es el centro del racimo, normalizado; `vectores` es la galería con
    la que nacerá el perfil cuando el racimo se fije.
    """
    unidad = normalizar(vector)
    racimo["suma"] = [s + x * peso for s, x in zip(racimo["suma"], unidad)]
    racimo["vector"] = normalizar(racimo["suma"])
    racimo["peso"] += peso
    incorporar(racimo["vectores"], vector)


def puntuar_racimo(racimo: dict[str, Any], vector: list[float]) -> float:
    """Contra el centro o contra la galería, lo que case mejor."""
    return max(coseno(vector, racimo["vector"]), puntuar(racimo["vectores"], vector))


# --------------------------------------------------------------------------- #
# El perfil en disco
# --------------------------------------------------------------------------- #


def voces(perfil: dict[str, Any]) -> list[list[float]]:
    """La galería de voz. Los perfiles de antes solo tenían `voz`: esa es la primera."""
    galeria = perfil.get("voces")
    if isinstance(galeria, list) and galeria:
        return galeria
    if perfil.get("voz"):
        return [perfil["voz"]]
    return []


def poner_voces(perfil: dict[str, Any], galeria: list[list[float]]) -> None:
    """Guarda la galería y deja en `voz` su centro, que es lo que leía todo lo de antes."""
    perfil["voces"] = galeria
    perfil["voz"] = centroide(galeria)
