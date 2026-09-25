"""Lo que la biometría recuerda mientras dura una sesión, y nada más.

Tres cosas, todas en memoria y ninguna en disco:

  · los **racimos**: desconocidos a medio aprender, de voz y de cara;
  · el **turno**: los trozos de voz seguidos de una misma persona, con la nota
    que ha ido sacando cada perfil, para decidir con la media y no con el
    último trozo;
  · lo **que se ve**: la última lectura de caras, que ayuda a la voz cuando
    duda, y las **pistas** que siguen cada cara de un fotograma al siguiente.

Si el núcleo se reinicia, se pierde todo esto y no pasa nada: los perfiles
fijados viven en `perfiles.json` y los lleva `biometria.py`.

Todo se toca con `bloqueo` cogido. Las estructuras son mutables y no se
reasignan nunca —se vacían y se rellenan—, para que quien las importe vea
siempre las mismas.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from typing import Any

from . import biometria_galeria as galeria

#: Pausa que corta el turno. La app manda una ventana al acabar cada frase o
#: cada tres segundos hablando seguido; más de cuatro segundos sin trozos es
#: que el que hablaba se calló.
PAUSA_TURNO = 4.0

#: Cuánto vale la última lectura de caras para ayudar a la voz. La app manda un
#: fotograma cada cuatro segundos; con ocho se tolera uno perdido.
VIGENCIA_CARAS = 8.0

#: Seguimiento de caras entre fotogramas: solape mínimo para decir «es la misma
#: cara que antes», y cuánto se recuerda una que dejó de verse.
IOU_PISTA = 0.3
VIGENCIA_PISTA = 10.0

_PATRON_DESCONOCIDO = re.compile(r"^Desconocido (\d+)$")

bloqueo = threading.Lock()

clusters_voz: dict[str, dict[str, Any]] = {}
clusters_cara: dict[str, dict[str, Any]] = {}

#: `cuando`: último trozo; `puntos`: nombre -> (suma de notas por segundo,
#: segundos); `racimo`: el desconocido que lleva el turno, si lo hay.
turno: dict[str, Any] = {}
#: `cuando` y `nombres` de la última lectura de caras.
vistas: dict[str, Any] = {}
pistas: list[dict[str, Any]] = []

#: El número más alto repartido en esta sesión. No basta con él: tras reiniciar
#: vuelve a cero y ya hay «Desconocido 1» en disco. Hasta el 2026-09-23 el
#: siguiente desconocido que se fijara con ese número **sobrescribía** el
#: perfil guardado. Ver `nueva_etiqueta`.
_contador = {"n": 0}

#: Reloj de la sesión. Aparte para que las pruebas puedan moverlo.
reloj = time.monotonic


def cortar_turno() -> None:
    turno.clear()
    turno.update({"cuando": float("-inf"), "puntos": {}, "racimo": None})


def reiniciar() -> None:
    """Vacía todo. Quien llame tiene que tener `bloqueo`."""
    clusters_voz.clear()
    clusters_cara.clear()
    _contador["n"] = 0
    cortar_turno()
    vistas.clear()
    vistas.update({"cuando": float("-inf"), "nombres": []})
    pistas.clear()


reiniciar()


# --------------------------------------------------------------------------- #
# Racimos y etiquetas
# --------------------------------------------------------------------------- #


def es_desconocido(nombre: str | None) -> bool:
    return bool(nombre and _PATRON_DESCONOCIDO.match(nombre))


def nueva_etiqueta(perfiles: dict[str, Any]) -> str:
    """El siguiente «Desconocido N» libre: ni en disco ni a medio aprender."""
    usados = [
        int(m.group(1))
        for nombre in (*perfiles, *clusters_voz, *clusters_cara)
        if (m := _PATRON_DESCONOCIDO.match(nombre))
    ]
    _contador["n"] = max([_contador["n"], *usados]) + 1
    return f"Desconocido {_contador['n']}"


def progreso(clusters: dict[str, dict[str, Any]], objetivo: float) -> dict[str, Any] | None:
    """El racimo más avanzado, para que la pantalla enseñe cuánto falta."""
    if not clusters:
        return None
    etiqueta, cluster = max(clusters.items(), key=lambda par: par[1]["peso"])
    return {"etiqueta": etiqueta, "peso": round(cluster["peso"], 1), "objetivo": objetivo}


def racimo_parecido(
    clusters: dict[str, dict[str, Any]], vector: list[float], umbral: float
) -> str | None:
    mejor_etiqueta: str | None = None
    mejor_similitud = -1.0
    for etiqueta, cluster in clusters.items():
        similitud = galeria.puntuar_racimo(cluster, vector)
        if similitud > mejor_similitud:
            mejor_etiqueta, mejor_similitud = etiqueta, similitud
    return mejor_etiqueta if mejor_similitud >= umbral else None


# --------------------------------------------------------------------------- #
# El turno de voz
# --------------------------------------------------------------------------- #


def lideres(perfiles: dict[str, Any]) -> list[tuple[str, float]]:
    """Los perfiles del turno con su media, el mejor primero."""
    medias = [
        (nombre, suma / peso)
        for nombre, (suma, peso) in turno["puntos"].items()
        if nombre in perfiles and peso > 0
    ]
    return sorted(medias, key=lambda par: par[1], reverse=True)


def lider(perfiles: dict[str, Any]) -> tuple[str | None, float]:
    """El perfil con mejor media en el turno, y esa media."""
    primeros = lideres(perfiles)
    return primeros[0] if primeros else (None, -1.0)


def seguir_o_cortar(
    perfiles: dict[str, Any],
    notas: dict[str, float],
    vector: list[float],
    ahora: float,
    umbral_cambio: float,
    umbral_otro: float,
) -> None:
    """Decide si este trozo sigue el turno anterior o empieza uno nuevo.

    Se corta con una pausa larga, cuando el trozo no se parece a quien llevaba
    el turno, o cuando él solo es claramente de otro conocido: la media del
    turno anterior no debe tapar a quien acaba de entrar.
    """
    if ahora - turno["cuando"] > PAUSA_TURNO:
        cortar_turno()
        return

    racimo = clusters_voz.get(turno["racimo"] or "")
    quien, _ = lider(perfiles)
    if racimo is not None:
        cambia = galeria.puntuar_racimo(racimo, vector) < umbral_cambio
    elif quien is not None:
        cambia = notas.get(quien, -1.0) < umbral_cambio
    else:
        cambia = False

    if not cambia and notas and quien is not None:
        otro = max(notas, key=notas.__getitem__)
        cambia = otro != quien and notas[otro] >= umbral_otro

    if cambia:
        cortar_turno()


def sumar_al_turno(notas: dict[str, float], segundos: float, ahora: float) -> None:
    for nombre, nota in notas.items():
        suma, peso = turno["puntos"].get(nombre, (0.0, 0.0))
        turno["puntos"][nombre] = (suma + nota * segundos, peso + segundos)
    turno["cuando"] = ahora


# --------------------------------------------------------------------------- #
# Lo que se ve
# --------------------------------------------------------------------------- #


def anotar_vistas(ahora: float, nombres: list[str | None]) -> None:
    vistas.update({"cuando": ahora, "nombres": nombres})


def cara_unica(ahora: float) -> str | None:
    """El nombre de la única cara a la vista, si hay exactamente una y es reciente."""
    if ahora - vistas["cuando"] > VIGENCIA_CARAS:
        return None
    nombres = vistas["nombres"]
    return nombres[0] if len(nombres) == 1 else None


def turno_reciente(ahora: float) -> bool:
    return ahora - turno["cuando"] <= VIGENCIA_CARAS


def _solape(a: list[int], b: list[int]) -> float:
    """Intersección sobre unión de dos cajas [x, y, ancho, alto]."""
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    ancho = max(0, min(ax2, bx2) - max(a[0], b[0]))
    alto = max(0, min(ay2, by2) - max(a[1], b[1]))
    comun = ancho * alto
    union = a[2] * a[3] + b[2] * b[3] - comun
    return comun / union if union > 0 else 0.0


def pista_para(caja: list[int], usadas: set[int], ahora: float) -> dict[str, Any]:
    """La pista de la cara que estaba en ese sitio hace un momento, o una nueva.

    Seguir la cara por su posición es lo que permite no fiarse de un solo
    fotograma: si alguien conocido gira la cabeza y SFace duda, la pista sabe
    que ahí estaba él y no abre un «Desconocido» con su cara de perfil.

    Las pistas que se caen por viejas se quitan antes de mirar, así que los
    índices de `usadas` solo valen dentro de un mismo fotograma.
    """
    if not usadas:
        pistas[:] = [p for p in pistas if ahora - p["visto"] <= VIGENCIA_PISTA]
    candidatas = [(i, _solape(caja, p["caja"])) for i, p in enumerate(pistas) if i not in usadas]
    i, solape = max(candidatas, key=lambda par: par[1], default=(-1, 0.0))
    if solape < IOU_PISTA:
        pistas.append({"caja": caja, "historial": deque(maxlen=3), "visto": ahora})
        i = len(pistas) - 1
    usadas.add(i)
    pistas[i]["caja"] = caja
    pistas[i]["visto"] = ahora
    return pistas[i]


def nombre_sostenido(pista: dict[str, Any]) -> str | None:
    """El nombre que la pista ha tenido al menos dos de las tres últimas veces."""
    cuenta: dict[str, int] = {}
    for nombre in pista["historial"]:
        if nombre:
            cuenta[nombre] = cuenta.get(nombre, 0) + 1
    mejor = max(cuenta, key=cuenta.__getitem__, default=None)
    return mejor if mejor is not None and cuenta[mejor] >= 2 else None
