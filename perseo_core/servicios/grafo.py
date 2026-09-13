"""El grafo del segundo cerebro, para mirarlo fuera de Obsidian.

El señor Persus pidió una ventana propia con el grafo del vault — no Obsidian
abierto, sino el cerebro como constelación, actualizándose mientras se habla.
Este módulo es la mitad del núcleo: leer el vault, tejer el grafo de enlaces
y servirlo a quien lo pinte.

CÓMO SE TEJE
------------
Cada `.md` del vault es un nodo; cada `[[enlace]]` dentro de él, una arista
hacia la nota destino. Las reglas, que son las de Obsidian con una excepción:

  * El título de una nota es el nombre de su fichero, no su primer encabezado.
  * Los enlaces llevan alias (`[[nota|como se ve]]`), anclas (`[[nota#punto]]`)
    y a veces carpeta delante (`[[carpeta/nota]]`); lo que cuenta es la nota,
    así que se recorta hasta su nombre.
  * Se comparan sin mayúsculas ni minúsculas de diferencia (`casefold`), como
    hace Obsidian al resolver. Las tildes sí cuentan: `Perseo` no enlaza con
    `perseo` si el fichero se llama con tilde.
  * **La excepción**: un enlace a una nota que no existe no crea nodo fantasma.
    El grafo de Obsidian los enseña apagados; aquí de momento no — son ruido
    de plantillas y de notas por escribir, y lo que se pidió fue mirar lo que
    hay. Se cuentan aparte, por si acaso.

LA CACHÉ
--------
El cliente pregunta cada pocos segundos («tiempo real» para un segundo
cerebro son segundos, no milisegundos), y releer el vault entero en cada petición sería
gastar disco por gusto: hay una caché de segundos que solo se rompe cuando
caduca o cambia la raíz. Cuando Perseo escribe una nota en mitad de una
conversación, el grafo la enseña en la siguiente vuelta de la caché.

ABRIR UNA NOTA (ENMIENDA DEL 2026-08-24, PEDIDA POR EL SEÑOR PERSUS)
--------------------------------------------------------------------
Pulsar un nodo del grafo tiene que abrir su nota en Obsidian. Por HTTP viaja
**cuál** de las notas del vault — el mismo trato que `proyectos`: nunca una
ruta libre ni una orden. El id se busca entre los ficheros reales y lo que se
abre es el URI `obsidian://open` de la coincidencia, resuelto aquí y no en el
cliente: quien mira el grafo puede estar en el móvil, y Obsidian vive en esta
máquina.

LOS HUÉRFANOS, APARTADOS (ENMIENDA DE ESE MISMO DÍA)
----------------------------------------------------
Al principio TODAS las notas eran nodos, también las que no enlazan con nadie:
un cerebro joven tiene más estrellas sueltas que constelaciones, y enseñarlo
vacío habría sido mentirlo. Viéndolo lleno se vio la otra cara — puntos sin
historia que ensucian la constelación— y se pidió lo contrario: ahora solo se
dibuja lo que está conectado, y las que quedan fuera se cuentan en
`apartadas`. Que desaparecieran callando sería mentir de otro modo.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Carpetas que no son cerebro: las interioridades de Obsidian y la papelera.
EXCLUIDAS = {".obsidian", ".trash", ".git"}

#: Techo de nodos. Por encima, la simulación del cliente se arrastra y el
#: dibujo deja de decir nada: mejor una constelación legible que una sopa.
TOPE_NODOS = 500

#: Los `[[...]]` de una nota. El contenido va hasta el primer `]]`; el alias y
#: el ancla se recortan después, que es donde viven `|` y `#`.
PATRON_ENLACE = re.compile(r"\[\[([^\[\]]+)\]\]")

#: Segundos que vale una lectura del vault antes de volver a leerlo.
TTL_SEGUNDOS = 4.0

_CERROJO = threading.Lock()
_ULTIMA: tuple[float, str, dict[str, Any]] | None = None


def _enlaces_de(texto: str) -> list[str]:
    """Los destinos de una nota, ya sin alias, ancla ni carpeta, en minúsculas.

    Un enlace puede venir con la carpeta delante — `[[proyectos/idea]]` —;
    lo que nombra la nota es lo último, igual que Obsidian resuelve por el
    nombre del fichero.
    """
    destinos: list[str] = []
    for crudo in PATRON_ENLACE.findall(texto):
        destino = crudo.split("|", 1)[0].split("#", 1)[0]
        destino = destino.split("/")[-1].strip().casefold()
        if destino:
            destinos.append(destino)
    return destinos


def construir(ruta_vault: Path) -> dict[str, Any]:
    """El grafo entero del vault. Ver la cabecera del módulo para las reglas."""
    global _ULTIMA

    clave_raiz = str(ruta_vault)
    with _CERROJO:
        if _ULTIMA is not None:
            instantaneo, raiz, datos = _ULTIMA
            if raiz == clave_raiz and (time.monotonic() - instantaneo) < TTL_SEGUNDOS:
                return datos

    datos = _tejer(ruta_vault)
    with _CERROJO:
        _ULTIMA = (time.monotonic(), clave_raiz, datos)
    return datos


def invalidar() -> None:
    """Tira la caché: la próxima petición relee el vault."""
    global _ULTIMA
    with _CERROJO:
        _ULTIMA = None


def _tejer(ruta_vault: Path) -> dict[str, Any]:
    """La lectura de verdad del vault. Sin caché, sin cerrojo: llámala con cabeza."""
    vacio: dict[str, Any] = {
        "nodos": [],
        "enlaces": [],
        "vault": ruta_vault.name,
        "total_notas": 0,
        "apartadas": 0,
        "fantasmas": 0,
        "generado": int(time.time()),
    }
    if not ruta_vault.is_dir():
        logger.warning("El vault %s no existe; grafo vacío.", ruta_vault)
        return vacio

    titulos: dict[str, str] = {}
    rutas: dict[str, str] = {}
    aristas: set[tuple[str, str]] = set()
    total_notas = 0

    for ruta in sorted(ruta_vault.rglob("*.md")):
        relativa = ruta.relative_to(ruta_vault)
        if any(parte.lower() in EXCLUIDAS for parte in relativa.parts[:-1]):
            continue
        total_notas += 1
        clave = ruta.stem.casefold()
        titulos.setdefault(clave, ruta.stem)
        # La ruta que Obsidian necesita para abrir la nota. Si el mismo título
        # vive en dos carpetas, gana la primera — y `abrir_nota` afina después.
        rutas.setdefault(clave, relativa.with_suffix("").as_posix())
        try:
            texto = ruta.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            logger.warning("No se pudo leer %s: %s", ruta, error)
            continue
        for destino in _enlaces_de(texto):
            if destino == clave:
                continue  # una nota que se enlaza a sí misma no es una arista
            aristas.add(tuple(sorted((clave, destino))))  # type: ignore[arg-type]

    # El grado sale de las aristas ÚNICAS: tres enlaces a la misma nota son una
    # conexión, no tres — es lo que hace el grafo de Obsidian al medir.
    grado: dict[str, int] = {}
    for a, b in aristas:
        grado[a] = grado.get(a, 0) + 1
        grado[b] = grado.get(b, 0) + 1

    # Los destinos sin fichero siguen siendo fantasmas: se cuentan, no se dibujan.
    extremos = {c for par in aristas for c in par}
    fantasmas = len(extremos - set(titulos))

    # Los huérfanos no se dibujan (ver la enmienda de la cabecera): queda
    # fuera toda nota sin ningún enlace FIRME — ni recibido ni dado, y hacia
    # otra nota que exista; un enlace a un fantasma no sostiene a nadie,
    # porque el fantasma tampoco se pinta y la nota quedaría flotando sola
    # con una arista rota. Se cuentan: el cifras de arriba las dice.
    firmes = {par for par in aristas if par[0] in titulos and par[1] in titulos}
    conectadas = {c for par in firmes for c in par}
    apartadas = len(set(titulos) - conectadas)

    candidatas = set(titulos) & conectadas
    if len(candidatas) > TOPE_NODOS:
        candidatas = set(
            sorted(candidatas, key=lambda c: (-grado.get(c, 0), c))[:TOPE_NODOS]
        )

    nodos = [
        {
            "id": clave,
            "titulo": titulos[clave],
            "grado": grado.get(clave, 0),
            "ruta": rutas[clave],
        }
        for clave in sorted(candidatas)
    ]
    enlaces = [
        {"a": a, "b": b}
        for a, b in sorted(aristas)
        if a in candidatas and b in candidatas
    ]

    return {
        "nodos": nodos,
        "enlaces": enlaces,
        # El nombre del vault es el que Obsidian pone en sus URI `obsidian://`:
        # la carpeta que lo contiene, tal y como él la conoce.
        "vault": ruta_vault.name,
        "total_notas": total_notas,
        "apartadas": apartadas,
        "fantasmas": fantasmas,
        "generado": int(time.time()),
    }


def abrir_nota(ruta_vault: Path, id_nota: str) -> str:
    """Abre en Obsidian la nota de un nodo del grafo. 'Éxito: …' o 'Error: …'.

    El mismo contrato de texto que `proyectos.abrir`: lo que sale de aquí
    acaba en una pantalla. Por el id — el título en minúsculas, igual que
    llega dibujado — se buscan los ficheros reales del vault; si un título
    está repetido, gana la nota menos profunda, que es la que Obsidian
    resolvería primero con su enlace más corto.
    """
    clave = id_nota.strip().casefold()
    if not clave:
        return "Error: falta qué nota abrir."
    if not ruta_vault.is_dir():
        return f"Error: el vault {ruta_vault} no existe."

    coincidencias: list[Path] = []
    for ruta in sorted(ruta_vault.rglob("*.md")):
        relativa = ruta.relative_to(ruta_vault)
        if any(parte.lower() in EXCLUIDAS for parte in relativa.parts[:-1]):
            continue
        if ruta.stem.casefold() == clave:
            coincidencias.append(relativa)

    if not coincidencias:
        return f"Error: no hay ninguna nota llamada '{id_nota}' en el vault."

    relativa = min(coincidencias, key=lambda r: (len(r.parts), str(r).casefold()))
    destino = relativa.with_suffix("").as_posix()
    url = (
        "obsidian://open"
        + "?vault=" + urllib.parse.quote(ruta_vault.name)
        + "&file=" + urllib.parse.quote(destino)
    )
    try:
        webbrowser.open(url)
    except OSError as error:
        logger.error("No se pudo abrir Obsidian para %s: %s", destino, error)
        return f"Error: no se pudo abrir Obsidian: {error}"
    return f"Éxito: abierta '{destino}' en Obsidian."
