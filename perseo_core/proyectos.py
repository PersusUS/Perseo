"""Los otros proyectos, abiertos desde el panel de Perseo.

Perseo no es lo único que hay en esta máquina: están `armario`, `cvscraper`, el
proyecto MAGI y lo que venga. Abrirlos era ir a buscar la carpeta cada vez, y el
panel ya está delante.

REGLA DE SEGURIDAD DE ESTE MÓDULO
---------------------------------
Esto ejecuta cosas, y el panel se alcanza **desde el tailnet**. Así que aquí
dentro se aplica lo mismo que en el agente `pc` (H-16), y una condición más que
es la que de verdad sostiene todo:

  1. **Lo que se puede abrir sale de un fichero del disco**, `<datos>/proyectos.json`,
     escrito por una persona. Nunca de la petición: por HTTP llega *cuál* de los
     proyectos de la lista, jamás qué ejecutar.
  2. **Nunca se invoca un shell.** Listas de argumentos, que el sistema no
     vuelve a parsear.
  3. **Tres formas de abrir y ninguna más**: una carpeta en el explorador, una
     URL http/https en el navegador, o un programa de la lista blanca del agente
     `pc` con la carpeta del proyecto como argumento. Un `orden` libre no
     existe, y no es un descuido: sería una shell remota con otro nombre.

Sin fichero no hay proyectos, y eso es un estado válido: el panel enseña cómo
crearlo y no se rompe nada.
"""

from __future__ import annotations

import json
import logging
import subprocess
import webbrowser
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import pc

logger = logging.getLogger(__name__)

#: Cómo se abre un proyecto. `programa` se resuelve contra la lista blanca del
#: agente `pc`, que es la misma lista que ya decide qué puede abrir Perseo por
#: voz: dos listas distintas acabarían discrepando.
MODOS = ("carpeta", "url", "programa")

NOMBRE_FICHERO = "proyectos.json"


@dataclass(frozen=True)
class Proyecto:
    """Una entrada de la lista, ya validada."""

    id: str
    nombre: str
    modo: str
    destino: str
    #: Para `modo == "programa"`: qué carpeta se le pasa como argumento.
    carpeta: str = ""
    descripcion: str = ""

    def a_dict(self) -> dict[str, Any]:
        return asdict(self)


def _valido(crudo: dict[str, Any]) -> Proyecto | None:
    """Convierte una entrada del fichero en `Proyecto`, o la descarta.

    Una entrada mal escrita se ignora con un aviso en el registro en vez de
    tumbar la lista entera: el fichero lo escribe una persona a mano, y perder
    los cinco proyectos buenos por una coma es peor que perder el malo.
    """
    id_proyecto = str(crudo.get("id", "")).strip()
    nombre = str(crudo.get("nombre", "")).strip() or id_proyecto
    modo = str(crudo.get("modo", "")).strip().lower()
    destino = str(crudo.get("destino", "")).strip()

    if not id_proyecto or not destino:
        logger.warning("Proyecto sin id o sin destino en %s; se ignora.", NOMBRE_FICHERO)
        return None
    if modo not in MODOS:
        logger.warning("Proyecto %r con modo %r desconocido; se ignora.", id_proyecto, modo)
        return None
    if modo == "url" and not pc._es_url(destino):
        logger.warning("Proyecto %r: %r no es una URL http/https.", id_proyecto, destino)
        return None
    if modo == "programa" and destino.lower() not in pc.APLICACIONES_PERMITIDAS:
        logger.warning(
            "Proyecto %r: %r no está en la lista blanca del agente pc.", id_proyecto, destino
        )
        return None

    return Proyecto(
        id=id_proyecto,
        nombre=nombre,
        modo=modo,
        destino=destino,
        carpeta=str(crudo.get("carpeta", "")).strip(),
        descripcion=str(crudo.get("descripcion", "")).strip(),
    )


def listar(directorio_datos: Path) -> list[Proyecto]:
    """Los proyectos declarados en `<datos>/proyectos.json`. Sin fichero, ninguno."""
    ruta = Path(directorio_datos) / NOMBRE_FICHERO
    try:
        crudo = json.loads(ruta.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("No se pudo leer %s: %s", ruta, e)
        return []

    if not isinstance(crudo, list):
        logger.warning("%s debería ser una lista de proyectos.", ruta)
        return []

    proyectos = [_valido(c) for c in crudo if isinstance(c, dict)]
    return [p for p in proyectos if p is not None]


def abrir(directorio_datos: Path, id_proyecto: str) -> str:
    """Abre un proyecto de la lista. Devuelve 'Éxito: …' o 'Error: …'.

    Devuelve texto y no lanza por lo mismo que `pc.controlar`: lo que sale de
    aquí acaba en una pantalla, y un fallo al abrir una carpeta no es una
    excepción del núcleo.
    """
    for proyecto in listar(directorio_datos):
        if proyecto.id == id_proyecto:
            break
    else:
        return f"Error: no hay ningún proyecto llamado '{id_proyecto}'."

    try:
        if proyecto.modo == "url":
            webbrowser.open(proyecto.destino)
            return f"Éxito: abierto {proyecto.nombre} en el navegador."

        if proyecto.modo == "carpeta":
            carpeta = Path(proyecto.destino)
            if not carpeta.is_dir():
                return f"Error: la carpeta de '{proyecto.nombre}' ya no existe: {carpeta}"
            # Sin shell y con la ruta como argumento aparte, igual que en `pc`.
            subprocess.Popen(["explorer.exe", str(carpeta)])
            return f"Éxito: abierta la carpeta de {proyecto.nombre}."

        tipo, objetivo = pc.APLICACIONES_PERMITIDAS[proyecto.destino.lower()]
        if tipo != "exe":
            return f"Error: '{proyecto.destino}' no se puede abrir con una carpeta dentro."
        argumentos = [objetivo]
        if proyecto.carpeta:
            carpeta = Path(proyecto.carpeta)
            if not carpeta.is_dir():
                return f"Error: la carpeta de '{proyecto.nombre}' ya no existe: {carpeta}"
            argumentos.append(str(carpeta))
        subprocess.Popen(argumentos)
        return f"Éxito: abierto {proyecto.nombre} con {proyecto.destino}."

    except OSError as e:
        logger.error("No se pudo abrir el proyecto %s: %s", id_proyecto, e)
        return f"Error: no se pudo abrir '{proyecto.nombre}': {e}"
