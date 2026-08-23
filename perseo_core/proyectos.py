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
  3. **Cuatro formas de abrir y ninguna más**: una carpeta en el explorador, una
     URL http/https en el navegador, un programa de la lista blanca del agente
     `pc` con la carpeta del proyecto como argumento, o **el arranque que el
     propio proyecto declare** (`modo: "arranque"`).

ENMIENDA DEL 2026-08-21, PEDIDA POR EL SEÑOR PERSUS
---------------------------------------------------
Aquí ponía que un `orden` libre no existía y que no era un descuido, porque
sería una shell remota con otro nombre. Sigue siendo verdad **de una orden que
llegue por la petición**, y eso no ha cambiado ni va a cambiar. Lo que se añade
es otra cosa: una orden que ya está **escrita en el fichero del disco**, junto al
resto del proyecto, por la misma persona que podría abrir una terminal y
escribirla a mano.

La diferencia no es de matiz. Quien escribe `proyectos.json` está delante de la
máquina; quien llega por HTTP manda un `id` y nada más. Si alguien puede escribir
ese fichero, ya tiene la máquina — la shell remota se la daría el sistema
operativo, no este módulo.

Lo que **no** se relaja al añadirlo:

  * `arranque` es una **lista de argumentos**, nunca una línea para un shell:
    `["npm", "run", "dev"]`, no `"npm run dev"`. Sin `shell=True` en ninguna
    parte, el sistema no vuelve a parsear nada.
  * El programa tiene que **existir** al validar la lista, o la entrada se
    descarta con un aviso como cualquier otra mal escrita.
  * La `carpeta` desde la que se arranca tiene que ser una carpeta de verdad.

Sin fichero no hay proyectos, y eso es un estado válido: el panel enseña cómo
crearlo y no se rompe nada.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import webbrowser
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import pc

logger = logging.getLogger(__name__)

#: Cómo se abre un proyecto. `programa` se resuelve contra la lista blanca del
#: agente `pc`, que es la misma lista que ya decide qué puede abrir Perseo por
#: voz: dos listas distintas acabarían discrepando. `arranque` es el añadido del
#: 2026-08-21: lo que el proyecto declare, en lista de argumentos y sin shell.
MODOS = ("carpeta", "url", "programa", "arranque")

NOMBRE_FICHERO = "proyectos.json"


@dataclass(frozen=True)
class Proyecto:
    """Una entrada de la lista, ya validada."""

    id: str
    nombre: str
    modo: str
    destino: str
    #: Para `modo == "programa"`: qué carpeta se le pasa como argumento.
    #: Para `modo == "arranque"`: desde qué carpeta se arranca.
    carpeta: str = ""
    descripcion: str = ""
    #: Solo para `modo == "arranque"`: la orden, ya troceada en argumentos.
    arranque: tuple[str, ...] = ()

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

    if not id_proyecto:
        logger.warning("Proyecto sin id en %s; se ignora.", NOMBRE_FICHERO)
        return None
    if modo not in MODOS:
        logger.warning("Proyecto %r con modo %r desconocido; se ignora.", id_proyecto, modo)
        return None
    # `arranque` es el único modo que no tiene destino: lo que se abre es la
    # orden que trae, y pedirle además un destino sería pedir un dato de adorno.
    if modo != "arranque" and not destino:
        logger.warning("Proyecto %r sin destino; se ignora.", id_proyecto)
        return None
    if modo == "url" and not pc._es_url(destino):
        logger.warning("Proyecto %r: %r no es una URL http/https.", id_proyecto, destino)
        return None
    if modo == "programa" and destino.lower() not in pc.APLICACIONES_PERMITIDAS:
        logger.warning(
            "Proyecto %r: %r no está en la lista blanca del agente pc.", id_proyecto, destino
        )
        return None

    arranque: tuple[str, ...] = ()
    if modo == "arranque":
        arranque = _arranque_valido(id_proyecto, crudo.get("arranque"))
        if not arranque:
            return None

    return Proyecto(
        id=id_proyecto,
        nombre=nombre,
        modo=modo,
        destino=destino,
        carpeta=str(crudo.get("carpeta", "")).strip(),
        descripcion=str(crudo.get("descripcion", "")).strip(),
        arranque=arranque,
    )


def _arranque_valido(id_proyecto: str, crudo: Any) -> tuple[str, ...]:
    """La orden de arranque, si está bien escrita. Vacía si no.

    Se exige **lista**, y no una cadena, a propósito: `"npm run dev"` en una sola
    pieza solo se puede ejecutar pasándoselo a un shell, y ahí es donde viven las
    comillas, los `&&` y el resto de la familia. Troceada, `subprocess` la pasa
    tal cual y el sistema no vuelve a leer nada.
    """
    if not isinstance(crudo, (list, tuple)) or not crudo:
        logger.warning(
            "Proyecto %r: 'arranque' tiene que ser una lista de argumentos, "
            "como [\"npm\", \"run\", \"dev\"].",
            id_proyecto,
        )
        return ()

    argumentos = [str(pieza).strip() for pieza in crudo]
    if not all(argumentos):
        logger.warning("Proyecto %r: 'arranque' tiene un argumento vacío.", id_proyecto)
        return ()

    if _resolver_programa(argumentos[0]) is None:
        # Igual que una carpeta que ya no existe: se descarta la entrada y se
        # dice, en vez de dejarla en la lista para que falle al pulsarla.
        logger.warning(
            "Proyecto %r: no se encuentra el programa %r del arranque.",
            id_proyecto,
            argumentos[0],
        )
        return ()

    return tuple(argumentos)


def _resolver_programa(programa: str) -> str | None:
    """Dónde está el ejecutable, o `None` si no está.

    Vale una ruta absoluta o un nombre que esté en el PATH. `shutil.which` es lo
    que resuelve también los `.cmd` y `.bat` de Windows, que es como se instalan
    `npm` y compañía: sin esto, `npm` no se encontraría nunca en esta máquina.
    """
    if os.path.isabs(programa):
        return programa if os.path.isfile(programa) else None
    return shutil.which(programa)


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

        if proyecto.modo == "arranque":
            return _arrancar(proyecto)

        tipo, objetivo = pc.APLICACIONES_PERMITIDAS[proyecto.destino.lower()]
        if tipo != "exe":
            return f"Error: '{proyecto.destino}' no se puede abrir con una carpeta dentro."
        # Con la ruta resuelta y no el nombre desnudo: `Popen(["chrome.exe"])`
        # solo funciona si está en el PATH, y los navegadores no lo están — es
        # exactamente lo que dice la cabecera de `pc.py`. Sin resolver, un
        # proyecto "programa" de Chrome o Firefox fallaba al pulsarlo.
        ruta = pc.resolver_ejecutable(objetivo)
        argumentos = [ruta or objetivo]
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


def _arrancar(proyecto: Proyecto) -> str:
    """Lanza el arranque declarado por el proyecto y se desentiende.

    No se espera a que termine ni se lee su salida: lo que se arranca aquí es un
    servidor de desarrollo o un editor, cosas que duran horas. Lo que se contesta
    es que se ha lanzado, que es lo único que se puede saber en ese momento.
    """
    carpeta = Path(proyecto.carpeta) if proyecto.carpeta else None
    if carpeta is not None and not carpeta.is_dir():
        return f"Error: la carpeta de '{proyecto.nombre}' ya no existe: {carpeta}"

    programa = _resolver_programa(proyecto.arranque[0])
    if programa is None:
        return f"Error: ya no se encuentra '{proyecto.arranque[0]}' en esta máquina."

    argumentos = [programa, *proyecto.arranque[1:]]

    # Sin ventana negra: esto lo lanza el núcleo, que no tiene consola, y una
    # consola huérfana se queda ahí hasta que alguien la cierra.
    banderas = 0
    if os.name == "nt":
        banderas = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS

    subprocess.Popen(
        argumentos,
        cwd=str(carpeta) if carpeta is not None else None,
        creationflags=banderas,
        close_fds=True,
    )
    orden = " ".join(proyecto.arranque)
    return f"Éxito: arrancado {proyecto.nombre} con «{orden}»."
