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
  3. **Cinco formas de abrir y ninguna más**: una carpeta en el explorador, una
     URL http/https en el navegador, un programa de la lista blanca del agente
     `pc` con la carpeta del proyecto como argumento, **el arranque que el
     propio proyecto declare** (`modo: "arranque"`), o **el servicio que el
     propio proyecto declare** (`modo: "servicio"`, añadido el 2026-08-24).

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

EL SERVICIO DEL 2026-08-24, PEDIDO POR EL SEÑOR PERSUS
-------------------------------------------------------
*«que cuando lo pulse se abra el proyecto en una pestaña nueva, no la carpeta;
que corra el código del proyecto, runneándolo». Es decir: pulsar «CVScraper ·
App» tiene que dejar el backend y el frontend corriendo y abrir su URL en una
pestaña del navegador, sin pasar por VS Code ni por una terminal.

Lo que hace `servicio`, y lo que no:

  * Arranca **los procesos que el fichero declare** (`servidores`: lista de
    órdenes, cada una con su carpeta) y abre la pestaña cuando el puerto de
    `destino` contesta — no antes, porque una pestaña sobre un servidor a medio
    cargar parece un proyecto roto.
  * Si el puerto ya está abierto, **no arranca nada**: abre la pestaña y dice
    que ya estaba. Y mientras arranca, una segunda pulsación no duplica
    procesos: hay un cerrojo por proyecto que se libera solo.
  * La salida de los procesos va a `<datos>/proyectos_logs/<id>.log`, porque un
    servidor desprendido que escribe en el vacío es un fallo sin pistas.
  * Lo mismo que siempre: órdenes escritas en el fichero del disco por una
    persona, listas de argumentos, sin shell, y por HTTP solo el `id`.

Sin fichero no hay proyectos, y eso es un estado válido: el panel enseña cómo
crearlos y no se rompe nada.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.parse
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
#: `servicio` es el del 2026-08-24: arrancar los procesos del proyecto y abrir
#: su pestaña cuando respiren — ver la enmienda de la cabecera.
MODOS = ("carpeta", "url", "programa", "arranque", "servicio")

NOMBRE_FICHERO = "proyectos.json"

#: Cuánto espera el vigilante de un servicio a que su puerto conteste antes de
#: darse por vencido. Vite y uvicorn tardan segundos; no minutos.
PLAZO_ARRANQUE_SEGUNDOS = 90

#: Los servicios que están arrancando ahora mismo, con cerrojo: una segunda
#: pulsación mientras un proyecto despega no tiene que duplicar procesos.
_EN_MARCHA: set[str] = set()
_CERROJO_SERVICIOS = threading.Lock()


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
    #: Solo para `modo == "servicio"`: los procesos que hay que arrancar, cada
    #: uno como {"arranque": [...], "carpeta": "..."}.
    servidores: tuple[dict[str, Any], ...] = ()
    #: Solo para `modo == "servicio"`: cómo quiere el proyecto que se le mire,
    #: {"ancho": …, "alto": …}. Cada app sabe el sitio que necesita.
    ventana: dict[str, int] | None = None
    #: Su color de identidad, para la franja superior de su ficha en el riel.
    #: Decorativo y validado con celo: si no es un #rrggbb, se ignora sin más.
    color: str = ""

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
    if modo in ("url", "servicio") and not pc._es_url(destino):
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

    servidores: tuple[dict[str, Any], ...] = ()
    ventana: dict[str, int] | None = None
    if modo == "servicio":
        servidores = _servidores_validos(id_proyecto, crudo.get("servidores"))
        if not servidores:
            return None
        # Una ventana mal escrita descarta la entrada, como cualquier otra;
        # la que no se declara vale con el tamaño por defecto.
        ventana = _ventana_valida(id_proyecto, crudo.get("ventana"))
        if ventana == {}:
            return None

    return Proyecto(
        id=id_proyecto,
        nombre=nombre,
        modo=modo,
        destino=destino,
        carpeta=str(crudo.get("carpeta", "")).strip(),
        descripcion=str(crudo.get("descripcion", "")).strip(),
        arranque=arranque,
        servidores=servidores,
        ventana=ventana,
        color=_color_valido(crudo.get("color")),
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


def _color_valido(crudo: Any) -> str:
    """El color de identidad del proyecto, si es un #rrggbb limpio.

    Decorativo: una entrada con un color mal escrito no se pierde por eso —
    se queda sin color y el riel usa el suyo.
    """
    if isinstance(crudo, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", crudo.strip()):
        return crudo.strip()
    return ""


def _servidores_validos(id_proyecto: str, crudo: Any) -> tuple[dict[str, Any], ...]:
    """Los procesos de un servicio, si están bien escritos. Vacío si no.

    Cada uno es un objeto con su orden — lista de argumentos, jamás una cadena,
    por lo mismo que `arranque` — y su carpeta opcional: cvscraper necesita dos
    procesos a la vez y no arrancan desde el mismo sitio.
    """
    if not isinstance(crudo, (list, tuple)) or not crudo:
        logger.warning(
            "Proyecto %r: 'servidores' tiene que ser una lista, como "
            '[{"arranque": ["python", "servidor.py"], "carpeta": "..."}].',
            id_proyecto,
        )
        return ()

    servidores: list[dict[str, Any]] = []
    for pieza in crudo:
        if not isinstance(pieza, dict):
            logger.warning(
                "Proyecto %r: cada servidor es un objeto con 'arranque'; se ignora la entrada.",
                id_proyecto,
            )
            return ()
        arranque = _arranque_valido(id_proyecto, pieza.get("arranque"))
        if not arranque:
            return ()
        servidores.append(
            {
                "arranque": list(arranque),
                "carpeta": str(pieza.get("carpeta", "")).strip(),
            }
        )
    return tuple(servidores)


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

        if proyecto.modo == "servicio":
            return _servir(proyecto, directorio_datos)

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

    # Sin ventana negra: esto lo lanza el núcleo, que no tiene consola. Solo
    # CREATE_NO_WINDOW — y no DETACHED_PROCESS, que se probó aquí y resultó
    # trampa (2026-08-24): con él, la salida de los .cmd de Windows (npm…) se
    # pierde entera y un servidor muerto deja el registro vacío, sin una pista.
    banderas = 0
    if os.name == "nt":
        banderas = subprocess.CREATE_NO_WINDOW

    subprocess.Popen(
        argumentos,
        cwd=str(carpeta) if carpeta is not None else None,
        creationflags=banderas,
        close_fds=True,
    )
    orden = " ".join(proyecto.arranque)
    return f"Éxito: arrancado {proyecto.nombre} con «{orden}»."


def _puerto_abierto(url: str, plazo: float = 0.6) -> bool:
    """True si algo escucha ya en el puerto de la URL.

    Un socket que conecta y nada más: no se lee HTTP, porque lo único que hay
    que saber aquí es «¿respira?». Se prueban todas las direcciones que dé el
    DNS — `localhost` puede ser IPv4 o IPv6 según el día — y con `connect_ex`
    no hay excepciones ruidosas cuando no llega.
    """
    partes = urllib.parse.urlsplit(url)
    host = partes.hostname or "127.0.0.1"
    puerto = partes.port or (443 if partes.scheme == "https" else 80)
    try:
        direcciones = socket.getaddrinfo(host, puerto, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False
    for familia, _tipo, _proto, _canonico, direccion in direcciones:
        try:
            with socket.socket(familia, socket.SOCK_STREAM) as enchufe:
                enchufe.settimeout(plazo)
                if enchufe.connect_ex(direccion) == 0:
                    return True
        except OSError:
            continue
    return False


def _abrir_navegador(url: str) -> None:
    """Una pestaña nueva. Función aparte para que las pruebas la sustituyan."""
    webbrowser.open_new_tab(url)


def _ventana_valida(id_proyecto: str, crudo: Any) -> dict[str, int] | None:
    """El tamaño de ventana que el proyecto declara para sí, si es válido.

    Cada app sabe cómo quiere que se la mire: el tablero de cvscraper pide
    más sitio que la consola de MAGI. Lo declara el fichero del disco — como
    todo aquí — y se exige número entero razonable, no cualquier cosa.
    """
    if crudo is None:
        return None
    if not isinstance(crudo, dict):
        logger.warning(
            "Proyecto %r: 'ventana' tiene que ser un objeto con 'ancho' y 'alto'.",
            id_proyecto,
        )
        return {}
    try:
        ancho = int(crudo.get("ancho", 0))
        alto = int(crudo.get("alto", 0))
    except (TypeError, ValueError):
        logger.warning("Proyecto %r: 'ventana' con medidas que no son números.", id_proyecto)
        return {}
    if not (300 <= ancho <= 7680 and 200 <= alto <= 4320):
        logger.warning("Proyecto %r: medidas de ventana fuera de todo sentido.", id_proyecto)
        return {}
    return {"ancho": ancho, "alto": alto}


def puerto_responde(url: str, plazo: float = 0.35) -> bool:
    """True si el servicio del proyecto está en marcha ahora mismo.

    Es lo que mira la pantalla al abrir el carril: una ficha que dijera
    «LANZAR» de una app que ya está corriendo mentiría, y lo mismo al revés.
    El plazo es corto a propósito — se pregunta por dos o tres puertos locales
    y nadie espera una respuesta de red para pintar una lista.
    """
    return _puerto_abierto(url, plazo=plazo)


def _cuando_este_listo(id_proyecto: str, nombre: str, url: str) -> None:
    """El vigilante de un servicio recién arrancado.

    Corre en un hilo suelto y muerto: espera a que el puerto conteste — la
    señal de que los procesos despegaron, que es lo que el estado vivo de
    `/proyectos` enseña a la pantalla — y libera el cerrojo pase lo que pase,
    o una segunda pulsación creería para siempre que sigue despegando. La
    ventana NO se abre aquí: la abre el cliente que pidió el arranque, cuando
    vea `vivo` en verde, con el tamaño que el proyecto pidió para sí.
    """
    try:
        # Dos miradas por segundo es suficiente para algo que tarda segundos,
        # y barato para la máquina.
        for _ in range(PLAZO_ARRANQUE_SEGUNDOS * 2):
            if _puerto_abierto(url):
                return
            time.sleep(0.5)
        logger.error(
            "%s no abrió su puerto en %s s; mira datos/proyectos_logs/%s.log",
            nombre,
            PLAZO_ARRANQUE_SEGUNDOS,
            id_proyecto,
        )
    finally:
        with _CERROJO_SERVICIOS:
            _EN_MARCHA.discard(id_proyecto)


def _servir(proyecto: Proyecto, directorio_datos: Path) -> str:
    """Arranca los procesos del servicio y deja a alguien esperando su puerto.

    Devuelve enseguida: lo que arranca aquí son servidores que duran horas, y
    la pestaña la abre el vigilante (`_cuando_este_listo`) cuando de verdad hay
    algo detrás. Si el puerto ya respira, no se arranca nada — abrir dos veces
    el mismo proyecto no puede costar dos servidores.
    """
    with _CERROJO_SERVICIOS:
        if proyecto.id in _EN_MARCHA:
            return (
                f"Éxito: {proyecto.nombre} ya se está arrancando; "
                "la pestaña se abre sola."
            )
        _EN_MARCHA.add(proyecto.id)

    entregado = False
    try:
        if _puerto_abierto(proyecto.destino):
            return f"Éxito: {proyecto.nombre} ya está en marcha."

        # La salida de los procesos, apuntada y no perdida: un servidor
        # desprendido que escribe en el vacío es un fallo sin pistas.
        registros = Path(directorio_datos) / "proyectos_logs"
        try:
            registros.mkdir(parents=True, exist_ok=True)
        except OSError:
            registros = None

        # CREATE_NO_WINDOW y no DETACHED_PROCESS, por lo mismo que en
        # `_arrancar`: detached silencia del todo la salida de los .cmd, y el
        # registro compartido es la única pista cuando un servidor muere.
        banderas = 0
        if os.name == "nt":
            banderas = subprocess.CREATE_NO_WINDOW

        lanzamientos: list[tuple[list[str], str | None]] = []
        for servidor in proyecto.servidores:
            programa = _resolver_programa(servidor["arranque"][0])
            if programa is None:
                return (
                    f"Error: ya no se encuentra '{servidor['arranque'][0]}' "
                    "en esta máquina."
                )
            carpeta = servidor["carpeta"] or proyecto.carpeta
            cwd: str | None = None
            if carpeta:
                ruta = Path(carpeta)
                if not ruta.is_dir():
                    return (
                        f"Error: la carpeta de '{proyecto.nombre}' ya no existe: {ruta}"
                    )
                cwd = str(ruta)
            lanzamientos.append(([programa, *servidor["arranque"][1:]], cwd))

        # Un solo registro por servicio, compartido por todos sus procesos y
        # abierto ANTES de lanzar el primero: si algo fallara a mitad, no
        # quedaría un proceso huérfano sin sitio donde escribir.
        salida: Any
        if registros is not None:
            salida = open(registros / f"{proyecto.id}.log", "ab")
        else:
            salida = subprocess.DEVNULL
        try:
            for argumentos, cwd in lanzamientos:
                subprocess.Popen(
                    argumentos,
                    cwd=cwd,
                    stdin=subprocess.DEVNULL,
                    stdout=salida,
                    stderr=subprocess.STDOUT,
                    creationflags=banderas,
                    close_fds=True,
                )
        finally:
            if salida is not subprocess.DEVNULL:
                salida.close()

        threading.Thread(
            target=_cuando_este_listo,
            args=(proyecto.id, proyecto.nombre, proyecto.destino),
            daemon=True,
            name=f"servicio-{proyecto.id}",
        ).start()
        entregado = True
        return (
            f"Éxito: arrancando {proyecto.nombre}; "
            "su ventana se abre sola cuando esté listo."
        )
    finally:
        if not entregado:
            with _CERROJO_SERVICIOS:
                _EN_MARCHA.discard(proyecto.id)
