"""Estado persistente del núcleo: configuración, base de datos y cola de trabajos.

La cola es la pieza central de la Fase A, y la razón es un requisito concreto
(R10 del plan): una petición hecha por voz tiene que sobrevivir a que se cuelgue
la llamada. Si el trabajo viviera en el contexto de la sesión de Gemini, colgar
lo borraría. Viviendo en SQLite, la siguiente llamada lo encuentra intacto.

Concurrencia: se usa una sola conexión con un cerrojo. Las llamadas entran desde
el bucle de asyncio vía `asyncio.to_thread`, que reparte entre varios hilos, y
`sqlite3` no admite compartir conexión entre hilos sin más. Con el volumen de un
sistema personal, serializar los accesos no cuesta nada medible y evita toda una
clase de errores.


"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import secrets
import socket
import sqlite3
import subprocess
import threading
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: La raíz del paquete: un escalón por encima de `infra/`. De aquí cuelgan
#: `datos/` y, un escalón más arriba, el vault por defecto.
RAIZ = Path(__file__).resolve().parent.parent

# Estados de un trabajo.
PENDIENTE = "pendiente"
EN_CURSO = "en_curso"
#: El agente ha parado a medias porque lo que viene es irreversible y quiere un
#: sí. No lo ejecuta nadie mientras esté así, y **no** es un huérfano: al
#: reiniciar el núcleo sigue esperando exactamente igual.
ESPERANDO = "esperando"
HECHO = "hecho"
FALLIDO = "fallido"
CANCELADO = "cancelado"
RECHAZADO = "rechazado"

#: Estados desde los que un trabajo todavía puede moverse. El resto son finales.
ABIERTOS = (PENDIENTE, EN_CURSO, ESPERANDO)

ORIGENES = ("voz", "texto", "disparador")

# Qué se ha hecho con un correo triado. `PENDIENTE_CORREO` no se guarda: es lo
# que significa no estar en la tabla, y marcar uno como pendiente otra vez es
# borrar la fila. Así el estado de un correo que nadie ha tocado no depende de
# que alguien lo escribiera bien.
ATENDIDO = "atendido"
DESCARTADO = "descartado"
PENDIENTE_CORREO = "pendiente"
ESTADOS_CORREO = (ATENDIDO, DESCARTADO, PENDIENTE_CORREO)

_ESQUEMA = """
CREATE TABLE IF NOT EXISTS trabajos (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    estado          TEXT    NOT NULL,
    agente          TEXT    NOT NULL,
    origen          TEXT    NOT NULL,
    -- Quién lo pidió, con el nombre del perfil que puso el reconocimiento de
    -- voz («Persus», «Javi»), o NULL cuando no se sabe. `origen` dice por qué
    -- puerta entró el trabajo; esta columna, de quién es la voz que lo pidió, y
    -- es lo que mira la política para no dejar que una visita mueva las manos.
    quien           TEXT,
    peticion        TEXT    NOT NULL,
    resultado       TEXT,
    error           TEXT,
    confirmacion    TEXT,
    intentos        INTEGER NOT NULL DEFAULT 0,
    creado_en       TEXT    NOT NULL,
    actualizado_en  TEXT    NOT NULL,
    reclamado_en    TEXT
);

CREATE INDEX IF NOT EXISTS idx_trabajos_pendientes ON trabajos (estado, id);
CREATE INDEX IF NOT EXISTS idx_trabajos_recientes  ON trabajos (creado_en DESC);

-- Cuántas veces se ha llamado hoy a cada servicio de fuera. Existe porque la
-- cuota gratuita de Gemini no se puede consultar: Google no publica ningún
-- endpoint que diga cuánto queda, así que lo único honesto es contar lo que
-- gasta este proceso. Ver `apuntar_uso`.
CREATE TABLE IF NOT EXISTS uso (
    dia       TEXT    NOT NULL,
    servicio  TEXT    NOT NULL,
    contador  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (dia, servicio)
);

-- Qué se ha hecho con cada correo triado. El triaje dice de qué va un correo y
-- ahí se quedaba: «requiere acción» sin forma de decir que ya está. Solo se
-- guardan los que se han tocado; lo que no aparece está pendiente, así que la
-- tabla crece con las decisiones y no con el buzón.
CREATE TABLE IF NOT EXISTS correos (
    id_mensaje      TEXT PRIMARY KEY,
    estado          TEXT NOT NULL,
    actualizado_en  TEXT NOT NULL
);

-- El chat escrito. La conversación vive aquí y no en la cara por la misma
-- razón que la cola: un turno de chat puede tardar minutos (herramientas,
-- web, encargos) y sobrevivir a que se cierre la pantalla es la regla R10
-- otra vez. `turno` es el semáforo de una conversación a la vez: mientras está
-- `ocupado`, otra petición para la misma sesión recibe un 409 en vez de
-- entrelazar dos respuestas.
CREATE TABLE IF NOT EXISTS chat_sesiones (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    titulo          TEXT    NOT NULL DEFAULT '',
    turno           TEXT    NOT NULL DEFAULT 'libre',
    creado_en       TEXT    NOT NULL,
    actualizado_en  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_mensajes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    sesion          INTEGER NOT NULL,
    rol             TEXT    NOT NULL,
    texto           TEXT    NOT NULL DEFAULT '',
    herramientas    TEXT    NOT NULL DEFAULT '[]',
    estado          TEXT    NOT NULL DEFAULT 'hecho',
    momento         TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chat_mensajes ON chat_mensajes (sesion, id);
"""

#: Columnas añadidas después de que hubiera bases de datos por ahí. `CREATE
#: TABLE IF NOT EXISTS` no las añade a una tabla que ya existe, así que hay que
#: mirarlo a mano al abrir.
_COLUMNAS_NUEVAS = {"confirmacion": "TEXT", "quien": "TEXT"}


# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Configuracion:
    """Ajustes del núcleo, resueltos una vez al arrancar."""

    #: Interfaces en las que se escucha. Es una lista porque en la Fase B el
    #: núcleo atiende a la vez al PC (bucle local) y al móvil (Tailscale), y
    #: enumerarlas es lo que permite no caer nunca en `0.0.0.0`.
    hosts: tuple[str, ...]
    puerto: int
    token: str
    ruta_db: Path
    directorio_datos: Path
    url_ollama: str
    modelo_router: str
    #: Credenciales del bot. Vacías significa "sin Telegram", y el núcleo
    #: arranca igual: es un canal más, no una pieza de la que dependa nada.
    telegram_token: str
    telegram_chat: str
    #: Se puede apuntar a otro sitio para probar sin tocar Telegram de verdad.
    telegram_api: str
    #: La dirección que se pone en el enlace "ver detalle" de las
    #: notificaciones. Va a un mensaje que sale de la máquina, así que apunta al
    #: tailnet y no al bucle local.
    url_base: str
    #: Disparadores que se ponen en marcha al arrancar (Fase D). Vacío es un
    #: estado válido: el núcleo funciona igual, solo que nadie empieza nada solo.
    disparadores: tuple[str, ...]
    #: Cada cuántos segundos le toca a cada disparador. Lo que no esté aquí usa
    #: el valor con el que se registró.
    intervalos: dict[str, float]
    #: De dónde salen los correos: `falso` (fichero, para verificar) o vacío.
    #: Cuando haya credenciales OAuth se añadirá `gmail`.
    correo_buzon: str
    #: Ruta del JSON que hace de buzón cuando `correo_buzon` es `falso`.
    correo_falso: str
    #: Motor del agente `dev`: vacío (Claude Code por línea de comandos) o
    #: `falso`, que simula sin gastar suscripción.
    dev_motor: str
    #: Qué se ejecuta cuando el motor es el de verdad.
    dev_ejecutable: str
    #: Desde dónde trabaja un encargo de código, y de dónde no puede salir.
    #: Por defecto la CARPETA DEL USUARIO, no este repositorio: los proyectos
    #: del señor Persus están interconectados y un agente encerrado en uno solo
    #: no puede leer el de al lado (2026-08-26). Se estrecha con
    #: `PERSEO_DEV_RAIZ` si alguna vez hace falta.
    dev_raiz: str
    #: Segundos que se le dan a un encargo antes de cortarlo.
    dev_tope: float
    #: Lo que tarda el motor falso, para poder comprobar que un encargo largo no
    #: deja al resto de la cola esperando.
    dev_tardanza_falsa: float
    #: Fichero con las credenciales de Google (Gmail y Calendar). Va en el
    #: directorio de datos, que está fuera de git: lleva un `refresh_token`.
    google_credenciales: str
    #: Navegador del agente `web`: vacío (HTTP de verdad) o `falso`.
    web_navegador: str
    #: Cuánto se descarga como mucho de una página.
    web_tope_bytes: int
    #: Cuánto se espera a una página.
    web_tope_segundos: float
    #: **Solo para las verificaciones.** Deja alcanzar el bucle local, que en
    #: producción está prohibido: sin esto no se podría comprobar el camino real
    #: contra un servidor de prueba. Ver `web.comprobar_url`.
    web_local: bool
    #: De dónde salen los eventos: `falso` (fichero, para verificar) o vacío.
    agenda_origen: str
    #: Ruta del JSON que hace de calendario cuando `agenda_origen` es `falso`.
    agenda_falsa: str
    #: Con cuántos minutos de antelación se avisa de un evento.
    agenda_antelacion: int
    #: Raíz del vault de Obsidian. Se sigue leyendo de `OBSIDIAN_VAULT_PATH`,
    #: que es la variable que ya usaba el indexador de v1: quien la tuviera
    #: puesta no tiene que cambiar nada. Que las rutas del vault no coincidieran
    #: entre módulos fue.
    vault: str
    #: Qué hay detrás del puerto del vault: vacío (ficheros, como hasta ahora) o
    #: `rest`, el plugin Local REST API de Obsidian. Con `rest` la ruta del
    #: vault deja de usarse: quien sabe dónde están las notas es Obsidian.
    vault_respaldo: str
    #: Dónde escucha el plugin. Por defecto su HTTPS del bucle local, que es lo
    #: que trae encendido de fábrica.
    vault_rest_url: str
    #: La clave del plugin, que sale en sus ajustes. Si no está en la variable
    #: se lee de `<datos>/obsidian.txt`, igual que el token de Telegram: el
    #: directorio de datos está fuera de git.
    vault_rest_clave: str
    #: Modelo de fuera que responde cuando Ollama no está: `gemma-4-31b-it`, por
    #: ejemplo. **Vacío = apagado**, y es lo que viene de fábrica: es la única
    #: pieza que manda a un tercero el texto que se está clasificando.
    modelo_suplente: str
    #: Clave de la API de Gemini, para el suplente. Se comparte con la que usa la
    #: app para la voz: `GEMINI_API_KEY`, o `<datos>/gemini.txt`.
    gemini_clave: str
    #: Certificado y clave para servir por HTTPS. Existen por el micrófono: el
    #: navegador solo deja grabar en un contexto seguro, y `http://` por el
    #: tailnet no lo es —el bucle local sí, por eso en el PC se puede probar sin
    #: esto—. Los da `tailscale cert`. Vacíos = HTTP de siempre.
    tls_certificado: str
    tls_clave: str
    #: Puerto del HTTPS. **Aparte del de siempre y no en su lugar**: un socket
    #: que habla TLS no contesta a quien llega en claro, así que servir HTTPS en
    #: el puerto de siempre no cambia la dirección, la rompe — y con ella los
    #: accesos directos que ya hay guardados. Ver `_donde_escuchar`.
    tls_puerto: int

    @property
    def telegram_configurado(self) -> bool:
        return bool(self.telegram_token and self.telegram_chat)

    @property
    def tls_listo(self) -> bool:
        """Si hay con qué servir HTTPS. Que los ficheros existan se mira aquí:
        una ruta escrita a mano que ya no apunta a nada dejaría al núcleo sin
        arrancar, y prefiero HTTP a nada."""
        if not (self.tls_certificado and self.tls_clave):
            return False
        return Path(self.tls_certificado).is_file() and Path(self.tls_clave).is_file()

    @property
    def url_base_alcanzable(self) -> bool:
        """Si el enlace que sale por Telegram sirve desde fuera de esta máquina.

        `127.0.0.1` en el móvil es **el móvil**: el enlace abre una página en
        blanco y nadie sabe por qué. Pasa siempre que se arranca sin
        `PERSEO_CORE_HOST=tailscale`, porque entonces la única interfaz es la
        local y `_url_por_defecto` no tiene otra cosa que ofrecer.
        """
        anfitrion = urllib.parse.urlsplit(self.url_base).hostname or ""
        return anfitrion not in LOCALES


#: Rango que Tailscale reparte entre los nodos del tailnet (CGNAT).
_RED_TAILSCALE = ipaddress.ip_network("100.64.0.0/10")

#: Sitios donde suele estar la herramienta de Tailscale en Windows y en Linux.
_RUTAS_TAILSCALE = (
    Path(r"C:\Program Files\Tailscale\tailscale.exe"),
    Path("/usr/bin/tailscale"),
    Path("/usr/local/bin/tailscale"),
)

LOCALES = ("127.0.0.1", "localhost", "::1")

#: Que preguntarle su IP a Tailscale no abra una consola. El núcleo arranca sin
#: ventana —lo lanza `pythonw` desde el vigilante—, y `tailscale.exe` es un
#: programa de consola: sin esto parpadeaba una caja negra en cada arranque
#: La salida se captura, así que nadie se pierde nada.
_SIN_VENTANA = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def direcciones_tailscale() -> tuple[str, ...]:
    """Las direcciones de esta máquina en el tailnet: la IPv4 y la IPv6.

    **Las dos, y esto costó una mañana el 2026-08-17.** MagicDNS publica para
    cada nodo un registro `A` y otro `AAAA` —la `fd7a:…`—, y un iPhone que
    resuelve por el túnel prefiere la IPv6. Escuchando solo en la IPv4, entrar
    por la dirección numérica funcionaba y entrar por el nombre no: el móvil
    llamaba a una puerta donde no había nadie. Desde el PC no se veía, porque
    ahí el nombre resolvía a la IPv4.

    La IPv4 va primero: es la que se pone en los enlaces (`_url_por_defecto`),
    donde una IPv6 entre corchetes solo estorba.
    """
    for ruta in _RUTAS_TAILSCALE:
        if not ruta.exists():
            continue
        try:
            salida = subprocess.run(
                [str(ruta), "ip"],
                capture_output=True,
                creationflags=_SIN_VENTANA,
                text=True,
                timeout=10,
                check=False,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            continue
        encontradas = tuple(linea.strip() for linea in salida.splitlines() if linea.strip())
        if encontradas:
            return encontradas

    suelta = _direccion_tailscale_por_interfaz()
    return (suelta,) if suelta else ()


def direccion_tailscale() -> str | None:
    """La IPv4 del tailnet. Se conserva porque es la que va en los enlaces."""
    for direccion in direcciones_tailscale():
        if ":" not in direccion:
            return direccion
    return None


def _direccion_tailscale_por_interfaz() -> str | None:
    """Sin la herramienta de Tailscale, se busca una interfaz del rango CGNAT.

    Funciona, pero conviene saber que ese rango también lo usan algunos
    operadores en la interfaz de salida: por eso es el segundo intento.
    """

    try:
        vistas = {
            info[4][0]
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
        }
    except OSError:
        return None
    for direccion in sorted(vistas):
        if ipaddress.ip_address(direccion) in _RED_TAILSCALE:
            return direccion
    return None


def _resolver_hosts(crudo: str) -> tuple[str, ...]:
    """Convierte `PERSEO_CORE_HOST` en la lista de interfaces donde escuchar.

    Acepta varias direcciones separadas por comas y la palabra `tailscale`, que
    se sustituye por la dirección del tailnet. `tailscale` **no** se resuelve a
    `0.0.0.0` cuando falla: si Tailscale no está levantado, se avisa y se queda
    solo en local. Abrir todas las interfaces por no encontrar una es
    exactamente el fallo que la Fase 3 quería evitar.
    """
    resueltos: list[str] = []
    for parte in crudo.split(","):
        pieza = parte.strip()
        if not pieza:
            continue
        if pieza.lower() != "tailscale":
            resueltos.append(pieza)
            continue

        direcciones = direcciones_tailscale()
        if not direcciones:
            logger.error(
                "Se pidió escuchar en Tailscale pero no se encontró la dirección del "
                "tailnet. ¿Está Tailscale conectado? Se sigue solo en local."
            )
            continue
        # El bucle local va siempre con Tailscale: si no, la app del PC y las
        # pruebas dejarían de poder hablar con el núcleo.
        resueltos.append("127.0.0.1")
        # Las dos del tailnet, IPv4 e IPv6: MagicDNS publica un registro de cada
        # tipo y un iPhone resuelve la IPv6 primero. Con solo la IPv4, entrar
        # por el nombre no llegaba a ninguna parte.
        resueltos.extend(direcciones)

    if not resueltos:
        resueltos.append("127.0.0.1")

    # Sin duplicados y en orden estable: aiohttp falla si se le repite una.
    return tuple(dict.fromkeys(resueltos))


def _directorio_datos() -> Path:
    ruta = Path(os.environ.get("PERSEO_CORE_DATOS", RAIZ / "datos"))
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta


def _resolver_token(directorio: Path) -> str:
    """Devuelve el token de acceso, generándolo la primera vez.

    La variable de entorno manda. Si no está, se usa (o se crea) un fichero en
    el directorio de datos, que está fuera de git.
    """
    del_entorno = os.environ.get("PERSEO_TOKEN", "").strip()
    if del_entorno:
        return del_entorno

    fichero = directorio / "token.txt"
    if fichero.exists():
        guardado = fichero.read_text(encoding="utf-8").strip()
        if guardado:
            return guardado

    nuevo = secrets.token_urlsafe(32)
    fichero.write_text(nuevo, encoding="utf-8")
    logger.warning(
        "Token de acceso generado en %s. Guárdalo: lo necesitas para hablar con el núcleo.",
        fichero,
    )
    return nuevo


def ajustes_guardados(directorio: Path) -> dict[str, str]:
    """Lo que hay en `<datos>/entorno.json`, o nada si no existe.

    **Por qué existe este fichero.** Una entrada del registro de Windows arranca
    un proceso sin las variables de entorno que uno escribe en su terminal. Sin
    esto, el núcleo que arranca con el PC es otro núcleo: sin Gmail, sin agenda,
    sin el vault por Obsidian, y con el enlace de Telegram apuntando al bucle
    local. Arranca, no falla, y hace la mitad — que es peor que no arrancar.

    Un JSON plano de `VARIABLE: valor`. La variable de entorno manda sobre él:
    esto son los valores por defecto de esta instalación, no una orden.
    """
    fichero = directorio / "entorno.json"
    if not fichero.is_file():
        return {}
    try:
        # `utf-8-sig` y no `utf-8`: este fichero se edita a mano, y el Bloc de
        # notas, `Set-Content -Encoding utf8` de PowerShell 5.1 y media Windows
        # le ponen un BOM delante. Con `utf-8` eso es un JSONDecodeError, y el
        # resultado es un núcleo sin correo, sin agenda y sin tailnet que
        # arranca igual y no se queja. Pasó el 2026-08-17.
        crudo = json.loads(fichero.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("No se pudo leer %s (%s); se sigue solo con el entorno.", fichero, e)
        return {}
    if not isinstance(crudo, dict):
        logger.warning("%s no es un objeto JSON; se ignora.", fichero)
        return {}
    return {str(c): str(v) for c, v in crudo.items()}


def cargar_configuracion() -> Configuracion:
    directorio = _directorio_datos()
    guardados = ajustes_guardados(directorio)

    def var(nombre: str, por_defecto: str) -> str:
        """El entorno primero, luego el fichero, luego lo de fábrica."""
        del_entorno = os.environ.get(nombre)
        if del_entorno is not None:
            return del_entorno
        return guardados.get(nombre, por_defecto)

    # Por defecto solo el bucle local. Para que entre el móvil se pone
    # `PERSEO_CORE_HOST=tailscale`, que añade la dirección del tailnet **sin**
    # quitar la local y sin pasar por `0.0.0.0`.
    hosts = _resolver_hosts(var("PERSEO_CORE_HOST", "127.0.0.1"))
    puerto = int(var("PERSEO_CORE_PUERTO", "8787"))

    # Por defecto se registran todos los disparadores conocidos: los que no
    # tengan de dónde tirar se retiran solos al arrancar, igual que Telegram sin
    # token. Se puede acotar la lista, o vaciarla, con `PERSEO_DISPARADORES=`.
    disparadores = tuple(
        pieza.strip()
        for pieza in var("PERSEO_DISPARADORES", "correo,agenda").split(",")
        if pieza.strip()
    )

    return Configuracion(
        hosts=hosts,
        puerto=puerto,
        token=_resolver_token(directorio),
        ruta_db=Path(var("PERSEO_CORE_DB", directorio / "estado.sqlite3")),
        directorio_datos=directorio,
        url_ollama=var("PERSEO_OLLAMA", "http://127.0.0.1:11434"),
        modelo_router=var("PERSEO_MODELO_ROUTER", "qwen3:4b"),
        telegram_token=_de_entorno_o_fichero("PERSEO_TELEGRAM_TOKEN", directorio / "telegram.txt", guardados),
        telegram_chat=_de_entorno_o_fichero("PERSEO_TELEGRAM_CHAT", directorio / "telegram_chat.txt", guardados),
        telegram_api=var("PERSEO_TELEGRAM_API", "https://api.telegram.org").rstrip("/"),
        url_base=var("PERSEO_URL_BASE", "").strip() or _url_por_defecto(hosts, puerto),
        disparadores=disparadores,
        intervalos={
            "correo": float(var("PERSEO_CORREO_INTERVALO", "300")),
            "agenda": float(var("PERSEO_AGENDA_INTERVALO", "600")),
        },
        correo_buzon=var("PERSEO_CORREO", "").strip().lower(),
        correo_falso=var("PERSEO_CORREO_FALSO", str(directorio / "buzon.json")),
        dev_motor=var("PERSEO_DEV_MOTOR", "").strip().lower(),
        dev_ejecutable=var("PERSEO_DEV_CLAUDE", "claude"),
        dev_raiz=var("PERSEO_DEV_RAIZ", str(Path.home())),
        dev_tope=float(var("PERSEO_DEV_TOPE", "900")),
        dev_tardanza_falsa=float(var("PERSEO_DEV_TARDANZA", "0")),
        google_credenciales=var("PERSEO_GOOGLE_CREDENCIALES", str(directorio / "google.json")),
        web_navegador=var("PERSEO_WEB", "").strip().lower(),
        web_tope_bytes=int(var("PERSEO_WEB_TOPE_BYTES", str(2 * 1024 * 1024))),
        web_tope_segundos=float(var("PERSEO_WEB_TOPE_SEGUNDOS", "20")),
        web_local=var("PERSEO_WEB_LOCAL", "").strip() == "1",
        agenda_origen=var("PERSEO_AGENDA", "").strip().lower(),
        agenda_falsa=var("PERSEO_AGENDA_FALSA", str(directorio / "agenda.json")),
        agenda_antelacion=int(var("PERSEO_AGENDA_ANTELACION", "60")),
        vault=var("OBSIDIAN_VAULT_PATH", str(RAIZ.parent / "obsidian_vault")),
        vault_respaldo=var("PERSEO_VAULT", "").strip().lower(),
        vault_rest_url=var("PERSEO_VAULT_REST", "https://127.0.0.1:27124").rstrip("/"),
        vault_rest_clave=_de_entorno_o_fichero("PERSEO_VAULT_CLAVE", directorio / "obsidian.txt", guardados),
        modelo_suplente=var("PERSEO_MODELO_SUPLENTE", "").strip(),
        gemini_clave=_de_entorno_o_fichero("GEMINI_API_KEY", directorio / "gemini.txt", guardados),
        tls_certificado=var("PERSEO_TLS_CERT", ""),
        tls_clave=var("PERSEO_TLS_CLAVE", ""),
        tls_puerto=int(var("PERSEO_TLS_PUERTO", str(puerto + 1))),
    )


def _de_entorno_o_fichero(variable: str, fichero: Path, guardados: dict[str, str] | None = None) -> str:
    """Lee un secreto de la variable de entorno o, si no está, de un fichero.

    El fichero vive en el directorio de datos, que está fuera de git. Es más
    cómodo que exportar la variable en cada arranque, y no deja el token del bot
    en el historial del terminal.
    """
    del_entorno = os.environ.get(variable, "").strip()
    if del_entorno:
        return del_entorno
    guardado = (guardados or {}).get(variable, "").strip()
    if guardado:
        return guardado
    if fichero.exists():
        return fichero.read_text(encoding="utf-8").strip()
    return ""


def _url_por_defecto(hosts: tuple[str, ...], puerto: int) -> str:
    """Dirección para los enlaces que salen de la máquina.

    Se prefiere una interfaz no local: el enlace lo abre el móvil desde el
    tailnet, y `127.0.0.1` allí apunta al propio teléfono.

    Y entre las no locales, **la IPv4 antes que la IPv6**. Desde que se escucha
    también en la `fd7a:…`, la primera de la lista podría ser una IPv6, y
    un enlace con una IPv6 dentro se lee fatal y encima hay que acordarse de los
    corchetes. Si solo hubiera IPv6, se pone con sus corchetes y se manda.
    """
    externos = [host for host in hosts if host not in LOCALES]
    for host in externos:
        if ":" not in host:
            return f"http://{host}:{puerto}"
    if externos:
        return f"http://[{externos[0]}]:{puerto}"
    return f"http://{hosts[0]}:{puerto}"


# --------------------------------------------------------------------------- #
# Base de datos
# --------------------------------------------------------------------------- #

_conexion: sqlite3.Connection | None = None
_cerrojo = threading.RLock()


def _ahora() -> str:
    """Marca de tiempo ISO-8601 en UTC, con sufijo Z.

    Siempre UTC: el sistema acabará repartido entre portátil, torre y Raspberry,
    y mezclar horas locales entre máquinas es una fuente de errores gratuita.
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def abrir(cfg: Configuracion) -> None:
    """Abre la base de datos y crea el esquema si hace falta. Idempotente."""
    global _conexion
    with _cerrojo:
        if _conexion is not None:
            return
        cfg.ruta_db.parent.mkdir(parents=True, exist_ok=True)
        conexion = sqlite3.connect(cfg.ruta_db, check_same_thread=False)
        conexion.row_factory = sqlite3.Row
        # WAL: permite leer mientras se escribe, que es justo lo que hace el
        # panel de la cola mientras un trabajo corre.
        conexion.execute("PRAGMA journal_mode=WAL")
        conexion.executescript(_ESQUEMA)
        _migrar_columnas(conexion)
        conexion.commit()
        _conexion = conexion
        logger.info("Base de datos lista en %s", cfg.ruta_db)


def _migrar_columnas(conexion: sqlite3.Connection) -> None:
    """Añade a `trabajos` las columnas que falten. Idempotente."""
    existentes = {f["name"] for f in conexion.execute("PRAGMA table_info(trabajos)")}
    for columna, tipo in _COLUMNAS_NUEVAS.items():
        if columna in existentes:
            continue
        conexion.execute(f"ALTER TABLE trabajos ADD COLUMN {columna} {tipo}")
        logger.info("Columna %r añadida a la tabla de trabajos.", columna)


def cerrar() -> None:
    global _conexion
    with _cerrojo:
        if _conexion is not None:
            _conexion.close()
            _conexion = None


def _db() -> sqlite3.Connection:
    if _conexion is None:
        raise RuntimeError("La base de datos no está abierta; llama a almacen.abrir() primero.")
    return _conexion


def _a_dict(fila: sqlite3.Row) -> dict[str, Any]:
    """Convierte una fila en diccionario, deshaciendo el JSON de los campos que lo llevan."""
    trabajo = dict(fila)
    for campo in ("peticion", "resultado", "confirmacion"):
        crudo = trabajo.get(campo)
        if crudo is None:
            continue
        try:
            trabajo[campo] = json.loads(crudo)
        except json.JSONDecodeError:
            # Un campo corrupto no debe tumbar el listado entero: se devuelve
            # tal cual y el problema se ve en la interfaz.
            logger.warning("Campo %s del trabajo %s no es JSON válido", campo, trabajo.get("id"))
    return trabajo


# --------------------------------------------------------------------------- #
# Cola de trabajos
# --------------------------------------------------------------------------- #


def encolar(
    agente: str,
    peticion: dict[str, Any],
    origen: str = "texto",
    quien: str | None = None,
) -> dict[str, Any]:
    """Añade un trabajo a la cola y devuelve el trabajo creado.

    `quien` es el perfil de la persona que lo pidió, si el reconocimiento de voz
    lo sabe. Sin él la cola no puede distinguir una orden del dueño de una de
    una visita, que es lo que pasaba hasta el 2026-09-12: `origen` decía «voz» y
    ahí se acababa la información.
    """
    if origen not in ORIGENES:
        raise ValueError(f"Origen desconocido: {origen!r}. Válidos: {ORIGENES}")

    quien = (quien or "").strip() or None
    momento = _ahora()
    with _cerrojo:
        cursor = _db().execute(
            """
            INSERT INTO trabajos (estado, agente, origen, quien, peticion, creado_en, actualizado_en)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                PENDIENTE,
                agente,
                origen,
                quien,
                json.dumps(peticion, ensure_ascii=False),
                momento,
                momento,
            ),
        )
        _db().commit()
        creado = obtener(int(cursor.lastrowid))
    assert creado is not None  # acabamos de insertarlo
    return creado


def reclamar(
    agentes: tuple[str, ...] | None = None, excluir: tuple[str, ...] = ()
) -> dict[str, Any] | None:
    """Toma el trabajo pendiente más antiguo y lo marca en curso.

    `BEGIN IMMEDIATE` toma el cerrojo de escritura antes de leer, así que dos
    trabajadores no pueden reclamar el mismo trabajo. Eso dejó de ser teórico en
    la Fase E: `dev` puede tardar minutos, y si lo atendiera el mismo trabajador
    que todo lo demás, un encargo de código dejaría el correo sin triar y la
    memoria sin responder mientras dura. Por eso hay dos carriles, y por eso este
    método filtra por agente.

    `agentes` limita a esos; `excluir` deja fuera esos. Sin ninguno de los dos se
    comporta exactamente como antes.
    """
    momento = _ahora()
    condiciones = ["estado = ?"]
    parametros: list[Any] = [PENDIENTE]
    if agentes:
        condiciones.append(f"agente IN ({','.join('?' * len(agentes))})")
        parametros.extend(agentes)
    if excluir:
        condiciones.append(f"agente NOT IN ({','.join('?' * len(excluir))})")
        parametros.extend(excluir)
    donde = " AND ".join(condiciones)

    with _cerrojo:
        db = _db()
        db.execute("BEGIN IMMEDIATE")
        try:
            fila = db.execute(
                f"SELECT id FROM trabajos WHERE {donde} ORDER BY id LIMIT 1", parametros
            ).fetchone()
            if fila is None:
                db.execute("ROLLBACK")
                return None

            db.execute(
                """
                UPDATE trabajos
                   SET estado = ?, reclamado_en = ?, actualizado_en = ?, intentos = intentos + 1
                 WHERE id = ?
                """,
                (EN_CURSO, momento, momento, fila["id"]),
            )
            db.execute("COMMIT")
        except Exception:
            db.execute("ROLLBACK")
            raise
        return obtener(int(fila["id"]))


def _cerrar_trabajo(
    id_trabajo: int,
    estado: str,
    resultado: Any = None,
    error: str | None = None,
) -> dict[str, Any] | None:
    momento = _ahora()
    with _cerrojo:
        _db().execute(
            """
            UPDATE trabajos
               SET estado = ?, resultado = ?, error = ?, actualizado_en = ?
             WHERE id = ?
            """,
            (
                estado,
                json.dumps(resultado, ensure_ascii=False) if resultado is not None else None,
                error,
                momento,
                id_trabajo,
            ),
        )
        _db().commit()
    return obtener(id_trabajo)


def completar(id_trabajo: int, resultado: Any) -> dict[str, Any] | None:
    return _cerrar_trabajo(id_trabajo, HECHO, resultado=resultado)


def fallar(id_trabajo: int, error: str) -> dict[str, Any] | None:
    return _cerrar_trabajo(id_trabajo, FALLIDO, error=error)


def cancelar(id_trabajo: int) -> dict[str, Any] | None:
    return _cerrar_trabajo(id_trabajo, CANCELADO)


# --------------------------------------------------------------------------- #
# Confirmación de acciones irreversibles
# --------------------------------------------------------------------------- #
#
# El plan (§7) dice que las acciones irreversibles piden un sí aunque la
# petición venga autenticada. Lo que hace falta para eso es que el trabajo pueda
# **pararse a mitad** y seguir después: si la confirmación viviera en memoria,
# cerrar el núcleo —o la pantalla del móvil— perdería la pregunta.
#
# Por eso la espera es un estado en SQLite y no una promesa en el proceso. Y por
# eso aprobar y rechazar comprueban el estado **dentro** del UPDATE: la respuesta
# puede llegar por la web y por Telegram a la vez, y solo una debe contar.


def pedir_confirmacion(
    id_trabajo: int, resumen: str, detalle: str = ""
) -> dict[str, Any] | None:
    """Deja el trabajo esperando un sí, guardando qué es lo que se pregunta."""
    momento = _ahora()
    confirmacion = {
        "resumen": resumen,
        "detalle": detalle,
        "pedida_en": momento,
        "decision": None,
        "decidida_en": None,
    }
    with _cerrojo:
        _db().execute(
            """
            UPDATE trabajos
               SET estado = ?, confirmacion = ?, reclamado_en = NULL, actualizado_en = ?
             WHERE id = ? AND estado = ?
            """,
            (
                ESPERANDO,
                json.dumps(confirmacion, ensure_ascii=False),
                momento,
                id_trabajo,
                EN_CURSO,
            ),
        )
        _db().commit()
    return obtener(id_trabajo)


def resolver_confirmacion(id_trabajo: int, aprobado: bool) -> dict[str, Any] | None:
    """Contesta a una confirmación pendiente.

    Aprobar devuelve el trabajo a la cola —el agente lo repite, y esta vez ve la
    decisión y sigue adelante—. Rechazar lo cierra.

    Devuelve `None` si el trabajo no estaba esperando: o no existe, o alguien se
    adelantó por el otro canal.
    """
    momento = _ahora()
    with _cerrojo:
        db = _db()
        db.execute("BEGIN IMMEDIATE")
        try:
            fila = db.execute(
                "SELECT confirmacion FROM trabajos WHERE id = ? AND estado = ?",
                (id_trabajo, ESPERANDO),
            ).fetchone()
            if fila is None:
                db.execute("ROLLBACK")
                return None

            try:
                confirmacion = json.loads(fila["confirmacion"] or "{}")
            except json.JSONDecodeError:
                confirmacion = {}
            confirmacion["decision"] = "aprobado" if aprobado else "rechazado"
            confirmacion["decidida_en"] = momento

            db.execute(
                """
                UPDATE trabajos
                   SET estado = ?, confirmacion = ?, actualizado_en = ?
                 WHERE id = ? AND estado = ?
                """,
                (
                    PENDIENTE if aprobado else RECHAZADO,
                    json.dumps(confirmacion, ensure_ascii=False),
                    momento,
                    id_trabajo,
                    ESPERANDO,
                ),
            )
            db.execute("COMMIT")
        except Exception:
            db.execute("ROLLBACK")
            raise
    return obtener(id_trabajo)


def obtener(id_trabajo: int) -> dict[str, Any] | None:
    with _cerrojo:
        fila = _db().execute("SELECT * FROM trabajos WHERE id = ?", (id_trabajo,)).fetchone()
    return _a_dict(fila) if fila else None


def listar(estado: str | None = None, limite: int = 50) -> list[dict[str, Any]]:
    limite = max(1, min(limite, 500))
    with _cerrojo:
        if estado:
            filas = _db().execute(
                "SELECT * FROM trabajos WHERE estado = ? ORDER BY id DESC LIMIT ?",
                (estado, limite),
            ).fetchall()
        else:
            filas = _db().execute(
                "SELECT * FROM trabajos ORDER BY id DESC LIMIT ?", (limite,)
            ).fetchall()
    return [_a_dict(f) for f in filas]


def recuento_por_estado() -> dict[str, int]:
    with _cerrojo:
        filas = _db().execute(
            "SELECT estado, COUNT(*) AS n FROM trabajos GROUP BY estado"
        ).fetchall()
    return {f["estado"]: f["n"] for f in filas}


def recuperar_huerfanos() -> int:
    """Devuelve a la cola los trabajos que estaban en curso al morir el proceso.

    Sin esto, un cierre inesperado deja trabajos clavados en `en_curso` para
    siempre: nadie los ejecuta y nadie los va a reclamar. Se llama al arrancar,
    cuando por definición no hay ningún trabajador vivo.
    """
    with _cerrojo:
        cursor = _db().execute(
            """
            UPDATE trabajos
               SET estado = ?, reclamado_en = NULL, actualizado_en = ?
             WHERE estado = ?
            """,
            (PENDIENTE, _ahora(), EN_CURSO),
        )
        _db().commit()
        recuperados = cursor.rowcount

    if recuperados:
        logger.warning("%d trabajo(s) huérfano(s) devueltos a la cola.", recuperados)
    return recuperados


# --------------------------------------------------------------------------- #
# Correos triados: qué se ha hecho con cada uno
# --------------------------------------------------------------------------- #


def marcar_correo(id_mensaje: str, estado: str) -> dict[str, str]:
    """Deja escrito qué se ha hecho con un correo. Marcar dos veces no duplica.

    `pendiente` borra la fila en vez de guardarla: no estar en la tabla es lo
    que significa estar pendiente, y con dos formas de decir lo mismo la que
    nadie mire acaba mintiendo.
    """
    id_mensaje = id_mensaje.strip()
    if not id_mensaje:
        raise ValueError("Un correo sin id no se puede marcar.")
    if estado not in ESTADOS_CORREO:
        raise ValueError(f"Estado de correo desconocido: {estado!r}")

    ahora = _ahora()
    with _cerrojo:
        if estado == PENDIENTE_CORREO:
            _db().execute("DELETE FROM correos WHERE id_mensaje = ?", (id_mensaje,))
        else:
            _db().execute(
                """
                INSERT INTO correos (id_mensaje, estado, actualizado_en) VALUES (?, ?, ?)
                ON CONFLICT (id_mensaje)
                DO UPDATE SET estado = excluded.estado, actualizado_en = excluded.actualizado_en
                """,
                (id_mensaje, estado, ahora),
            )
        _db().commit()
    return {"id": id_mensaje, "estado": estado, "actualizado_en": ahora}


def correos_marcados() -> dict[str, str]:
    """Los correos que alguien ha tocado, por id. El resto están pendientes.

    Se devuelve entero y no por lotes: son las decisiones de un buzón personal,
    no un histórico. Si algún día pesa, se corta por fecha.
    """
    with _cerrojo:
        filas = _db().execute("SELECT id_mensaje, estado FROM correos").fetchall()
    return {f["id_mensaje"]: f["estado"] for f in filas}


# --------------------------------------------------------------------------- #
# Cuota de los servicios de fuera
# --------------------------------------------------------------------------- #


def dia_de_cuota() -> str:
    """El día al que se le apunta el gasto, en UTC.

    **No coincide con el día de Google**, que reinicia las cuotas gratuitas a
    medianoche del Pacífico. Se usa UTC igual que en el resto del almacén porque
    la alternativa —cargar una zona horaria— arrastra `tzdata` en Windows para
    ganar unas horas de precisión en un número que ya es aproximado: aquí solo se
    ve lo que gasta este proceso, y la app de voz gasta por su cuenta.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def apuntar_uso(servicio: str, cantidad: int = 1) -> None:
    """Suma llamadas al contador de hoy. **Nunca lanza.**

    Llevar la cuenta no puede tumbar una clasificación de correo: si la base de
    datos no está abierta —el caso de una prueba unitaria— o el INSERT falla, se
    pierde el recuento y se sigue. Es la única parte del sistema donde tragarse
    un error es lo correcto, porque el dato es informativo y lo que protege es el
    trabajo de verdad.
    """
    try:
        with _cerrojo:
            _db().execute(
                """
                INSERT INTO uso (dia, servicio, contador) VALUES (?, ?, ?)
                ON CONFLICT (dia, servicio)
                DO UPDATE SET contador = contador + excluded.contador
                """,
                (dia_de_cuota(), servicio, cantidad),
            )
            _db().commit()
    except (sqlite3.Error, RuntimeError) as e:
        logger.debug("No se pudo apuntar el uso de %r (%s).", servicio, e)


def uso_de_hoy() -> dict[str, int]:
    """Cuántas llamadas lleva hoy cada servicio de fuera."""
    with _cerrojo:
        filas = _db().execute(
            "SELECT servicio, contador FROM uso WHERE dia = ?", (dia_de_cuota(),)
        ).fetchall()
    return {f["servicio"]: int(f["contador"]) for f in filas}


# --------------------------------------------------------------------------- #
# El chat escrito: sesiones y mensajes
# --------------------------------------------------------------------------- #


def crear_sesion_chat(titulo: str = "") -> dict[str, Any]:
    """Una conversación nueva. El título lo pone el primer mensaje si no viene."""
    ahora = _ahora()
    with _cerrojo:
        cursor = _db().execute(
            "INSERT INTO chat_sesiones (titulo, turno, creado_en, actualizado_en) "
            "VALUES (?, 'libre', ?, ?)",
            (titulo.strip(), ahora, ahora),
        )
        _db().commit()
        id_sesion = int(cursor.lastrowid)
    return {"id": id_sesion, "titulo": titulo.strip(), "turno": "libre", "creado_en": ahora}


def sesiones_chat() -> list[dict[str, Any]]:
    """Las conversaciones, la más reciente primero."""
    with _cerrojo:
        filas = _db().execute(
            "SELECT id, titulo, turno, creado_en, actualizado_en FROM chat_sesiones "
            "ORDER BY actualizado_en DESC"
        ).fetchall()
    return [dict(f) for f in filas]


def obtener_sesion_chat(id_sesion: int) -> dict[str, Any] | None:
    with _cerrojo:
        fila = _db().execute(
            "SELECT id, titulo, turno, creado_en, actualizado_en FROM chat_sesiones WHERE id = ?",
            (id_sesion,),
        ).fetchone()
    return dict(fila) if fila else None


def borrar_sesion_chat(id_sesion: int) -> bool:
    """Borra la sesión y sus mensajes. Solo si está libre: no se corta un turno."""
    with _cerrojo:
        fila = _db().execute(
            "SELECT turno FROM chat_sesiones WHERE id = ?", (id_sesion,)
        ).fetchone()
        if fila is None:
            return False
        if fila["turno"] == "ocupado":
            raise ValueError("La conversación está ocupada; espera a que termine el turno.")
        _db().execute("DELETE FROM chat_mensajes WHERE sesion = ?", (id_sesion,))
        _db().execute("DELETE FROM chat_sesiones WHERE id = ?", (id_sesion,))
        _db().commit()
    return True


def anadir_mensaje_chat(sesion: int, rol: str, texto: str, estado: str = "hecho") -> int:
    """Un mensaje en una conversación. `escribiendo` es el estado inicial del
    mensaje de Perseo: el texto llega por trozos y la cara lo lee sondeando.

    El primer mensaje de usuario nombra la conversación — ahí y no al cerrar el
    turno, porque un turno puede fallar y el título no depende de él.
    """
    ahora = _ahora()
    with _cerrojo:
        cursor = _db().execute(
            "INSERT INTO chat_mensajes (sesion, rol, texto, herramientas, estado, momento) "
            "VALUES (?, ?, ?, '[]', ?, ?)",
            (sesion, rol, texto, estado, ahora),
        )
        if rol == "usuario":
            recortado = " ".join(texto.split())[:60]
            _db().execute(
                "UPDATE chat_sesiones SET titulo = ? "
                "WHERE id = ? AND (titulo = '' OR titulo IS NULL)",
                (recortado, sesion),
            )
        _db().execute(
            "UPDATE chat_sesiones SET actualizado_en = ? WHERE id = ?", (ahora, sesion)
        )
        _db().commit()
        return int(cursor.lastrowid)


def actualizar_mensaje_chat(
    id_mensaje: int,
    texto: str | None = None,
    estado: str | None = None,
    herramientas: list[str] | None = None,
) -> None:
    """El avance de un mensaje de Perseo. Cada campo es opcional a propósito:
    el streaming solo toca `texto`, y las herramientas llegan al final."""
    cambios: list[tuple[Any, str]] = []
    if texto is not None:
        cambios.append((texto, "texto"))
    if estado is not None:
        cambios.append((estado, "estado"))
    if herramientas is not None:
        cambios.append((json.dumps(herramientas, ensure_ascii=False), "herramientas"))
    if not cambios:
        return
    with _cerrojo:
        _db().execute(
            f"UPDATE chat_mensajes SET {', '.join(f'{c} = ?' for _, c in cambios)} WHERE id = ?",
            (*[v for v, _ in cambios], id_mensaje),
        )
        _db().commit()


def mensajes_chat(id_sesion: int, tope: int = 200) -> list[dict[str, Any]]:
    """Los mensajes de una conversación, los últimos `tope`.

    Las herramientas viajan decodificadas —la cara no debería tener que saber
    que en disco es JSON— y los mensajes sin texto aún (`escribiendo` recién
    nacido) también salen: es la burbuja vacía que enseña «está en ello».
    """
    with _cerrojo:
        filas = _db().execute(
            "SELECT id, rol, texto, herramientas, estado, momento FROM chat_mensajes "
            "WHERE sesion = ? ORDER BY id DESC LIMIT ?",
            (id_sesion, tope),
        ).fetchall()
    salida = []
    for f in reversed(filas):
        try:
            herramientas = json.loads(f["herramientas"] or "[]")
        except json.JSONDecodeError:
            herramientas = []
        salida.append({**dict(f), "herramientas": herramientas})
    return salida


def marcar_turno_chat(id_sesion: int, turno: str) -> None:
    """Abre o cierra el semáforo de una conversación.

    Ocupar lo que ya está ocupado levanta `ValueError` — es exactamente el 409
    que contesta la API cuando otra pantalla está a mitad de turno. Liberar,
    en cambio, es idempotente: dos caminos soltando el mismo semáforo no se
    estorban.
    """
    with _cerrojo:
        cursor = _db().execute(
            "UPDATE chat_sesiones SET turno = ?, actualizado_en = ? WHERE id = ? AND turno != ?",
            (turno, _ahora(), id_sesion, turno),
        )
        _db().commit()
        if cursor.rowcount:
            return
        fila = _db().execute(
            "SELECT turno FROM chat_sesiones WHERE id = ?", (id_sesion,)
        ).fetchone()
        if fila is None:
            raise ValueError("No existe esa conversación.")
        if turno == "ocupado":
            raise ValueError(f"La conversación está {fila['turno']}.")


def reiniciar_turnos_chat() -> None:
    """Al arrancar, ninguna conversación está a mitad de turno.

    El semáforo vive en la base para que dos pantallas se respeten, pero un
    apagón lo dejaría en `ocupado` para siempre. Al abrir, todo libre: si había
    un turno en marcha, su trabajo vuelve a la cola por `recuperar_huerfanos`
    y el mensaje a medias se retoma cuando el trabajo se reclame otra vez.
    """
    with _cerrojo:
        _db().execute("UPDATE chat_sesiones SET turno = 'libre' WHERE turno != 'libre'")
        _db().commit()
