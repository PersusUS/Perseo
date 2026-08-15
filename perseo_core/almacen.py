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

Ver bitacora/05_PLAN_PERSEO_V2.md §2 y §9 (Fase A).
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RAIZ = Path(__file__).resolve().parent

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

_ESQUEMA = """
CREATE TABLE IF NOT EXISTS trabajos (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    estado          TEXT    NOT NULL,
    agente          TEXT    NOT NULL,
    origen          TEXT    NOT NULL,
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
"""

#: Columnas añadidas después de que hubiera bases de datos por ahí. `CREATE
#: TABLE IF NOT EXISTS` no las añade a una tabla que ya existe, así que hay que
#: mirarlo a mano al abrir.
_COLUMNAS_NUEVAS = {"confirmacion": "TEXT"}


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

    @property
    def telegram_configurado(self) -> bool:
        return bool(self.telegram_token and self.telegram_chat)


#: Rango que Tailscale reparte entre los nodos del tailnet (CGNAT).
_RED_TAILSCALE = ipaddress.ip_network("100.64.0.0/10")

#: Sitios donde suele estar la herramienta de Tailscale en Windows y en Linux.
_RUTAS_TAILSCALE = (
    Path(r"C:\Program Files\Tailscale\tailscale.exe"),
    Path("/usr/bin/tailscale"),
    Path("/usr/local/bin/tailscale"),
)

LOCALES = ("127.0.0.1", "localhost", "::1")


def direccion_tailscale() -> str | None:
    """Devuelve la dirección de esta máquina dentro del tailnet, o `None`.

    Se pregunta primero a la propia herramienta de Tailscale, que es la única
    fuente que no se puede confundir. Si no está instalada se recorren las
    interfaces buscando una del rango CGNAT — funciona, pero conviene saber que
    ese rango también lo usan algunos operadores en la interfaz de salida, así
    que es el segundo intento y no el primero.
    """
    for ruta in _RUTAS_TAILSCALE:
        if not ruta.exists():
            continue
        try:
            salida = subprocess.run(
                [str(ruta), "ip", "-4"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            continue
        for linea in salida.splitlines():
            candidata = linea.strip()
            if candidata:
                return candidata

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

        direccion = direccion_tailscale()
        if direccion is None:
            logger.error(
                "Se pidió escuchar en Tailscale pero no se encontró la dirección del "
                "tailnet. ¿Está Tailscale conectado? Se sigue solo en local."
            )
            continue
        # El bucle local va siempre con Tailscale: si no, la app del PC y las
        # pruebas dejarían de poder hablar con el núcleo.
        resueltos.append("127.0.0.1")
        resueltos.append(direccion)

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


def cargar_configuracion() -> Configuracion:
    directorio = _directorio_datos()

    # Por defecto solo el bucle local. Para que entre el móvil se pone
    # `PERSEO_CORE_HOST=tailscale`, que añade la dirección del tailnet **sin**
    # quitar la local y sin pasar por `0.0.0.0`.
    hosts = _resolver_hosts(os.environ.get("PERSEO_CORE_HOST", "127.0.0.1"))
    puerto = int(os.environ.get("PERSEO_CORE_PUERTO", "8787"))

    return Configuracion(
        hosts=hosts,
        puerto=puerto,
        token=_resolver_token(directorio),
        ruta_db=Path(os.environ.get("PERSEO_CORE_DB", directorio / "estado.sqlite3")),
        directorio_datos=directorio,
        url_ollama=os.environ.get("PERSEO_OLLAMA", "http://127.0.0.1:11434"),
        modelo_router=os.environ.get("PERSEO_MODELO_ROUTER", "qwen3:4b"),
        telegram_token=_de_entorno_o_fichero("PERSEO_TELEGRAM_TOKEN", directorio / "telegram.txt"),
        telegram_chat=_de_entorno_o_fichero("PERSEO_TELEGRAM_CHAT", directorio / "telegram_chat.txt"),
        telegram_api=os.environ.get("PERSEO_TELEGRAM_API", "https://api.telegram.org").rstrip("/"),
        url_base=os.environ.get("PERSEO_URL_BASE", "").strip() or _url_por_defecto(hosts, puerto),
    )


def _de_entorno_o_fichero(variable: str, fichero: Path) -> str:
    """Lee un secreto de la variable de entorno o, si no está, de un fichero.

    El fichero vive en el directorio de datos, que está fuera de git. Es más
    cómodo que exportar la variable en cada arranque, y no deja el token del bot
    en el historial del terminal.
    """
    del_entorno = os.environ.get(variable, "").strip()
    if del_entorno:
        return del_entorno
    if fichero.exists():
        return fichero.read_text(encoding="utf-8").strip()
    return ""


def _url_por_defecto(hosts: tuple[str, ...], puerto: int) -> str:
    """Dirección para los enlaces que salen de la máquina.

    Se prefiere una interfaz no local: el enlace lo abre el móvil desde el
    tailnet, y `127.0.0.1` allí apunta al propio teléfono.
    """
    for host in hosts:
        if host not in LOCALES:
            return f"http://{host}:{puerto}"
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


def encolar(agente: str, peticion: dict[str, Any], origen: str = "texto") -> dict[str, Any]:
    """Añade un trabajo a la cola y devuelve el trabajo creado."""
    if origen not in ORIGENES:
        raise ValueError(f"Origen desconocido: {origen!r}. Válidos: {ORIGENES}")

    momento = _ahora()
    with _cerrojo:
        cursor = _db().execute(
            """
            INSERT INTO trabajos (estado, agente, origen, peticion, creado_en, actualizado_en)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (PENDIENTE, agente, origen, json.dumps(peticion, ensure_ascii=False), momento, momento),
        )
        _db().commit()
        creado = obtener(int(cursor.lastrowid))
    assert creado is not None  # acabamos de insertarlo
    return creado


def reclamar() -> dict[str, Any] | None:
    """Toma el trabajo pendiente más antiguo y lo marca en curso.

    `BEGIN IMMEDIATE` toma el cerrojo de escritura antes de leer, así que dos
    trabajadores no pueden reclamar el mismo trabajo. Hoy solo hay uno, pero la
    Fase D añadirá disparadores que encolan en paralelo, y no quiero descubrir
    esto entonces.
    """
    momento = _ahora()
    with _cerrojo:
        db = _db()
        db.execute("BEGIN IMMEDIATE")
        try:
            fila = db.execute(
                "SELECT id FROM trabajos WHERE estado = ? ORDER BY id LIMIT 1", (PENDIENTE,)
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


def rechazar(id_trabajo: int) -> dict[str, Any] | None:
    return _cerrar_trabajo(id_trabajo, RECHAZADO)


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
