"""Qué está en pie ahora mismo, y qué no.

`/salud` contesta si el núcleo respira; esto contesta algo distinto: **de qué
está capado hoy**. La diferencia importa porque casi todo lo que sostiene a
Perseo puede faltar sin que nada falle a gritos —Ollama apagado, Obsidian
cerrado, un `refresh_token` revocado, la cuota diaria agotada— y el sistema
sigue arrancando, contestando y sin quejarse. Eso es un acierto de diseño y a la
vez la razón de que haga falta una pantalla: lo que se rompe en silencio hay que
poder mirarlo.

Tres reglas de esta pieza:

1. **Ningún sondeo puede tumbar la respuesta.** Cada comprobación va en su
   `try`, con su tope de tiempo, y lo que salga mal se convierte en una pieza en
   rojo con el motivo dentro. Un panel de estado que devuelve 500 no informa de
   nada.
2. **Se recuerda lo que se acaba de preguntar.** La pantalla se mira desde el
   móvil y se refresca sola; sin memoria, cada vistazo dispararía una vuelta a
   Ollama, otra al plugin de Obsidian y un testigo nuevo de Google. Los sondeos
   caros duran más en memoria que los baratos.
3. **No se sondea lo que no está pedido.** Si no hay Gmail configurado, la pieza
   de Google dice "apagado" sin tocar la red. "Apagado" no es un fallo: es una
   capacidad que hoy no está, y la pantalla lo distingue del rojo a propósito.

Sobre la cuota: **Google no publica ningún sitio donde consultar lo que queda**
del plan gratuito. Así que no se consulta — se cuenta lo que gasta este proceso
(ver `almacen.apuntar_uso`) y se compara con los topes leídos a mano en AI Studio
el 2026-08-16. Es un número aproximado por debajo, y la pantalla lo dice: la app
de voz gasta por su cuenta y no pasa por aquí.

Ver bitacora/06_HANDOFF.md §5 y §10, y bitacora/07_PWA.md.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

import aiohttp

from . import agenda, almacen, politica
from .agentes import REGISTRO, Router
from .disparadores import REGISTRO as DISPARADORES

logger = logging.getLogger(__name__)

#: Los cuatro colores de la pantalla. `apagado` no es un fallo —es algo sin
#: configurar— y por eso no comparte casilla con `malo`: confundirlos haría que
#: el panel estuviera siempre en rojo y dejara de mirarse.
OK = "ok"
AVISO = "aviso"
MALO = "malo"
APAGADO = "apagado"

#: Lo que se espera como mucho a cada sondeo. Corto a propósito: esta pantalla se
#: abre desde el móvil por el tailnet, y una comprobación que tarda es peor que
#: una que dice que no.
TOPE_SONDEO = 4

#: La raíz del repositorio, para mirar el disco donde vive Perseo.
RAIZ = Path(__file__).resolve().parent.parent

#: Cuánto se recuerda cada sondeo. Ollama y Obsidian son locales y baratos;
#: Google pide un testigo nuevo a un servidor de fuera, así que se pregunta una
#: vez cada cinco minutos por muchas veces que se mire la pantalla.
MEMORIA_LOCAL = 10
MEMORIA_GOOGLE = 300

#: Peticiones al día del plan gratuito, leídas en aistudio.google.com/rate-limit
#: —comprobadas otra vez el 2026-08-24, tabla entera delante—. **No salen de la
#: documentación**: Google dejó de publicarlas, y la que había escrita estaba
#: desfasada por un factor de doce. Lo que enseñó la tabla de agosto: los Flash
#: nuevos (3.6, 3.7) dan las mismas 20 al día que el 2.5, así que estrenarlos no
#: compra nada; los `flash-lite` dan 500 **cada uno**, y la API en vivo no tiene
#: tope diario. Se comparan por
#: trozo del nombre porque la familia manda sobre la versión: cualquier Gemma
#: tiene el tope de Gemma. El orden importa —`flash-lite` antes que `flash`— y
#: lo que no encaje se queda sin tope, que se pinta como "sin tope conocido" en
#: vez de inventarse uno.
TOPES_DIARIOS: tuple[tuple[str, int], ...] = (
    ("gemma", 14400),
    ("flash-lite", 500),
    ("flash", 20),
)

#: Cuándo arrancó este proceso. Se fija al importar el módulo, que en el arranque
#: normal ocurre a los pocos milisegundos de empezar.
_ARRANQUE = time.monotonic()
_ARRANQUE_RELOJ = datetime.now(timezone.utc)


@dataclass(frozen=True)
class Pieza:
    """Una fila del panel: qué es, cómo está y qué hacer si está mal."""

    id: str
    nombre: str
    estado: str
    detalle: str
    #: El comando o el gesto que lo arregla. Vacío cuando no hay nada que hacer.
    #: Es lo que separa un panel útil de una lista de luces rojas.
    arreglo: str = ""


# --------------------------------------------------------------------------- #
# Memoria de sondeos
# --------------------------------------------------------------------------- #

_recordado: dict[str, tuple[float, Pieza]] = {}


def olvidar() -> None:
    """Tira lo recordado. Para las pruebas y para el cierre del núcleo."""
    _recordado.clear()


async def _recordando(clave: str, ttl: float, hacer: Callable[[], Awaitable[Pieza]]) -> Pieza:
    guardado = _recordado.get(clave)
    ahora = time.monotonic()
    if guardado is not None and ahora - guardado[0] < ttl:
        return guardado[1]
    pieza = await hacer()
    _recordado[clave] = (ahora, pieza)
    return pieza


async def _sin_caerse(id: str, nombre: str, hacer: Callable[[], Awaitable[Pieza]]) -> Pieza:
    """Convierte cualquier fallo inesperado en una pieza roja, nunca en un 500."""
    try:
        return await hacer()
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 - a propósito: ver la regla 1
        logger.exception("Fallo comprobando %s.", id)
        return Pieza(id, nombre, MALO, f"No se pudo comprobar: {e}")


# --------------------------------------------------------------------------- #
# Los sondeos
# --------------------------------------------------------------------------- #


async def _ollama(cfg: almacen.Configuracion, http: aiohttp.ClientSession) -> Pieza:
    """El modelo de casa. Es el que sostiene el triaje diario (§5 del handoff)."""
    try:
        async with http.get(
            f"{cfg.url_ollama}/api/tags", timeout=aiohttp.ClientTimeout(total=TOPE_SONDEO)
        ) as respuesta:
            if respuesta.status != 200:
                return Pieza(
                    "ollama",
                    "Modelo local",
                    MALO,
                    f"Ollama respondió {respuesta.status} en {cfg.url_ollama}.",
                )
            datos = await respuesta.json()
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
        # El caso normal en un portátil: no está arrancado. Es rojo igualmente,
        # porque sin él el correo del día no se clasifica.
        return Pieza(
            "ollama",
            "Modelo local",
            MALO,
            f"Ollama no responde en {cfg.url_ollama} ({e}).",
            "Arráncalo: `ollama serve`. Sin él no hay triaje de correo.",
        )

    nombres = [str(m.get("name", "")) for m in (datos.get("models") or [])]
    familia = cfg.modelo_router.split(":")[0]
    if cfg.modelo_router not in nombres and not any(n.split(":")[0] == familia for n in nombres):
        return Pieza(
            "ollama",
            "Modelo local",
            AVISO,
            f"Ollama está en pie pero no tiene {cfg.modelo_router}.",
            f"ollama pull {cfg.modelo_router}",
        )
    return Pieza(
        "ollama",
        "Modelo local",
        OK,
        f"{cfg.modelo_router} listo · {len(nombres)} modelo(s) en Ollama.",
    )


def _suplente(cfg: almacen.Configuracion) -> Pieza:
    """El de fuera, que solo entra si el de casa no está."""
    if not cfg.modelo_suplente:
        return Pieza(
            "suplente",
            "Modelo suplente",
            APAGADO,
            "Apagado: si Ollama no está, no se clasifica nada.",
            # Gemma daba 14.400 al día y era la recomendación de aquí; se probó
            # el 2026-08-24 y contesta con su razonamiento en voz alta en vez de
            # con el JSON que se le pide — el mismo fallo que ya salía en el
            # registro («El suplente devolvió algo que no es JSON»). El que sí
            # obedece es un flash-lite, aunque su cuota sea más corta.
            "PERSEO_MODELO_SUPLENTE=gemini-3.1-flash-lite. Es lo único que manda "
            "a un tercero el texto que se clasifica.",
        )
    if not cfg.gemini_clave:
        return Pieza(
            "suplente",
            "Modelo suplente",
            AVISO,
            f"{cfg.modelo_suplente} pedido, pero sin clave: no contestaría.",
            "GEMINI_API_KEY, o <datos>/gemini.txt.",
        )
    return Pieza(
        "suplente",
        "Modelo suplente",
        OK,
        f"{cfg.modelo_suplente}, para cuando Ollama no esté.",
    )


async def _vault(cfg: almacen.Configuracion) -> Pieza:
    """La memoria. Dos respaldos, y solo uno depende de que algo esté abierto."""
    from . import memoria

    if cfg.vault_respaldo != "rest":
        return Pieza("vault", "Memoria", OK, f"En ficheros: {memoria.ruta_vault(cfg)}")

    if not cfg.vault_rest_clave:
        return Pieza(
            "vault",
            "Memoria",
            AVISO,
            "Pedido el plugin de Obsidian, pero sin clave: se sigue escribiendo en ficheros.",
            "La clave sale en los ajustes del plugin; va en <datos>/obsidian.txt.",
        )

    # Si el núcleo ya tiene una sesión abierta contra el plugin, se usa esa. Una
    # nueva por sondeo dejaría un `ClientSession` sin cerrar cada diez segundos.
    abierto = memoria.respaldo()
    rest = abierto if isinstance(abierto, memoria.VaultRest) else None
    propio = rest is None
    if rest is None:
        rest = memoria.VaultRest(cfg.vault_rest_url, cfg.vault_rest_clave)

    try:
        await rest.comprobar()
    except (RuntimeError, asyncio.TimeoutError) as e:
        return Pieza(
            "vault",
            "Memoria",
            MALO,
            str(e),
            "Abre Obsidian: el servidor es suyo, y sin él la memoria falla.",
        )
    finally:
        if propio:
            await rest.cerrar()

    return Pieza("vault", "Memoria", OK, f"Plugin de Obsidian en {cfg.vault_rest_url}.")


async def _google(cfg: almacen.Configuracion) -> Pieza:
    """Gmail y Calendar. Se sondea pidiendo un testigo, que es lo que caduca."""
    from . import google_api

    pedidos = [
        nombre
        for nombre, pedido in (
            ("Gmail", cfg.correo_buzon == "gmail"),
            ("Calendar", cfg.agenda_origen == "google"),
        )
        if pedido
    ]
    if not pedidos:
        return Pieza(
            "google",
            "Google",
            APAGADO,
            "Ni Gmail ni Calendar están pedidos.",
            "PERSEO_CORREO=gmail PERSEO_AGENDA=google",
        )

    try:
        await asyncio.wait_for(google_api.comprobar(cfg), timeout=TOPE_SONDEO * 3)
    except google_api.SinCredenciales as e:
        return Pieza(
            "google", "Google", MALO, str(e), "python -m perseo_core.autorizar_google"
        )
    except (RuntimeError, aiohttp.ClientError, asyncio.TimeoutError) as e:
        return Pieza("google", "Google", MALO, f"Google no contesta ({e}).")

    return Pieza("google", "Google", OK, f"Credenciales buenas · {' y '.join(pedidos)}.")


def _telegram(cfg: almacen.Configuracion) -> Pieza:
    if not cfg.telegram_configurado:
        return Pieza(
            "telegram",
            "Telegram",
            APAGADO,
            "Sin bot: no sale ningún aviso al móvil.",
            "python -m perseo_core.telegram, con el núcleo parado.",
        )
    if not cfg.url_base_alcanzable:
        # Pasa siempre que se arranca sin Tailscale, y el síntoma —un enlace que
        # abre una página en blanco en el iPhone— no se parece a la causa.
        return Pieza(
            "telegram",
            "Telegram",
            AVISO,
            f"El bot avisa, pero el enlace «ver detalle» apunta a {cfg.url_base}: "
            "en el móvil eso es el móvil.",
            "Arranca con PERSEO_CORE_HOST=tailscale.",
        )
    return Pieza("telegram", "Telegram", OK, f"Avisos al móvil desde {cfg.url_base}. Solo avisa: la decisión se da por voz o en las pantallas.")


def _mcp(cfg: almacen.Configuracion) -> Pieza:
    from . import mcp as modulo_mcp

    if not modulo_mcp.definiciones:
        return Pieza("mcp", "MCP", APAGADO, "Sin servidores configurados.", f"Crea {cfg.directorio_datos / 'mcp.json'}.")
    nombres = ", ".join(sorted(modulo_mcp.definiciones))
    return Pieza("mcp", "MCP", OK, f"{len(modulo_mcp.definiciones)} servidor(es): {nombres}.")


def _correo(cfg: almacen.Configuracion) -> Pieza:
    if cfg.correo_buzon == "gmail":
        return Pieza("correo", "Correo", OK, f"Gmail, cada {cfg.intervalos.get('correo', 300):.0f} s.")
    if cfg.correo_buzon == "falso":
        return Pieza("correo", "Correo", AVISO, f"Buzón de mentira: {cfg.correo_falso}")
    return Pieza("correo", "Correo", APAGADO, "Sin buzón.", "PERSEO_CORREO=gmail")


def _agenda(cfg: almacen.Configuracion) -> Pieza:
    if cfg.agenda_origen == "google":
        return Pieza(
            "agenda",
            "Agenda",
            OK,
            f"Google Calendar, avisando {cfg.agenda_antelacion} min antes.",
        )
    if cfg.agenda_origen == "falso":
        return Pieza("agenda", "Agenda", AVISO, f"Calendario de mentira: {cfg.agenda_falsa}")
    return Pieza("agenda", "Agenda", APAGADO, "Sin calendario.", "PERSEO_AGENDA=google")


def _dev(cfg: almacen.Configuracion) -> Pieza:
    if cfg.dev_motor == "falso":
        return Pieza(
            "dev",
            "Encargos de código",
            AVISO,
            "Motor simulado: los encargos no se ejecutan de verdad.",
            "Quita PERSEO_DEV_MOTOR.",
        )
    return Pieza("dev", "Encargos de código", OK, f"{cfg.dev_ejecutable} sobre {cfg.dev_raiz}")


def _web(cfg: almacen.Configuracion) -> Pieza:
    if cfg.web_navegador == "falso":
        return Pieza("web", "Navegador", AVISO, "Navegador simulado: no se lee ninguna página.")
    return Pieza("web", "Navegador", OK, "HTTP de verdad, sin alcanzar la red de casa.")


def _chat(cfg: almacen.Configuracion) -> Pieza:
    """El chat escrito vive de la misma clave que el suplente: sin ella no
    hay cabeza para los turnos, y es un «apagado» y no un rojo a propósito."""
    from . import chat as modulo_chat

    if not cfg.gemini_clave:
        return Pieza(
            "chat",
            "Chat escrito",
            APAGADO,
            f"Sin clave de Gemini no hay turnos ({modulo_chat.MODELO_POR_DEFECTO} es quien piensa).",
            "GEMINI_API_KEY, o <datos>/gemini.txt.",
        )
    modelos = modulo_chat._modelos()
    # El de reserva no es un adorno: cada modelo tiene su propio cubo de cuota
    # diaria, así que decir cuál hay detrás es decir cuánto aguanta el chat.
    detalle = f"{modelos[0]}, con herramientas."
    if len(modelos) > 1:
        detalle += f" De reserva, {', '.join(modelos[1:])} — cada uno con su cuota."
    return Pieza("chat", "Chat escrito", OK, detalle)


def _confianza() -> Pieza:
    hasta = politica.confianza_hasta()
    if hasta is None:
        return Pieza("confianza", "Confirmaciones", OK, "Lo irreversible pide un sí.")
    return Pieza(
        "confianza",
        "Confirmaciones",
        AVISO,
        f"Modo confianza encendido hasta las {hasta.astimezone().strftime('%H:%M')}: "
        "lo irreversible se ejecuta sin preguntar.",
    )


# --------------------------------------------------------------------------- #
# Cuota
# --------------------------------------------------------------------------- #


def tope_diario(modelo: str) -> int | None:
    """Peticiones al día que admite ese modelo, o `None` si no se sabe."""
    nombre = modelo.lower()
    for trozo, tope in TOPES_DIARIOS:
        if trozo in nombre:
            return tope
    return None


def _cuota(cfg: almacen.Configuracion, usos: dict[str, int]) -> dict[str, Any]:
    """Lo gastado hoy, por modelo. Cuenta por debajo, y lo dice."""
    modelos = sorted(set(usos) | ({cfg.modelo_suplente} if cfg.modelo_suplente else set()))
    return {
        "dia": almacen.dia_de_cuota(),
        "nota": (
            "Solo cuenta lo que gasta el núcleo; la app de voz no pasa por aquí. "
            "El día se cuenta en UTC y Google reinicia a medianoche del Pacífico."
        ),
        "servicios": [
            {"modelo": m, "usadas": usos.get(m, 0), "tope": tope_diario(m)} for m in modelos
        ],
    }


# --------------------------------------------------------------------------- #
# El panel entero
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Telemetría de la máquina
# --------------------------------------------------------------------------- #


def _bytes_legibles(n: float) -> str:
    for unidad in ("B", "kB", "MB", "GB", "TB"):
        if n < 1024 or unidad == "TB":
            return f"{n:.0f} {unidad}" if unidad == "B" else f"{n:.1f} {unidad}"
        n /= 1024
    return f"{n:.1f} TB"


#: Últimos contadores de red, para poder dar velocidad y no un total desde el
#: arranque de Windows —que es un número enorme que no dice nada—.
_ultima_red: tuple[float, int, int] | None = None

#: Las últimas muestras de CPU y memoria, para dibujar una línea en vez de un
#: número. Vive en memoria y se pierde al reiniciar el núcleo, y está bien: esto
#: es para mirar de reojo si algo se está calentando ahora, no una serie
#: histórica. Guardarla en la base de datos sería otra tabla que crece sola.
_HISTORIAL: deque[dict[str, float]] = deque(maxlen=90)

#: Cuándo se apuntó la última, sin redondear. Ver `_apuntar_muestra`.
_ultima_muestra: float | None = None

#: Cada cuánto se apunta una muestra. `telemetria()` se llama en cada vistazo a
#: la pantalla, y con dos clientes mirando eso son dos muestras por refresco: sin
#: este suelo, el historial se llenaría de puntos del mismo instante.
SEGUNDOS_ENTRE_MUESTRAS = 8.0

#: Cuántas citas del día caben en la pantalla antes de que deje de leerse.
EVENTOS_EN_PANTALLA = 4


def telemetria() -> dict[str, Any]:
    """CPU, memoria, disco y red de esta máquina. **Nunca lanza.**

    Es la única pieza del estado que habla de la máquina y no de Perseo, y va
    aparte por eso: si un día el núcleo vive en la Raspberry Pi, esto describe
    la Pi. Sin `psutil` se devuelve `disponible: false` y la pantalla enseña un
    hueco en vez de romperse.
    """
    global _ultima_red
    try:
        import psutil  # noqa: PLC0415  (perezoso: el núcleo arranca sin él)
    except ImportError:
        return {"disponible": False, "motivo": "psutil no está instalado"}

    try:
        memoria = psutil.virtual_memory()
        disco = psutil.disk_usage(str(RAIZ.anchor or RAIZ))
        red = psutil.net_io_counters()
        ahora = time.monotonic()

        subida = bajada = 0.0
        if _ultima_red is not None:
            momento, enviados, recibidos = _ultima_red
            transcurrido = max(ahora - momento, 0.001)
            subida = max(red.bytes_sent - enviados, 0) / transcurrido
            bajada = max(red.bytes_recv - recibidos, 0) / transcurrido
        _ultima_red = (ahora, red.bytes_sent, red.bytes_recv)

        datos: dict[str, Any] = {
            "disponible": True,
            # `interval=None` da el porcentaje desde la llamada anterior, que es
            # justo lo que se quiere en una pantalla que se refresca sola. Con un
            # intervalo, esta función bloquearía el bucle ese tiempo.
            "cpu": psutil.cpu_percent(interval=None),
            "nucleos": psutil.cpu_count(logical=True),
            "memoria": {
                "usado": memoria.total - memoria.available,
                "total": memoria.total,
                "porcentaje": memoria.percent,
                "legible": f"{_bytes_legibles(memoria.total - memoria.available)} de {_bytes_legibles(memoria.total)}",
            },
            "disco": {
                "usado": disco.used,
                "total": disco.total,
                "porcentaje": disco.percent,
                "legible": f"{_bytes_legibles(disco.used)} de {_bytes_legibles(disco.total)}",
            },
            "red": {
                "subida": subida,
                "bajada": bajada,
                "legible": f"↑ {_bytes_legibles(subida)}/s · ↓ {_bytes_legibles(bajada)}/s",
            },
        }

        bateria = psutil.sensors_battery()
        if bateria is not None:
            datos["bateria"] = {
                "porcentaje": round(bateria.percent),
                "enchufado": bool(bateria.power_plugged),
            }

        _apuntar_muestra(datos["cpu"], memoria.percent)
        datos["historial"] = list(_HISTORIAL)
        return datos
    except Exception as e:  # noqa: BLE001  (un número informativo no tumba el panel)
        logger.warning("No se pudo leer la telemetría: %s", e)
        return {"disponible": False, "motivo": str(e)}


def _apuntar_muestra(cpu: float, memoria: float) -> None:
    """Guarda una muestra si ha pasado el suelo de tiempo desde la anterior.

    El instante se guarda **aparte y sin redondear**. Guardado redondeado, el de
    la muestra puede quedar unas décimas por delante del reloj y la resta sale
    negativa: la muestra siguiente se descarta por «venir del futuro». Es un
    fallo de dos décimas que solo asoma cuando dos vistazos caen muy juntos, o
    sea justo en las pruebas y con dos pantallas abiertas.
    """
    global _ultima_muestra
    ahora = time.time()
    if _ultima_muestra is not None and ahora - _ultima_muestra < SEGUNDOS_ENTRE_MUESTRAS:
        return
    _ultima_muestra = ahora
    _HISTORIAL.append(
        {"momento": round(ahora, 1), "cpu": round(float(cpu), 1), "memoria": round(float(memoria), 1)}
    )


def olvidar_historial() -> None:
    """Vacía las muestras. Existe para las pruebas, que si no se contaminan."""
    global _ultima_muestra
    _HISTORIAL.clear()
    _ultima_muestra = None


# --------------------------------------------------------------------------- #
# Presencia: lo que un asistente sabe sin que se lo preguntes
# --------------------------------------------------------------------------- #


async def presencia(cfg: almacen.Configuracion) -> dict[str, Any]:
    """Qué hay delante ahora mismo: qué se está haciendo, qué correo espera y
    qué toca en la agenda.

    Todo lo de aquí ya estaba en el sistema —en la cola, en el triaje, en el
    calendario— y era el usuario quien tenía que ir a buscarlo a tres sitios.
    """
    datos: dict[str, Any] = {"haciendo": None, "correo": {}, "proximo_evento": None, "eventos": []}

    try:
        trabajos, marcados = await asyncio.gather(
            asyncio.to_thread(almacen.listar, None, 50),
            asyncio.to_thread(almacen.correos_marcados),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("No se pudo reunir la presencia: %s", e)
        return datos

    en_curso = [t for t in trabajos if t["estado"] == almacen.EN_CURSO]
    esperando = [t for t in trabajos if t["estado"] == almacen.ESPERANDO]
    if en_curso:
        datos["haciendo"] = {"id": en_curso[0]["id"], "agente": en_curso[0]["agente"]}
    datos["esperando_un_si"] = len(esperando)

    # Correos triados que nadie ha resuelto todavía, por cajón. Es el mismo
    # criterio que la pestaña de Correo: lo que no está marcado está pendiente.
    pendientes: dict[str, int] = {}
    for trabajo in trabajos:
        resultado = trabajo.get("resultado")
        if not isinstance(resultado, dict):
            continue
        for correo in resultado.get("clasificados") or []:
            if correo.get("clase") == "ignorar" or marcados.get(correo.get("id")):
                continue
            pendientes[correo["clase"]] = pendientes.get(correo["clase"], 0) + 1
    datos["correo"] = pendientes

    # El calendario se pregunta solo si está configurado: sin esto, una pantalla
    # que se refresca sola pediría un testigo de Google cada pocos segundos.
    #
    # Y se pide EL DEL AGENTE, no uno nuevo. `abrir_calendario` fabricaba un
    # `CalendarioGoogle` por sondeo, cada uno con su `ClientSession` que nadie
    # cerraba nunca: el panel refresca cada veinte segundos, así que el núcleo
    # dejaba tres sesiones abiertas por minuto —1.417 en el registro— y pedía
    # un testigo de Google nuevo con cada una. Ahora se reutiliza el que ya
    # está abierto (2026-08-24).
    try:
        calendario = agenda.calendario()
    except RuntimeError:
        # El agente `agenda` no está iniciado (arranque con PERSEO_DISPARADORES
        # vacío, o un test). Sin calendario, y sin ruido.
        calendario = None
    if calendario is not None:
        try:
            eventos = await asyncio.wait_for(
                calendario.proximos(timedelta(hours=24)), timeout=TOPE_SONDEO
            )
            if eventos:
                datos["proximo_evento"] = eventos[0].a_dict()
                # Y el resto del día, con tope: la pantalla enseña una lista
                # corta, y traerse veinte reuniones para pintar cuatro es
                # ancho de banda del túnel tirado (misma lección que H-36).
                datos["eventos"] = [e.a_dict() for e in eventos[:EVENTOS_EN_PANTALLA]]
        except Exception as e:  # noqa: BLE001
            logger.warning("No se pudo mirar el calendario: %s", e)

    return datos


#: Dónde deja `perseo actualizar` la marca de la última construcción. Es el
#: mismo valor que queda incrustado en el binario de la app de escritorio, y por
#: eso vale para lo que se inventó: ver de un vistazo si las dos pantallas
#: enseñan lo mismo o una se quedó en una versión vieja.
NOMBRE_VERSION = "version.json"


def version_construida(cfg: almacen.Configuracion) -> dict[str, Any]:
    """La marca de la última construcción, o vacío si nunca se construyó.

    No es un dato crítico: si el fichero no está o está roto, se contesta con un
    diccionario vacío y las pantallas dicen «sin sellar». Reventar el estado
    entero por esto sería cambiar un aviso por una pantalla en blanco.
    """
    fichero = Path(cfg.directorio_datos) / NOMBRE_VERSION
    try:
        datos = json.loads(fichero.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return datos if isinstance(datos, dict) else {}


async def reunir(cfg: almacen.Configuracion, router: Router) -> dict[str, Any]:
    """Todo lo que pinta la pestaña de Estado, en una sola respuesta."""
    async with aiohttp.ClientSession() as http:
        piezas = await asyncio.gather(
            _sin_caerse(
                "ollama",
                "Modelo local",
                lambda: _recordando("ollama", MEMORIA_LOCAL, lambda: _ollama(cfg, http)),
            ),
            _sin_caerse(
                "vault",
                "Memoria",
                lambda: _recordando("vault", MEMORIA_LOCAL, lambda: _vault(cfg)),
            ),
            _sin_caerse(
                "google",
                "Google",
                lambda: _recordando("google", MEMORIA_GOOGLE, lambda: _google(cfg)),
            ),
        )

    # Lo que se sabe sin preguntar a nadie: no se recuerda porque no cuesta nada.
    piezas = [
        *piezas,
        _suplente(cfg),
        _telegram(cfg),
        _mcp(cfg),
        _correo(cfg),
        _agenda(cfg),
        _dev(cfg),
        _web(cfg),
        _chat(cfg),
        _confianza(),
    ]

    recuento, usos, contexto, maquina = await asyncio.gather(
        asyncio.to_thread(almacen.recuento_por_estado),
        asyncio.to_thread(almacen.uso_de_hoy),
        presencia(cfg),
        asyncio.to_thread(telemetria),
    )

    return {
        "generado": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "encendido_desde": _ARRANQUE_RELOJ.replace(microsecond=0).isoformat(),
        "encendido_segundos": int(time.monotonic() - _ARRANQUE),
        "router_local": router.disponible,
        "maquina": maquina,
        "presencia": contexto,
        "piezas": [asdict(p) for p in piezas],
        "trabajos": recuento,
        "version": version_construida(cfg),
        "agentes": sorted(REGISTRO),
        "disparadores": [
            {
                "nombre": nombre,
                "activo": nombre in cfg.disparadores,
                "intervalo": float(cfg.intervalos.get(nombre, d.intervalo)),
            }
            for nombre, d in sorted(DISPARADORES.items())
        ],
        "cuota": _cuota(cfg, usos),
    }
