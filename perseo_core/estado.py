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
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

import aiohttp

from . import almacen, politica
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

#: Cuánto se recuerda cada sondeo. Ollama y Obsidian son locales y baratos;
#: Google pide un testigo nuevo a un servidor de fuera, así que se pregunta una
#: vez cada cinco minutos por muchas veces que se mire la pantalla.
MEMORIA_LOCAL = 10
MEMORIA_GOOGLE = 300

#: Peticiones al día del plan gratuito, leídas en aistudio.google.com/rate-limit
#: el 2026-08-16. **No salen de la documentación**: Google dejó de publicarlas, y
#: la que había escrita estaba desfasada por un factor de doce. Se comparan por
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
            "PERSEO_MODELO_SUPLENTE=gemma-4-31b-it. Es lo único que manda a un "
            "tercero el texto que se clasifica.",
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
    return Pieza("telegram", "Telegram", OK, f"Avisos y aprobaciones desde {cfg.url_base}.")


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
        _correo(cfg),
        _agenda(cfg),
        _dev(cfg),
        _web(cfg),
        _confianza(),
    ]

    recuento, usos = await asyncio.gather(
        asyncio.to_thread(almacen.recuento_por_estado),
        asyncio.to_thread(almacen.uso_de_hoy),
    )

    return {
        "generado": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "encendido_desde": _ARRANQUE_RELOJ.replace(microsecond=0).isoformat(),
        "encendido_segundos": int(time.monotonic() - _ARRANQUE),
        "router_local": router.disponible,
        "piezas": [asdict(p) for p in piezas],
        "trabajos": recuento,
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
