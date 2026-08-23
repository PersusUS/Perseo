"""Telegram: avisos que llegan al bolsillo. Solo avisa, nunca pregunta ni escucha.

Telegram está aquí por una limitación concreta: **Tailscale no da push**. iOS
obliga a pasar por APNs, así que un núcleo que solo hable por el tailnet puede
enseñarte cosas cuando abres la pantalla, pero no puede avisarte. Telegram sí.

A cambio, es un tercero. De ahí la regla que gobierna todo este módulo:

    **titular por Telegram, detalle por Tailscale.**

Por Telegram sale lo justo para saber que algo pasó —y un enlace para leerlo en
la web, que va cifrada por WireGuard y no pasa por servidores ajenos. Por eso se
manda `resumen` y nunca `detalle` ni `resultado`.

**Recortado el 2026-08-22 a notificador de una sola dirección** (encargo N-1 del
señor Persus: «telegram no me está sirviendo de nada»). Fuera los botones de
aprobar/rechazar, fuera el sondeo de `getUpdates` y con ellos el filtro de chat:
un canal que solo escribe no tiene de quién defenderse. Lo que espera un sí se
sigue anunciando —es lo único parado esperándote— pero la decisión se da donde
hay una persona: **por voz durante una llamada**, o en el panel y la web.

Una cosa sigue sin ser opcional: **sin token configurado, el núcleo arranca
igual.** Telegram es un canal más, no una pieza de la que dependa nada.

Ver bitacora/05_PLAN_PERSEO_V2.md §5 y §7, y bitacora/06_HANDOFF.md §4 y §12.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

import aiohttp

from . import almacen
from .bus import Bus, Evento

logger = logging.getLogger(__name__)

#: Segundos de margen sobre la espera del sondeo para el timeout total de la
#: sesión HTTP. El nombre viene de cuando aquí había un `getUpdates` de sondeo
#: largo; hoy solo fija cuánto se aguanta colgada una llamada a la API.
ESPERA_SONDEO = 25

# -- qué se manda, y qué no ------------------------------------------------- #
#
# Replanteado el 2026-08-22, porque lo que llegaba al móvil no servía para nada:
# «Trabajo #7 terminado» no dice qué se hizo, ni de qué agente, ni si hay que
# hacer algo. Un aviso que no cambia lo que vas a hacer es ruido, y el ruido
# acaba en un canal silenciado — que es peor que no tener canal.
#
# Las reglas, en el orden en que se aplican:
#
#   1. **Lo que espera un sí, siempre.** Es lo único que está parado esperándote.
#      Sin botones: la decisión se da por voz en una llamada, o en el panel y la
#      web, donde el enlace de abajo lleva.
#   2. **Lo que falla, siempre.** Un fallo cambia lo que vas a hacer, y encima
#      llega con la primera línea del error, que suele bastar para saber si es
#      la red o es el código.
#   3. **Lo que termina, solo si el agente escribió un titular.** «3 correos, 1
#      requiere acción» dice algo; «Trabajo #7 terminado» no. Si el agente no
#      supo resumirlo, no merecía molestar.
#   4. **Lo que lanzaste tú, nunca.** Si has encolado algo desde el panel o
#      desde la llamada, estás mirando la pantalla: ahí lo verás terminar. Solo
#      se avisa de lo que hicieron los disparadores por su cuenta.
#   5. **Lo cancelado y lo rechazado, nunca.** Lo cancelaste tú.
#
# Y en todos: **titular por Telegram, detalle por Tailscale** (regla de arriba).
# El pie dice de quién y cuándo, que es lo que hacía falta para no tener que
# abrir la web solo para saber si aquello era el correo o la agenda.

#: Tope de lo que se enseña de una petición o de un error. Telegram corta a
#: 4096, pero el problema no es ese: un aviso de tres pantallas no se lee.
LARGO_MAXIMO = 160


def _recortar(texto: str, tope: int = LARGO_MAXIMO) -> str:
    limpio = " ".join(str(texto).split())
    return limpio if len(limpio) <= tope else limpio[: tope - 1].rstrip() + "…"


def resumir_peticion(trabajo: dict[str, Any]) -> str:
    """Qué se le pidió, en una línea y en castellano.

    Mismas reglas que el panel (`Panel.tsx::resumirPeticion`): un trabajo de
    correo trae el lote entero dentro, y volcarlo manda el JSON de veinte
    correos por Telegram.
    """
    peticion = trabajo.get("peticion")
    if not isinstance(peticion, dict):
        return _recortar(peticion) if peticion else "sin petición"

    mensajes = peticion.get("mensajes")
    if isinstance(mensajes, list):
        return f"{len(mensajes)} correo{'' if len(mensajes) == 1 else 's'} del buzón"

    for clave in ("texto", "consulta", "titulo", "accion", "orden"):
        valor = peticion.get(clave)
        if valor:
            return _recortar(valor)

    return _recortar(", ".join(sorted(peticion)) or "sin petición")


def _pie(trabajo: dict[str, Any]) -> str:
    """De quién es esto y cuándo pasó."""
    partes = [str(trabajo.get("agente") or "perseo"), f"#{trabajo.get('id')}"]
    partes.append(datetime.now().strftime("%H:%M"))
    return " · ".join(partes)


def redactar(evento: Evento, url_base: str) -> tuple[str, list[list[dict[str, Any]]]] | None:
    """El mensaje que sale al móvil, o `None` si este evento no merece molestar.

    Es una función pura a propósito: decidir qué se manda es lo que más se va a
    discutir de este módulo, y así se puede probar sin levantar un Telegram.
    """
    trabajo = evento.datos.get("trabajo") or {}
    if trabajo.get("id") is None:
        return None

    ver = [{"text": "Ver en Perseo", "url": f"{url_base}/"}]

    if evento.tipo == "trabajo.espera_confirmacion":
        confirmacion = trabajo.get("confirmacion") or {}
        resumen = confirmacion.get("resumen") or resumir_peticion(trabajo)
        # El `detalle` se queda deliberadamente fuera: para eso está el enlace,
        # que va por el tailnet. Y la decisión no se toma aquí: por voz en una
        # llamada, o en la pantalla que abre el enlace.
        return (
            f"Perseo espera un sí\n\n{_recortar(resumen)}\n\n{_pie(trabajo)}",
            [ver],
        )

    if evento.tipo == "trabajo.fallido":
        error = _recortar(str(trabajo.get("error") or "").splitlines()[0] if trabajo.get("error") else "sin detalle")
        return (
            f"Falló: {resumir_peticion(trabajo)}\n\n{error}\n\n{_pie(trabajo)}",
            [ver],
        )

    if evento.tipo != "trabajo.hecho":
        # Cancelado y rechazado: los decidiste tú hace dos segundos.
        return None

    # Lo que encolas desde el panel o desde la llamada lo estás mirando: avisar
    # al móvil de eso es contarte lo que ya ves.
    if trabajo.get("origen") != "disparador":
        return None

    resultado = trabajo.get("resultado")
    titular = ""
    if isinstance(resultado, dict):
        titular = str(resultado.get("titular") or "").strip()
    if not titular:
        # Un disparador que no sabe resumir lo que ha hecho no tiene nada que
        # decir por aquí. El trabajo sigue en la cola, que es donde se mira.
        return None

    return (f"{_recortar(titular, 300)}\n\n{_pie(trabajo)}", [ver])


class Telegram:
    """Puente de solo salida entre el bus del núcleo y un chat de Telegram."""

    def __init__(self, cfg: almacen.Configuracion, bus: Bus) -> None:
        self._cfg = cfg
        self._bus = bus
        self._sesion: aiohttp.ClientSession | None = None
        self._parar = asyncio.Event()

    # -- ciclo de vida ----------------------------------------------------- #

    async def ejecutar(self) -> None:
        """Escucha el bus y anuncia, hasta que se pida parar."""
        if not self._cfg.telegram_configurado:
            logger.info("Telegram no configurado; se sigue sin ese canal.")
            return

        self._sesion = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=ESPERA_SONDEO + 15)
        )
        logger.info("Telegram en marcha (chat %s).", self._cfg.telegram_chat)
        if not self._cfg.url_base_alcanzable:
            # Los avisos salen igual: el titular es lo que importa, y el enlace
            # es un extra. Pero se dice, porque el síntoma —"le doy a Ver
            # detalle y sale una página en blanco"— no lleva a la causa ni de
            # lejos: en el móvil, `127.0.0.1` es el móvil.
            logger.warning(
                "El enlace 'ver detalle' apunta a %s, que desde el móvil no lleva a "
                "ninguna parte. Arranca con PERSEO_CORE_HOST=tailscale, o pon "
                "PERSEO_URL_BASE a mano.",
                self._cfg.url_base,
            )
        try:
            await self._anunciar_eventos()
        finally:
            await self._sesion.close()
            self._sesion = None
            logger.info("Telegram detenido.")

    def detener(self) -> None:
        self._parar.set()

    # -- llamadas a la API ------------------------------------------------- #

    async def _llamar(self, metodo: str, **carga: Any) -> dict[str, Any] | None:
        """Llama a un método de la API. Devuelve `None` si no se pudo.

        Nunca lanza: que Telegram esté caído no debe tumbar el núcleo ni impedir
        que el trabajo siga su curso por la web.
        """
        assert self._sesion is not None
        url = f"{self._cfg.telegram_api}/bot{self._cfg.telegram_token}/{metodo}"
        try:
            async with self._sesion.post(url, json=carga) as respuesta:
                datos = await respuesta.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
            logger.warning("Telegram: %s falló (%s)", metodo, e)
            return None

        if not datos.get("ok"):
            logger.warning("Telegram: %s devolvió %s", metodo, datos.get("description"))
            return None
        return datos

    async def _enviar(self, texto: str, botones: list[list[dict[str, Any]]] | None = None) -> None:
        carga: dict[str, Any] = {"chat_id": self._cfg.telegram_chat, "text": texto}
        if botones:
            carga["reply_markup"] = {"inline_keyboard": botones}
        await self._llamar("sendMessage", **carga)

    # -- del núcleo al móvil ------------------------------------------------ #

    async def _anunciar_eventos(self) -> None:
        # Se sale de aquí por cancelación al cerrar el núcleo, igual que el
        # flujo SSE de la API. Mirar además el evento de parada obligaría a
        # competir contra la espera de la cola, y en esa carrera se pierden
        # eventos.
        async with self._bus.suscribir() as eventos:
            async for evento in eventos:
                await self._anunciar(evento)

    async def _anunciar(self, evento: Evento) -> None:
        """Manda lo que `redactar` diga, y calla cuando dice que no hay nada."""
        mensaje = redactar(evento, self._cfg.url_base)
        if mensaje is None:
            return
        texto, botones = mensaje
        await self._enviar(texto, botones=botones)


# --------------------------------------------------------------------------- #
# Puesta en marcha a mano
#
# Configurar el canal necesita dos datos y solo uno lo da @BotFather. El otro
# —el `chat_id` propio— no se puede consultar en ninguna parte: aparece cuando
# alguien le escribe al bot, y hay que sacarlo de ahí. Esto es eso: el destino de
# los avisos. Nada más se lee de Telegram jamás.
# --------------------------------------------------------------------------- #

#: De dónde puede salir un chat en una actualización. Un mensaje normal, uno
#: editado, y el que viene pegado a la pulsación de un botón.
_ENVOLTORIOS = ("message", "edited_message", "channel_post")


def chats_vistos(actualizaciones: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Los chats que le han escrito al bot: identificador y a quién pertenece.

    Sin repetir y en orden de aparición, para que la lista sirva para elegir
    cuando ha escrito más de uno.
    """
    vistos: dict[str, str] = {}
    for actualizacion in actualizaciones:
        if not isinstance(actualizacion, dict):
            continue
        candidatos = [(actualizacion.get(e) or {}).get("chat") for e in _ENVOLTORIOS]
        pulsacion = actualizacion.get("callback_query") or {}
        candidatos.append((pulsacion.get("message") or {}).get("chat"))
        for chat in candidatos:
            if not isinstance(chat, dict) or chat.get("id") is None:
                continue
            vistos.setdefault(str(chat["id"]), _nombre_del_chat(chat))
    return list(vistos.items())


def _nombre_del_chat(chat: dict[str, Any]) -> str:
    """Cómo llamar a un chat por pantalla. Solo para que el usuario se reconozca."""
    partes = [str(chat.get(c, "")).strip() for c in ("first_name", "last_name", "title")]
    nombre = " ".join(p for p in partes if p)
    usuario = str(chat.get("username", "")).strip()
    if usuario:
        nombre = f"{nombre} (@{usuario})" if nombre else f"@{usuario}"
    return nombre or "sin nombre"


async def _pedir(cfg: almacen.Configuracion, metodo: str, **carga: Any) -> dict[str, Any]:
    """Como `Telegram._llamar`, pero para la línea de comandos: aquí sí se lanza.

    En marcha, que Telegram falle no puede tumbar el núcleo. Configurando es al
    revés: un fallo silencioso deja al usuario mirando una pantalla que no dice
    qué ha pasado.
    """
    url = f"{cfg.telegram_api}/bot{cfg.telegram_token}/{metodo}"
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as sesion:
        async with sesion.post(url, json=carga) as respuesta:
            datos = await respuesta.json()
    if not datos.get("ok"):
        raise RuntimeError(f"{metodo} devolvió {datos.get('description')}")
    return datos


def _sincrono() -> None:  # pragma: no cover - atajo para la línea de comandos
    """`python -m perseo_core.telegram`: descubre el `chat_id` y lo deja puesto.

    **Con el núcleo parado** para que nada más esté leyendo. Sin `chat_id` no hay
    a quién enviar los avisos, y este dato no se puede consultar en ninguna
    parte: aparece cuando alguien le escribe al bot.
    """
    import sys

    cfg = almacen.cargar_configuracion()
    if not cfg.telegram_token:
        print(
            "No hay token del bot. Lo da @BotFather, y se pone en "
            f"PERSEO_TELEGRAM_TOKEN o en {cfg.directorio_datos / 'telegram.txt'}."
        )
        sys.exit(1)

    async def guion() -> None:
        yo = (await _pedir(cfg, "getMe")).get("result") or {}
        print(f"El bot es @{yo.get('username')} ({yo.get('first_name')}).")

        # `timeout=0`: aquí se mira lo que hay y se sale. Es la única lectura
        # que hace este módulo, y solo mientras se configura: el núcleo en marcha
        # no lee nunca.
        chats = chats_vistos((await _pedir(cfg, "getUpdates", timeout=0)).get("result") or [])

        if cfg.telegram_chat:
            print(f"El chat ya está configurado: {cfg.telegram_chat}.")
            if chats and cfg.telegram_chat not in dict(chats):
                print(
                    "Aviso: quien ha escrito al bot no es ese chat "
                    f"({', '.join(i for i, _ in chats)}). Los avisos van al configurado."
                )
            return

        if not chats:
            print(
                "Nadie le ha escrito al bot todavía. Mándale algo desde tu móvil "
                "(/start vale) y vuelve a ejecutar esto."
            )
            return

        if len(chats) > 1:
            print("Han escrito varios chats. Elige el tuyo y ponlo a mano:")
            for identificador, nombre in chats:
                print(f"  {identificador}  {nombre}")
            print(f"  -> {cfg.directorio_datos / 'telegram_chat.txt'}")
            return

        identificador, nombre = chats[0]
        destino = cfg.directorio_datos / "telegram_chat.txt"
        destino.write_text(identificador, encoding="utf-8")
        print(f"Chat de {nombre} guardado: {identificador} -> {destino}")

    try:
        asyncio.run(guion())
    except (RuntimeError, aiohttp.ClientError) as e:
        print(f"No: {e}")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    _sincrono()
