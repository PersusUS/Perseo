"""Telegram: notificaciones que llegan al bolsillo y aprobaciones desde el móvil.

Telegram está aquí por una limitación concreta: **Tailscale no da push**. iOS
obliga a pasar por APNs, así que un núcleo que solo hable por el tailnet puede
enseñarte cosas cuando abres la pantalla, pero no puede avisarte. Telegram sí.

A cambio, es un tercero. De ahí la regla que gobierna todo este módulo:

    **titular por Telegram, detalle por Tailscale.**

Por Telegram sale lo justo para decidir —qué se pregunta y qué trabajo es— y un
enlace. El cuerpo de un correo, el contenido de un fichero o cualquier resultado
se leen en la web, que va cifrada por WireGuard y no pasa por servidores ajenos.
Por eso se manda `resumen` y nunca `detalle` ni `resultado`.

Dos cosas más que no son opcionales:

1. **Solo se atiende al chat configurado.** Un bot de Telegram es público: quien
   sepa su nombre puede escribirle. Sin ese filtro, cualquiera podría aprobar
   una acción irreversible pulsando un botón.
2. **Sin token configurado, el núcleo arranca igual.** Telegram es un canal más,
   no una pieza de la que dependa nada.

Ver bitacora/05_PLAN_PERSEO_V2.md §5 y §7, y bitacora/06_HANDOFF.md §4.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

from . import almacen
from .bus import Bus, Evento

logger = logging.getLogger(__name__)

#: Segundos que se deja abierta cada llamada a getUpdates. Long polling: el
#: servidor de Telegram no contesta hasta que hay algo o se agota el plazo, así
#: que esperar más es menos tráfico, no más lentitud.
ESPERA_SONDEO = 25

#: Tras un error de red se espera esto antes de reintentar, para no castigar a
#: la API cuando lo que se ha caído es el wifi.
ESPERA_TRAS_FALLO = 5

#: Eventos que se anuncian, y cómo se titulan. Los que no están aquí —como
#: `trabajo.encolado`— no se mandan: llenar el móvil de avisos de tránsito es la
#: forma más rápida de que se silencie el canal.
TITULARES = {
    "trabajo.hecho": "Trabajo #{id} terminado",
    "trabajo.fallido": "Trabajo #{id} falló",
    "trabajo.rechazado": "Trabajo #{id} rechazado",
}


class Telegram:
    """Puente entre el bus del núcleo y un chat de Telegram."""

    def __init__(self, cfg: almacen.Configuracion, bus: Bus) -> None:
        self._cfg = cfg
        self._bus = bus
        self._sesion: aiohttp.ClientSession | None = None
        self._parar = asyncio.Event()
        #: Identificador de la última actualización procesada. Telegram las
        #: reenvía hasta que se confirman con `offset`, así que esto es lo que
        #: evita atender dos veces el mismo botón tras un reinicio.
        self._offset: int | None = None

    # -- ciclo de vida ----------------------------------------------------- #

    async def ejecutar(self) -> None:
        """Escucha el bus y el chat a la vez, hasta que se pida parar."""
        if not self._cfg.telegram_configurado:
            logger.info("Telegram no configurado; se sigue sin ese canal.")
            return

        self._sesion = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=ESPERA_SONDEO + 15)
        )
        logger.info("Telegram en marcha (chat %s).", self._cfg.telegram_chat)
        try:
            await asyncio.gather(self._anunciar_eventos(), self._atender_respuestas())
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
        trabajo = evento.datos.get("trabajo") or {}
        id_trabajo = trabajo.get("id")
        if id_trabajo is None:
            return

        if evento.tipo == "trabajo.espera_confirmacion":
            confirmacion = trabajo.get("confirmacion") or {}
            resumen = confirmacion.get("resumen") or "¿Confirmas?"
            # El `detalle` se queda deliberadamente fuera: para eso está el
            # enlace, que va por el tailnet.
            await self._enviar(
                f"Perseo necesita un sí\n\n{resumen}\n\nTrabajo #{id_trabajo}",
                botones=[
                    [
                        {"text": "Aprobar", "callback_data": f"aprobar:{id_trabajo}"},
                        {"text": "Rechazar", "callback_data": f"rechazar:{id_trabajo}"},
                    ],
                    [{"text": "Ver detalle", "url": f"{self._cfg.url_base}/"}],
                ],
            )
            return

        # Un agente puede escribir su propio titular. Es lo que necesita la Fase
        # D: "3 correos, 1 requiere acción" dice algo, y "Trabajo #7 terminado"
        # no. La regla del canal la sigue poniendo este módulo —lo que llegue
        # aquí se manda tal cual—, así que el titular lo compone el agente
        # sabiendo que sale por un tercero, y el detalle se queda en la cola.
        # Un titular vacío significa "no merece molestar": no se manda nada.
        resultado = trabajo.get("resultado")
        if evento.tipo == "trabajo.hecho" and isinstance(resultado, dict) and "titular" in resultado:
            titular = str(resultado.get("titular") or "").strip()
            if titular:
                await self._enviar(
                    f"{titular}\n\nTrabajo #{id_trabajo}",
                    botones=[[{"text": "Ver detalle", "url": f"{self._cfg.url_base}/"}]],
                )
            return

        plantilla = TITULARES.get(evento.tipo)
        if plantilla:
            await self._enviar(plantilla.format(id=id_trabajo))

    # -- del móvil al núcleo ------------------------------------------------ #

    async def _atender_respuestas(self) -> None:
        while not self._parar.is_set():
            datos = await self._llamar(
                "getUpdates", offset=self._offset, timeout=ESPERA_SONDEO
            )
            if datos is None:
                # Puede ser la red, o que se esté cerrando. Si es lo segundo, la
                # espera se corta sola.
                try:
                    await asyncio.wait_for(self._parar.wait(), timeout=ESPERA_TRAS_FALLO)
                except asyncio.TimeoutError:
                    pass
                continue

            for actualizacion in datos.get("result") or []:
                # Se confirma siempre, incluso lo que se descarta: si no,
                # Telegram reenviaría eternamente el mensaje de un desconocido.
                self._offset = int(actualizacion["update_id"]) + 1
                await self._procesar(actualizacion)

    def _es_del_chat(self, chat: dict[str, Any] | None) -> bool:
        return bool(chat) and str(chat.get("id")) == self._cfg.telegram_chat

    async def _procesar(self, actualizacion: dict[str, Any]) -> None:
        pulsacion = actualizacion.get("callback_query")
        if pulsacion is None:
            mensaje = actualizacion.get("message") or {}
            if self._es_del_chat(mensaje.get("chat")):
                await self._enviar(
                    "Por aquí solo atiendo confirmaciones. "
                    f"Para hablar con Perseo: {self._cfg.url_base}/"
                )
            return

        chat = ((pulsacion.get("message") or {}).get("chat")) or {}
        if not self._es_del_chat(chat):
            # Un bot de Telegram es público. Sin este filtro, cualquiera que lo
            # encontrara podría aprobar una acción irreversible.
            logger.warning("Telegram: pulsación descartada, viene del chat %s.", chat.get("id"))
            await self._llamar(
                "answerCallbackQuery", callback_query_id=pulsacion["id"], text="No autorizado"
            )
            return

        decision, _, crudo = str(pulsacion.get("data", "")).partition(":")
        if decision not in ("aprobar", "rechazar") or not crudo.isdigit():
            await self._llamar("answerCallbackQuery", callback_query_id=pulsacion["id"])
            return

        id_trabajo = int(crudo)
        trabajo = await asyncio.to_thread(
            almacen.resolver_confirmacion, id_trabajo, decision == "aprobar"
        )

        if trabajo is None:
            # La web se adelantó, o el trabajo se canceló. No es un error.
            await self._llamar(
                "answerCallbackQuery",
                callback_query_id=pulsacion["id"],
                text="Ese trabajo ya estaba resuelto",
            )
            return

        # Se publica en el bus para que la web se entere de lo que se decidió
        # desde el móvil, igual que el móvil se entera de lo que se decide en la
        # web.
        self._bus.publicar(
            "trabajo.aprobado" if decision == "aprobar" else "trabajo.rechazado", trabajo=trabajo
        )
        await self._llamar(
            "answerCallbackQuery",
            callback_query_id=pulsacion["id"],
            text="Aprobado" if decision == "aprobar" else "Rechazado",
        )
        # Los botones ya no valen para nada: dejarlos invita a pulsarlos otra vez.
        mensaje = pulsacion.get("message") or {}
        if mensaje.get("message_id") is not None:
            await self._llamar(
                "editMessageReplyMarkup",
                chat_id=self._cfg.telegram_chat,
                message_id=mensaje["message_id"],
                reply_markup={"inline_keyboard": []},
            )


# --------------------------------------------------------------------------- #
# Puesta en marcha a mano
#
# Configurar el canal necesita dos datos y solo uno lo da @BotFather. El otro
# —el `chat_id` propio— no se puede consultar en ninguna parte: aparece cuando
# alguien le escribe al bot, y hay que sacarlo de ahí. Esto es eso.
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

    **Con el núcleo parado.** Telegram solo deja un `getUpdates` a la vez: con el
    núcleo sondeando, esto se lleva un 409 que habla de "otra petición" y no de
    lo que de verdad pasa.
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

        # `timeout=0`: aquí se mira lo que hay y se sale. El sondeo largo es del
        # núcleo. Y sin `offset` no se confirma nada, así que lo que llegue
        # ahora lo seguirá viendo el núcleo cuando arranque.
        chats = chats_vistos((await _pedir(cfg, "getUpdates", timeout=0)).get("result") or [])

        if cfg.telegram_chat:
            print(f"El chat ya está configurado: {cfg.telegram_chat}.")
            if chats and cfg.telegram_chat not in dict(chats):
                print(
                    "Aviso: quien ha escrito al bot no es ese chat "
                    f"({', '.join(i for i, _ in chats)}). Solo se atiende al configurado."
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
