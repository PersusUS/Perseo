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
