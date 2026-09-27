"""Telegram de dos sentidos: escribirle a Perseo como se le escribe a Instinct.

El 2026-08-22 Telegram se recortó a solo avisar («telegram no me está sirviendo
de nada»), y esa decisión sigue siendo la de fábrica: **esto viene apagado**.
Con `PERSEO_TELEGRAM_CONVERSAR=1` se enciende la otra mitad, la que Instinct
tiene de serie: le escribes al bot y contesta, en el **hilo principal**
(`servicios/hilo.py`), el mismo que ves en el panel y en el móvil.

**El precio, dicho claro.** La regla de `caras/telegram.py` era «titular por
Telegram, detalle por Tailscale»: por un tercero solo viajaban recuentos.
Conversar por Telegram es mandarle a Telegram lo que dices y lo que te contesta
Perseo, con sus correos, citas y lo que haga falta. Por eso es un interruptor y
no el comportamiento de fábrica, y por eso está en `docs/PRIVACIDAD.md`.

**Solo su chat.** Lo que llegue de cualquier otro chat —un bot de Telegram lo
puede encontrar cualquiera— no se atiende: se apunta una vez en el registro y
nada más. Una ubicación compartida en su chat se guarda como su última
ubicación (`servicios/ubicacion.py`).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import aiohttp

from ..infra.bus import Bus
from ..infra.configuracion import Configuracion
from ..servicios import hilo, recordatorios, ubicacion

logger = logging.getLogger(__name__)

#: El sondeo largo de `getUpdates`: Telegram sostiene la petición hasta que hay algo.
SONDEO = 25

#: Lo que se deja de contestar al arrancar: mensajes de antes de encender, que ya
#: no esperan respuesta.
MARGEN_ARRANQUE = 120


def encendido(cfg: Configuracion) -> bool:
    return bool(cfg.telegram_token and cfg.telegram_chat) and os.environ.get(
        "PERSEO_TELEGRAM_CONVERSAR", ""
    ).strip().lower() in ("1", "si", "sí", "true")


class TelegramConversa:
    def __init__(self, cfg: Configuracion, bus: Bus) -> None:
        self._cfg = cfg
        self._bus = bus
        self._parar = asyncio.Event()
        self._sesion: aiohttp.ClientSession | None = None
        self._desde = 0
        self._ajenos_vistos: set[str] = set()
        self._en_curso: set[asyncio.Task[Any]] = set()

    def detener(self) -> None:
        self._parar.set()

    async def _llamar(self, metodo: str, **carga: Any) -> dict[str, Any] | None:
        assert self._sesion is not None
        url = f"{self._cfg.telegram_api}/bot{self._cfg.telegram_token}/{metodo}"
        try:
            async with self._sesion.post(url, json=carga) as r:
                datos = await r.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
            logger.warning("Telegram (conversa): %s falló (%s)", metodo, e)
            return None
        return datos if isinstance(datos, dict) and datos.get("ok") else None

    async def ejecutar(self) -> None:
        if not encendido(self._cfg):
            logger.info("Telegram de dos sentidos apagado (PERSEO_TELEGRAM_CONVERSAR).")
            return
        self._sesion = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=SONDEO + 15))
        arranque = time.time()
        logger.info("Telegram de dos sentidos en marcha: contesta en el hilo principal.")
        try:
            while not self._parar.is_set():
                datos = await self._llamar("getUpdates", offset=self._desde, timeout=SONDEO)
                if datos is None:
                    await asyncio.sleep(5)
                    continue
                for actualizacion in datos.get("result") or []:
                    self._desde = max(self._desde, int(actualizacion.get("update_id", 0)) + 1)
                    self.atender(actualizacion, arranque)
        finally:
            await self._sesion.close()
            self._sesion = None

    def atender(self, actualizacion: dict[str, Any], arranque: float) -> str | None:
        """Decide qué hacer con una actualización. Devuelve qué se hizo, para las pruebas."""
        mensaje = actualizacion.get("message") or {}
        chat = str((mensaje.get("chat") or {}).get("id", ""))
        if not chat:
            return None
        if chat != str(self._cfg.telegram_chat):
            if chat not in self._ajenos_vistos:
                self._ajenos_vistos.add(chat)
                logger.warning("Telegram: un chat que no es el suyo (%s) escribió al bot; se ignora.", chat)
            return "ajeno"
        if float(mensaje.get("date") or 0) < arranque - MARGEN_ARRANQUE:
            return "viejo"
        sitio = mensaje.get("location")
        if isinstance(sitio, dict):
            ubicacion.guardar(
                self._cfg.directorio_datos, sitio.get("latitude"), sitio.get("longitude"),
                recordatorios.ahora_local(), sitio.get("horizontal_accuracy"), "Telegram",
            )
            self._lanzar(self._enviar("Ubicación guardada."))
            return "ubicacion"
        texto = str(mensaje.get("text") or "").strip()
        if not texto or texto.startswith("/start"):
            return None
        self._lanzar(self._turno(texto))
        return "turno"

    def _lanzar(self, corrutina: Any) -> None:
        tarea = asyncio.create_task(corrutina)
        self._en_curso.add(tarea)
        tarea.add_done_callback(self._en_curso.discard)

    async def _turno(self, texto: str) -> None:
        await self._llamar("sendChatAction", chat_id=self._cfg.telegram_chat, action="typing")
        try:
            respuesta = await hilo.hablar(self._cfg.directorio_datos, texto, "telegram")
        except Exception as e:  # noqa: BLE001 — que el mensaje no se quede sin respuesta
            logger.exception("Telegram: el turno falló.")
            respuesta = f"No he podido contestar: {e}"
        await self._enviar(respuesta)

    async def _enviar(self, texto: str) -> None:
        for i in range(0, max(len(texto), 1), 4000):
            await self._llamar("sendMessage", chat_id=self._cfg.telegram_chat, text=texto[i : i + 4000] or "…")
