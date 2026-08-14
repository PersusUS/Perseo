"""Arranque de perseo-core.

    python -m perseo_core

Levanta, en este orden: base de datos, recuperación de trabajos huérfanos, bus,
router, trabajador y API. El orden importa — los huérfanos se recuperan antes de
que el trabajador empiece a reclamar, cuando por definición no hay nadie
ejecutando nada.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import sys

from aiohttp import web

from . import almacen, api
from .agentes import Router, Trabajador
from .bus import Bus

logger = logging.getLogger("perseo_core")


def _configurar_registro() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )


def _avisar_de_la_escucha(cfg: almacen.Configuracion) -> None:
    for host in cfg.hosts:
        logger.info("Escuchando en http://%s:%d", host, cfg.puerto)
        if host in almacen.LOCALES:
            continue
        # No es un error —la Fase B lo requiere— pero sí algo que conviene ver
        # en el registro para no descubrirlo por accidente.
        logger.warning(
            "El núcleo escucha en una interfaz no local (%s). "
            "Asegúrate de que es la de Tailscale y no una red abierta.",
            host,
        )
    logger.info("Token en %s", cfg.directorio_datos / "token.txt")


async def arrancar() -> None:
    cfg = almacen.cargar_configuracion()

    almacen.abrir(cfg)
    almacen.recuperar_huerfanos()

    bus = Bus()
    router = Router(cfg)
    await router.abrir()

    trabajador = Trabajador(bus)
    tarea_trabajador = asyncio.create_task(trabajador.ejecutar(), name="trabajador")

    runner = web.AppRunner(api.crear_app(cfg, bus, router))
    await runner.setup()
    # Una lista de hosts, nunca `0.0.0.0`: se abren exactamente las interfaces
    # enumeradas y ninguna más.
    sitio = web.TCPSite(runner, list(cfg.hosts), cfg.puerto)
    await sitio.start()
    _avisar_de_la_escucha(cfg)

    # Espera hasta que llegue una señal de parada. En Windows `add_signal_handler`
    # no está implementado, así que se cae al manejador síncrono de `signal`.
    parada = asyncio.Event()
    bucle = asyncio.get_running_loop()
    for nombre in ("SIGINT", "SIGTERM"):
        senal = getattr(signal, nombre, None)
        if senal is None:
            continue
        try:
            bucle.add_signal_handler(senal, parada.set)
        except NotImplementedError:
            signal.signal(senal, lambda *_: bucle.call_soon_threadsafe(parada.set))

    try:
        await parada.wait()
    finally:
        logger.info("Cerrando…")
        trabajador.detener()
        tarea_trabajador.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await tarea_trabajador
        await runner.cleanup()
        await router.cerrar()
        almacen.cerrar()
        logger.info("Adiós.")


def main() -> None:
    _configurar_registro()
    try:
        asyncio.run(arrancar())
    except KeyboardInterrupt:
        # Ctrl+C en Windows puede llegar por aquí en lugar de por el manejador.
        pass


if __name__ == "__main__":
    main()
