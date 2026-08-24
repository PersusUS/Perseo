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
import ssl
import sys

# Con alias: este paquete tiene su propio `web` —el agente— y sin el alias uno
# tapa al otro. El sintoma es un AttributeError en `web.AppRunner` al arrancar.
from aiohttp import web as servidor

from . import agenda, almacen, api, chat, correo, dev, mcp, memoria, pc, politica, web
from .agentes import Router, Trabajador
from .bus import Bus
from .disparadores import Planificador
from .telegram import Telegram

# Estos siete se importan por sus efectos: al cargarse registran sus agentes —y
# `correo` y `agenda`, además, sus disparadores—. Sin el import el registro está
# vacío y el núcleo arranca sin agentes sin decir por qué.
_ = (agenda, chat, correo, dev, memoria, pc, web)

logger = logging.getLogger("perseo_core")


def _configurar_registro() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )


def _tls(cfg: almacen.Configuracion) -> ssl.SSLContext | None:
    """El contexto para servir por HTTPS, o `None` para seguir en HTTP.

    Esto existe por el micrófono del móvil: el navegador solo deja grabar en un
    contexto seguro, y `http://` por el tailnet no lo es. El certificado lo da
    `tailscale cert`; con las rutas vacías —o apuntando a algo que ya no está—
    el núcleo arranca en HTTP como siempre, que es mejor que no arrancar.
    """
    if not cfg.tls_listo:
        if cfg.tls_certificado or cfg.tls_clave:
            logger.warning(
                "Hay certificado configurado pero no se encuentra (%s). Se sirve por HTTP.",
                cfg.tls_certificado or cfg.tls_clave,
            )
        return None
    contexto = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    contexto.load_cert_chain(cfg.tls_certificado, cfg.tls_clave)
    # Sin esto el servidor no contesta al ALPN que ofrece el cliente, y la pila
    # de red de iOS corta la conexión sin dar ninguna razón: desde el iPhone se
    # ve un "no se puede conectar" idéntico al de un puerto cerrado. Windows y
    # Python se lo tragan, así que el fallo solo aparece en el móvil — que es el
    # único sitio donde hace falta este HTTPS. Costó una mañana el 2026-08-17.
    # Solo `http/1.1`: aiohttp no habla HTTP/2, y ofrecer `h2` sería mentir.
    contexto.set_alpn_protocols(["http/1.1"])
    return contexto


def _donde_escuchar(
    cfg: almacen.Configuracion,
) -> list[tuple[str, int, ssl.SSLContext | None]]:
    """Qué se abre y cómo. El HTTPS **añade**, nunca sustituye.

    Se aprendió por las malas el 2026-08-17: al empezar a servir HTTPS en la
    interfaz del tailnet, `http://100.64.0.1:8787` —que es lo que tenían
    guardado el navegador del PC y el acceso directo del móvil— dejó de
    contestar. Un socket que habla TLS no puede contestar a quien llega en
    claro, así que aquello no fue un cambio de dirección: fue romperla.

    Por eso el puerto de siempre sigue siendo HTTP en todas las interfaces, y el
    HTTPS vive en uno propio. Nada de lo que funcionaba deja de funcionar, y el
    micrófono del móvil —que necesita contexto seguro— tiene por dónde entrar.

    Y el certificado va **solo** en el tailnet: lo emite Tailscale para
    `msi.taild61051.ts.net`, así que por `127.0.0.1` fallaría la verificación
    por nombre. Ahí entran la app de escritorio, el detector y `perseo estado`,
    y el bucle local ya cuenta como contexto seguro de todas formas.
    """
    sitios: list[tuple[str, int, ssl.SSLContext | None]] = [
        (host, cfg.puerto, None) for host in cfg.hosts
    ]

    contexto = _tls(cfg)
    if contexto is not None:
        sitios += [
            (host, cfg.tls_puerto, contexto)
            for host in cfg.hosts
            if host not in almacen.LOCALES
        ]
    return sitios


def _avisar_de_la_escucha(
    cfg: almacen.Configuracion, sitios: list[tuple[str, int, ssl.SSLContext | None]]
) -> None:
    for host, puerto, contexto in sitios:
        # Una IPv6 sin corchetes deja una línea que no se puede copiar y pegar:
        # `http://fd7a:...:8787` no es una URL válida.
        anfitrion = f"[{host}]" if ":" in host else host
        logger.info("Escuchando en %s://%s:%d", "https" if contexto else "http", anfitrion, puerto)
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


def _ya_contesta_otro_nucleo(cfg: almacen.Configuracion) -> bool:
    """¿Hay ya un núcleo vivo en el puerto? Entonces este sobra.

    Sin esta pregunta, un segundo núcleo hace todo el arranque —base de datos,
    MCP, trabajadores, Telegram— y muere al final con `OSError 10048` al intentar
    abrir un puerto que ya es de otro. El vigilante lo ve morir mal, lo vuelve a
    arrancar, y ahí queda el bucle: el 2026-08-24 dio seis vueltas en seis
    minutos escribiendo trazas de trescientas líneas. Peor todavía es lo que
    parece desde fuera —el núcleo «funciona», porque el viejo sigue en pie— y
    todo lo que se prueba va contra el código de antes.

    Un 401 vale igual que un 200: lo que se pregunta no es si nos dejan entrar,
    sino si hay alguien ahí.
    """
    import urllib.error
    import urllib.request

    url = f"http://127.0.0.1:{cfg.puerto}/salud"
    try:
        with urllib.request.urlopen(url, timeout=2) as respuesta:
            return respuesta.status < 500
    except urllib.error.HTTPError:
        return True
    except OSError:
        return False


async def arrancar() -> None:
    cfg = almacen.cargar_configuracion()

    if await asyncio.to_thread(_ya_contesta_otro_nucleo, cfg):
        # Salida limpia a propósito: para el vigilante, un código 0 es una orden
        # de retirarse (regla 2 de `commands/vigilante.py`), que es justo lo que
        # toca cuando el trabajo ya lo está haciendo otro.
        logger.warning(
            "Ya hay un núcleo contestando en el puerto %d. Este se retira sin "
            "tocar nada. Si lo que quieres es reiniciarlo: `perseo parar` y "
            "luego `perseo on`.",
            cfg.puerto,
        )
        return

    almacen.abrir(cfg)
    almacen.recuperar_huerfanos()
    # El semáforo del chat escrito vive en la base; al arrancar, nadie está a
    # mitad de un turno, así que todo libre.
    almacen.reiniciar_turnos_chat()

    # La política de §7 se aplica en el trabajador, así que tiene que estar en pie
    # antes de que ninguno reclame nada.
    politica.iniciar(cfg.directorio_datos)

    # Los servidores MCP no se arrancan aquí: se cargan sus definiciones y su
    # nivel de política, y cada proceso nace la primera vez que se le usa.
    await mcp.iniciar(cfg)

    bus = Bus()
    router = Router(cfg)
    await router.abrir()

    # Tres carriles. `dev` puede tardar minutos, y con un solo trabajador un
    # encargo de código dejaba el correo sin triar mientras durase. El chat
    # tiene el suyo porque un turno puede irse a los dos minutos entre
    # herramientas, y no debe frenar ni al triaje ni a la cola general.
    trabajador = Trabajador(bus, excluir=("dev", "chat"), nombre="general")
    tarea_trabajador = asyncio.create_task(trabajador.ejecutar(), name="trabajador")

    trabajador_dev = Trabajador(bus, agentes=("dev",), nombre="dev")
    tarea_dev = asyncio.create_task(trabajador_dev.ejecutar(), name="trabajador-dev")

    trabajador_chat = Trabajador(bus, agentes=("chat",), nombre="chat")
    tarea_chat = asyncio.create_task(trabajador_chat.ejecutar(), name="trabajador-chat")

    # El triaje del correo comparte una sola sesión contra Ollama entre trabajos.
    correo.iniciar(cfg)
    memoria.iniciar(cfg)
    dev.iniciar(cfg)
    web.iniciar(cfg)
    agenda.iniciar(cfg)
    chat.iniciar(cfg, router)

    # Sin token configurado se retira sola tras avisar: es un canal más.
    telegram = Telegram(cfg, bus)
    tarea_telegram = asyncio.create_task(telegram.ejecutar(), name="telegram")

    # Los disparadores que no tengan de dónde tirar se retiran solos.
    planificador = Planificador(cfg, bus)
    tarea_disparadores = asyncio.create_task(planificador.ejecutar(), name="disparadores")

    runner = servidor.AppRunner(api.crear_app(cfg, bus, router))
    await runner.setup()
    # Una lista de hosts, nunca `0.0.0.0`: se abren exactamente las interfaces
    # enumeradas y ninguna más. Y **un sitio por interfaz**, porque no todas se
    # sirven igual: ver `_tls_de`.
    sitios = _donde_escuchar(cfg)
    for host, puerto, contexto in sitios:
        sitio = servidor.TCPSite(runner, host, puerto, ssl_context=contexto)
        await sitio.start()
    _avisar_de_la_escucha(cfg, sitios)

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
        trabajador_dev.detener()
        trabajador_chat.detener()
        telegram.detener()
        planificador.detener()
        for tarea in (tarea_trabajador, tarea_dev, tarea_chat, tarea_telegram, tarea_disparadores):
            tarea.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await tarea
        await runner.cleanup()
        await router.cerrar()
        await correo.detener()
        await memoria.detener()
        dev.detener()
        await web.detener()
        await agenda.detener()
        await chat.detener()
        await mcp.detener()
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
