"""API HTTP + SSE del núcleo.

Esta es la pieza que **revierte una decisión de la Fase 3**, así que conviene
dejar escrito por qué. Aquella fase descartó HTTP con este argumento:

    "un servidor HTTP local es alcanzable por cualquier proceso de la máquina, y
    una de las herramientas que expondría es `controlar_pc`"

El argumento sigue siendo correcto. Lo que cambia es que el requisito R1 —texto
desde el iPhone— no se puede cumplir por tuberías. Así que se abre HTTP, pero
cerrando por diseño lo que la Fase 3 protegía:

1. **La API no expone herramientas.** Expone conversación y cola de trabajos.
   `controlar_pc` y compañía viven detrás del despachador interno; no hay ninguna
   ruta HTTP que llegue a ellas.
2. **No escucha en 0.0.0.0.** Por defecto solo en el bucle local; en la Fase B se
   apunta a la dirección de Tailscale. Un proceso cualquiera de la máquina no la
   ve en la interfaz del tailnet.
3. **Token en toda petición.** Sin token, 401. Comparación en tiempo constante.
4. Las acciones irreversibles seguirán exigiendo confirmación (Fase E, §7),
   aunque la petición venga autenticada.

Sobre la cookie: `EventSource` del navegador no permite cabeceras propias, así
que la alternativa habitual es meter el token en la URL. No se hace: las URLs
acaban en historiales y registros. En su lugar `POST /sesion` canjea el token por
una cookie HttpOnly y el flujo SSE se autentica con ella.

Ver bitacora/05_PLAN_PERSEO_V2.md §5 y §7, y bitacora/03_ROADMAP.md (Fase 3).
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from pathlib import Path
from typing import Any

from aiohttp import web

from . import almacen
from .agentes import REGISTRO, Router
from .bus import Bus

logger = logging.getLogger(__name__)

DIRECTORIO_WEB = Path(__file__).resolve().parent / "web"

CLAVE_CFG: web.AppKey[almacen.Configuracion] = web.AppKey("cfg")
CLAVE_BUS: web.AppKey[Bus] = web.AppKey("bus")
CLAVE_ROUTER: web.AppKey[Router] = web.AppKey("router")

COOKIE_SESION = "perseo_sesion"

#: Rutas que se sirven sin token. `/salud` porque es la comprobación de vida, y
#: la web porque el navegador no puede mandar una cabecera en la primera carga:
#: pide el token en pantalla y lo canjea por cookie contra `/sesion`. El HTML no
#: lleva nada dentro — sin cookie válida, todo lo que pide devuelve 401.
RUTAS_PUBLICAS = frozenset({"/salud", "/", "/manifest.webmanifest"})

#: Cada cuántos segundos se envía un comentario SSE para que ningún intermediario
#: cierre la conexión por inactividad.
LATIDO_SEGUNDOS = 20


# --------------------------------------------------------------------------- #
# Autenticación
# --------------------------------------------------------------------------- #


def _token_de_peticion(peticion: web.Request) -> str:
    cabecera = peticion.headers.get("Authorization", "")
    if cabecera.startswith("Bearer "):
        return cabecera[7:].strip()
    return peticion.cookies.get(COOKIE_SESION, "")


@web.middleware
async def _autenticar(peticion: web.Request, handler):
    # El parámetro se llama `handler` a propósito: aiohttp lo pasa por nombre,
    # así que renombrarlo a algo más castellano rompe el servidor en marcha.
    if peticion.path in RUTAS_PUBLICAS:
        return await handler(peticion)

    cfg = peticion.app[CLAVE_CFG]
    # compare_digest evita filtrar el token por diferencias de tiempo.
    if not secrets.compare_digest(_token_de_peticion(peticion), cfg.token):
        raise web.HTTPUnauthorized(
            text=json.dumps({"error": "Token ausente o incorrecto"}),
            content_type="application/json",
        )
    return await handler(peticion)


async def _abrir_sesion(peticion: web.Request) -> web.Response:
    """Canjea el token por una cookie, para que `EventSource` pueda autenticarse.

    El middleware ya ha validado el token antes de llegar aquí.
    """
    cfg = peticion.app[CLAVE_CFG]
    respuesta = web.json_response({"ok": True})
    respuesta.set_cookie(
        COOKIE_SESION,
        cfg.token,
        httponly=True,
        samesite="Strict",
        max_age=60 * 60 * 24 * 30,
        # `secure` queda fuera a propósito: dentro del tailnet se sirve por HTTP
        # plano y el tráfico ya va cifrado por WireGuard.
    )
    return respuesta


# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #


async def _salud(peticion: web.Request) -> web.Response:
    """Comprobación de vida. Pública a propósito, y sin nada sensible dentro."""
    router = peticion.app[CLAVE_ROUTER]
    recuento = await asyncio.to_thread(almacen.recuento_por_estado)
    return web.json_response(
        {
            "ok": True,
            "servicio": "perseo-core",
            "fase": "D",
            "agentes": sorted(REGISTRO),
            "router_local": router.disponible,
            "trabajos": recuento,
        }
    )


async def _indice(peticion: web.Request) -> web.FileResponse:
    """La web del núcleo: chat, cola de trabajos y aprobaciones."""
    return web.FileResponse(
        DIRECTORIO_WEB / "index.html",
        headers={"Cache-Control": "no-cache"},
    )


#: Manifiesto de PWA, en línea para no depender de un fichero más. Sin icono
#: propio todavía: iOS usa el `apple-touch-icon` y ninguno de los dos hace falta
#: para guardar la página en la pantalla de inicio.
_MANIFIESTO = {
    "name": "Perseo",
    "short_name": "Perseo",
    "start_url": "/",
    "display": "standalone",
    "background_color": "#101014",
    "theme_color": "#101014",
}


async def _manifiesto(peticion: web.Request) -> web.Response:
    return web.json_response(_MANIFIESTO, content_type="application/manifest+json")


async def _cuerpo_json(peticion: web.Request) -> dict[str, Any]:
    try:
        datos = await peticion.json()
    except json.JSONDecodeError:
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "El cuerpo no es JSON válido"}),
            content_type="application/json",
        )
    if not isinstance(datos, dict):
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "Se esperaba un objeto JSON"}),
            content_type="application/json",
        )
    return datos


async def _mensaje(peticion: web.Request) -> web.Response:
    """Entrada conversacional: el router decide si se contesta ya o se encola."""
    datos = await _cuerpo_json(peticion)
    texto = str(datos.get("texto", "")).strip()
    if not texto:
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "Falta 'texto'"}), content_type="application/json"
        )

    origen = datos.get("origen", "texto")
    if origen not in almacen.ORIGENES:
        raise web.HTTPBadRequest(
            text=json.dumps({"error": f"Origen inválido. Válidos: {list(almacen.ORIGENES)}"}),
            content_type="application/json",
        )

    router = peticion.app[CLAVE_ROUTER]
    bus = peticion.app[CLAVE_BUS]

    ruta = await router.decidir(texto)
    if not ruta.hay_que_encolar:
        bus.publicar("mensaje.respondido", texto=texto, respuesta=ruta.respuesta)
        return web.json_response(
            {"destino": "responder", "respuesta": ruta.respuesta, "motivo": ruta.motivo}
        )

    trabajo = await asyncio.to_thread(almacen.encolar, ruta.agente, {"texto": texto}, origen)
    bus.publicar("trabajo.encolado", trabajo=trabajo)
    return web.json_response(
        {"destino": "encolar", "motivo": ruta.motivo, "trabajo": trabajo}, status=202
    )


async def _crear_trabajo(peticion: web.Request) -> web.Response:
    """Encola directamente, saltándose el router. Para disparadores y pruebas."""
    datos = await _cuerpo_json(peticion)
    agente = str(datos.get("agente", "")).strip()
    if agente not in REGISTRO:
        raise web.HTTPBadRequest(
            text=json.dumps({"error": f"Agente desconocido. Disponibles: {sorted(REGISTRO)}"}),
            content_type="application/json",
        )

    peticion_agente = datos.get("peticion") or {}
    if not isinstance(peticion_agente, dict):
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "'peticion' debe ser un objeto"}),
            content_type="application/json",
        )

    origen = datos.get("origen", "texto")
    try:
        trabajo = await asyncio.to_thread(almacen.encolar, agente, peticion_agente, origen)
    except ValueError as e:
        raise web.HTTPBadRequest(
            text=json.dumps({"error": str(e)}), content_type="application/json"
        )

    peticion.app[CLAVE_BUS].publicar("trabajo.encolado", trabajo=trabajo)
    return web.json_response(trabajo, status=201)


async def _listar_trabajos(peticion: web.Request) -> web.Response:
    estado = peticion.query.get("estado")
    try:
        limite = int(peticion.query.get("limite", "50"))
    except ValueError:
        limite = 50
    trabajos = await asyncio.to_thread(almacen.listar, estado, limite)
    return web.json_response({"trabajos": trabajos})


def _id_de_ruta(peticion: web.Request) -> int:
    try:
        return int(peticion.match_info["id"])
    except (KeyError, ValueError):
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "Identificador inválido"}),
            content_type="application/json",
        )


async def _ver_trabajo(peticion: web.Request) -> web.Response:
    trabajo = await asyncio.to_thread(almacen.obtener, _id_de_ruta(peticion))
    if trabajo is None:
        raise web.HTTPNotFound(
            text=json.dumps({"error": "No existe ese trabajo"}),
            content_type="application/json",
        )
    return web.json_response(trabajo)


async def _cancelar_trabajo(peticion: web.Request) -> web.Response:
    id_trabajo = _id_de_ruta(peticion)
    actual = await asyncio.to_thread(almacen.obtener, id_trabajo)
    if actual is None:
        raise web.HTTPNotFound(
            text=json.dumps({"error": "No existe ese trabajo"}),
            content_type="application/json",
        )
    if actual["estado"] not in almacen.ABIERTOS:
        raise web.HTTPConflict(
            text=json.dumps({"error": f"El trabajo ya está {actual['estado']}"}),
            content_type="application/json",
        )

    trabajo = await asyncio.to_thread(almacen.cancelar, id_trabajo)
    peticion.app[CLAVE_BUS].publicar("trabajo.cancelado", trabajo=trabajo)
    return web.json_response(trabajo)


async def _responder_confirmacion(peticion: web.Request) -> web.Response:
    """Aprueba o rechaza un trabajo parado a la espera de un sí.

    La misma pregunta puede estar abierta en la web y en el móvil a la vez, así
    que quien resuelve es el UPDATE condicional del almacén: el segundo en
    llegar se encuentra el trabajo ya movido y recibe un 409. Contestar dos
    veces no ejecuta la acción dos veces.
    """
    id_trabajo = _id_de_ruta(peticion)
    aprobado = peticion.match_info["decision"] == "aprobar"

    trabajo = await asyncio.to_thread(almacen.resolver_confirmacion, id_trabajo, aprobado)
    if trabajo is None:
        actual = await asyncio.to_thread(almacen.obtener, id_trabajo)
        if actual is None:
            raise web.HTTPNotFound(
                text=json.dumps({"error": "No existe ese trabajo"}),
                content_type="application/json",
            )
        raise web.HTTPConflict(
            text=json.dumps(
                {"error": f"El trabajo no está esperando confirmación (está {actual['estado']})"}
            ),
            content_type="application/json",
        )

    peticion.app[CLAVE_BUS].publicar(
        "trabajo.aprobado" if aprobado else "trabajo.rechazado", trabajo=trabajo
    )
    return web.json_response(trabajo)


async def _eventos(peticion: web.Request) -> web.StreamResponse:
    """Flujo SSE con todo lo que pasa en el núcleo."""
    respuesta = web.StreamResponse(
        headers={
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )
    await respuesta.prepare(peticion)

    bus = peticion.app[CLAVE_BUS]

    async def latir() -> None:
        # Tarea aparte, en lugar de un `wait_for` sobre el flujo: cancelar la
        # espera de la cola en cada latido podría perder un evento por carrera.
        while True:
            await asyncio.sleep(LATIDO_SEGUNDOS)
            await respuesta.write(b": latido\n\n")

    tarea_latido = asyncio.create_task(latir())
    try:
        async with bus.suscribir() as eventos:
            await respuesta.write(b": conectado\n\n")
            async for evento in eventos:
                carga = json.dumps(evento.a_dict(), ensure_ascii=False)
                await respuesta.write(f"data: {carga}\n\n".encode("utf-8"))
    except (ConnectionResetError, asyncio.CancelledError):
        # El cliente se fue (pantalla apagada, túnel caído). No es un error.
        pass
    finally:
        tarea_latido.cancel()

    return respuesta


# --------------------------------------------------------------------------- #
# Construcción
# --------------------------------------------------------------------------- #


def crear_app(cfg: almacen.Configuracion, bus: Bus, router: Router) -> web.Application:
    app = web.Application(middlewares=[_autenticar])
    app[CLAVE_CFG] = cfg
    app[CLAVE_BUS] = bus
    app[CLAVE_ROUTER] = router

    app.add_routes(
        [
            web.get("/", _indice),
            web.get("/manifest.webmanifest", _manifiesto),
            web.get("/salud", _salud),
            web.post("/sesion", _abrir_sesion),
            web.post("/mensaje", _mensaje),
            web.post("/trabajos", _crear_trabajo),
            web.get("/trabajos", _listar_trabajos),
            web.get("/trabajos/{id}", _ver_trabajo),
            web.post("/trabajos/{id}/cancelar", _cancelar_trabajo),
            web.post("/trabajos/{id}/{decision:aprobar|rechazar}", _responder_confirmacion),
            web.get("/eventos", _eventos),
        ]
    )
    return app
