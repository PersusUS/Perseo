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
import contextlib
import json
import logging
import math
import secrets
from pathlib import Path
from typing import Any

from aiohttp import web

from . import almacen, biometria, dev, estado, grafo, habitos, politica, proyectos, tareas
from .agentes import REGISTRO, Router
from .bus import Bus

logger = logging.getLogger(__name__)

# La carpeta se llama `interfaz` y no `web` por una razón concreta: desde la
# Fase E hay un `web.py` —el agente que lee páginas— en este mismo paquete. Un
# módulo y un directorio con el mismo nombre conviven hasta que alguien añade un
# `__init__.py` a la carpeta, y entonces el agente desaparece del registro sin
# que nada avise. Se ha renombrado antes de que pasara.
DIRECTORIO_WEB = Path(__file__).resolve().parent / "interfaz"

CLAVE_CFG: web.AppKey[almacen.Configuracion] = web.AppKey("cfg")
CLAVE_BUS: web.AppKey[Bus] = web.AppKey("bus")
CLAVE_ROUTER: web.AppKey[Router] = web.AppKey("router")

COOKIE_SESION = "perseo_sesion"

#: Rutas que se sirven sin token. `/salud` porque es la comprobación de vida, y
#: la web porque el navegador no puede mandar una cabecera en la primera carga:
#: pide el token en pantalla y lo canjea por cookie contra `/sesion`. El HTML no
#: lleva nada dentro — sin cookie válida, todo lo que pide devuelve 401.
#: Los iconos también van sin token: los pide el sistema operativo al guardar
#: la página en la pantalla de inicio, y esas peticiones no llevan cookie. Sin
#: esto se llevan un 401 y iOS pone una captura de la página como icono.
#: El fondo y el avatar también: se ven en la pantalla que pide el token, que
#: por definición es la que se mira **antes** de tener cookie. Con token serían
#: dos huecos negros justo donde se comprueba que has llegado al sitio correcto.
#: Son la misma ola y la misma cara que la app de escritorio, y no dicen nada de
#: nadie.
RUTAS_PUBLICAS = frozenset(
    {
        "/salud",
        "/",
        "/manifest.webmanifest",
        "/hokusai-bg.png",
        "/perseo-avatar.jpg",
        "/icono-180.png",
        "/icono-512.png",
        "/apple-touch-icon.png",
        "/apple-touch-icon-precomposed.png",
        "/favicon.ico",
    }
)

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
    en_cookie = peticion.cookies.get(COOKIE_SESION, "")
    if en_cookie:
        return en_cookie
    # La ventana del grafo no puede mandar cabecera en su primera carga y no
    # lleva sesión canjeada: el token viaja en `?t=`, igual que la PWA canjea
    # el suyo por cookie. Sigue SIENDO token — quien no lo tenga, 401.
    return peticion.query.get("t", "")


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
            "fase": "E",
            "agentes": sorted(REGISTRO),
            "router_local": router.disponible,
            "trabajos": recuento,
        }
    )


async def _estado(peticion: web.Request) -> web.Response:
    """De qué está capado el sistema hoy: Ollama, Obsidian, Google, cuota, cola.

    **Va con token, y no en `/salud`.** Lo que hay aquí dentro es el mapa de por
    dónde entrar: qué credenciales están puestas, qué modelo se usa, qué canal
    avisa y si ahora mismo lo irreversible se ejecuta sin preguntar. `/salud`
    sigue siendo pública porque no dice nada de eso.
    """
    return web.json_response(
        await estado.reunir(peticion.app[CLAVE_CFG], peticion.app[CLAVE_ROUTER])
    )


async def _indice(peticion: web.Request) -> web.FileResponse:
    """La web del núcleo: chat, cola de trabajos y aprobaciones."""
    return web.FileResponse(
        DIRECTORIO_WEB / "index.html",
        headers={"Cache-Control": "no-cache"},
    )


async def _pagina_grafo(peticion: web.Request) -> web.FileResponse:
    """La ventana del grafo del segundo cerebro: el vault como constelación.

    Un solo fichero, sin build — la misma escuela que la PWA. El token llegó
    en `?t=` (el middleware ya lo validó) y la página lo reusa para pedir los
    datos; no se guarda en ningún sitio.
    """
    return web.FileResponse(
        DIRECTORIO_WEB / "grafo.html",
        headers={"Cache-Control": "no-cache"},
    )


async def _datos_grafo(peticion: web.Request) -> web.Response:
    """El grafo del vault: notas como nodos, enlaces `[[...]]` como aristas."""
    from . import memoria  # perezoso, como en `estado`: nada de cargarlo por defecto

    cfg = peticion.app[CLAVE_CFG]
    datos = await asyncio.to_thread(grafo.construir, memoria.ruta_vault(cfg))
    return web.json_response(datos)


async def _abrir_nota_grafo(peticion: web.Request) -> web.Response:
    """Abre en Obsidian la nota de un nodo del grafo.

    Por aquí llega **cuál** de las notas del vault y nada más — el id se busca
    entre los ficheros reales y lo que se abre es su URI `obsidian://`, igual
    que `proyectos` solo abre lo que ya vive escrito en su fichero del disco.
    """
    try:
        cuerpo = await peticion.json()
        id_nota = str(cuerpo.get("id", ""))
    except (json.JSONDecodeError, TypeError, AttributeError):
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "Cuerpo inválido"}), content_type="application/json"
        )

    from . import memoria

    cfg = peticion.app[CLAVE_CFG]
    resultado = await asyncio.to_thread(
        grafo.abrir_nota, memoria.ruta_vault(cfg), id_nota
    )
    if resultado.startswith("Error:"):
        raise web.HTTPBadRequest(
            text=json.dumps({"error": resultado}), content_type="application/json"
        )
    return web.json_response({"resultado": resultado})


#: Manifiesto de PWA, en línea para no depender de un fichero más. Los iconos
#: sí son ficheros: son la ola de Hokusai recortada en cuadrado, la misma imagen
#: que la app usa de fondo, para que el icono del móvil y la app de escritorio
#: sean reconociblemente lo mismo.
_MANIFIESTO = {
    "name": "Perseo",
    "short_name": "Perseo",
    "start_url": "/",
    "display": "standalone",
    # El mismo negro que la app de voz: la PWA abre a pantalla completa en el
    # iPhone y un fondo distinto se ve como un parpadeo al arrancar.
    "background_color": "#000000",
    "theme_color": "#000000",
    "icons": [
        {"src": "/icono-180.png", "sizes": "180x180", "type": "image/png"},
        {"src": "/icono-512.png", "sizes": "512x512", "type": "image/png"},
        # `maskable` deja que Android lo recorte a su forma sin comerse la ola.
        {"src": "/icono-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable"},
    ],
}


async def _icono(peticion: web.Request) -> web.FileResponse:
    """Los iconos de la PWA. iOS pide `apple-touch-icon.png` por su cuenta.

    El sistema los pide **sin cookie**, así que están en `RUTAS_PUBLICAS`: un
    icono no es información, y devolver 401 aquí es lo que dejaba la pantalla de
    inicio con una captura en vez de la ola.
    """
    nombre = peticion.path.lstrip("/")
    if nombre.startswith("apple-touch-icon"):
        nombre = "icono-180.png"
    return web.FileResponse(
        DIRECTORIO_WEB / nombre,
        headers={"Cache-Control": "max-age=86400"},
    )


async def _arte(peticion: web.Request) -> web.FileResponse:
    """El fondo y el avatar de la interfaz. Mismo criterio que los iconos.

    Son ficheros y no CSS incrustado porque pesan 190 KB entre los dos: dentro
    del HTML se descargarían enteros en cada carga, y por el túnel de Tailscale
    eso se nota. Como fichero aparte los cachea el navegador.
    """
    return web.FileResponse(
        DIRECTORIO_WEB / peticion.path.lstrip("/"),
        headers={"Cache-Control": "max-age=604800"},
    )


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


async def _ver_confianza(peticion: web.Request) -> web.Response:
    """Si el modo confianza está encendido y hasta cuándo.

    Va autenticado, no en `/salud`: saber si ahora mismo lo irreversible se
    ejecuta sin preguntar es justo lo que no debe contarse sin token.
    """
    hasta = politica.confianza_hasta()
    return web.json_response(
        {"confianza": hasta is not None, "hasta": hasta.isoformat() if hasta else None}
    )


async def _cambiar_confianza(peticion: web.Request) -> web.Response:
    """Enciende o apaga el modo confianza (§7 del plan).

    Encendido, lo irreversible deja de pedir un sí mientras dura. Caduca solo: un
    interruptor que se queda puesto para siempre es lo que la política evita.
    """
    datos = await _cuerpo_json(peticion)
    if datos.get("activo") is False:
        politica.desactivar_confianza()
        return web.json_response({"confianza": False, "hasta": None})

    try:
        minutos = float(datos.get("minutos", politica.MINUTOS_CONFIANZA))
    except (TypeError, ValueError):
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "'minutos' debe ser un número"}),
            content_type="application/json",
        )
    if not math.isfinite(minutos):
        # `NaN` e infinitos atraviesan el `float()` y, sin este guardo, el NaN
        # acababa recortado a "un minuto de confianza" en vez de rechazarse.
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "'minutos' debe ser un número finito"}),
            content_type="application/json",
        )

    hasta = await asyncio.to_thread(politica.activar_confianza, minutos)
    peticion.app[CLAVE_BUS].publicar("confianza.cambiada", hasta=hasta.isoformat())
    return web.json_response({"confianza": True, "hasta": hasta.isoformat()})


async def _listar_trabajos(peticion: web.Request) -> web.Response:
    estado = peticion.query.get("estado")
    try:
        limite = int(peticion.query.get("limite", "50"))
    except ValueError:
        limite = 50
    trabajos = await asyncio.to_thread(almacen.listar, estado, limite)
    return web.json_response({"trabajos": [_con_progreso(t) for t in trabajos]})


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
    return web.json_response(_con_progreso(trabajo))


async def _ver_actividad(peticion: web.Request) -> web.Response:
    """El paso a paso de un encargo de código: el suyo y el de sus subagentes.

    Es lo que convierte «HECHO (16 vueltas)» en algo que se puede depurar. La
    bitácora sobrevive al encargo —y al reinicio del núcleo, porque también se
    escribe en disco—, así que esto contesta igual a los cinco minutos que a la
    mañana siguiente, que es cuando uno se pregunta qué hizo de verdad.
    """
    id_trabajo = _id_de_ruta(peticion)
    trabajo = await asyncio.to_thread(almacen.obtener, id_trabajo)
    if trabajo is None:
        raise web.HTTPNotFound(
            text=json.dumps({"error": "No existe ese trabajo"}),
            content_type="application/json",
        )
    actividad = await asyncio.to_thread(dev.actividad_de, id_trabajo)
    return web.json_response({**actividad, "estado": trabajo.get("estado")})


def _con_progreso(trabajo: dict[str, Any]) -> dict[str, Any]:
    """Añade por dónde va el encargo, si es uno de código y sigue vivo.

    Un encargo de `dev` tarda minutos. Sin esto, la pantalla enseña «en curso»
    y nada más durante todo ese rato; con esto dice «Editando api.py». El dato
    vive en memoria del núcleo y solo lo sabe el motor sobre el SDK: los que
    hablan por consola no cuentan nada hasta el final, y entonces el campo no
    aparece — que es distinto de aparecer vacío.
    """
    if trabajo.get("agente") != "dev" or trabajo.get("estado") != almacen.EN_CURSO:
        return trabajo
    paso = dev.progreso_de(int(trabajo["id"]))
    return {**trabajo, "progreso": paso} if paso else trabajo


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


async def _listar_correos(peticion: web.Request) -> web.Response:
    """Qué se ha hecho con cada correo triado. Lo que no salga está pendiente."""
    marcados = await asyncio.to_thread(almacen.correos_marcados)
    return web.json_response({"marcados": marcados})


async def _marcar_correo(peticion: web.Request) -> web.Response:
    """Mueve un correo a atendido, descartado o de vuelta a pendiente.

    Es el único sitio donde el triaje deja de ser de solo lectura. No toca
    Gmail: aquí se anota lo que **tú** has hecho, y marcar leído en el buzón es
    otra cosa que además necesitaría un permiso que el testigo no tiene.
    """
    datos = await _cuerpo_json(peticion)
    estado = str(datos.get("estado", "")).strip().lower()
    if estado not in almacen.ESTADOS_CORREO:
        raise web.HTTPBadRequest(
            text=json.dumps({"error": f"Estado inválido. Válidos: {list(almacen.ESTADOS_CORREO)}"}),
            content_type="application/json",
        )

    id_mensaje = peticion.match_info["id"]
    try:
        marcado = await asyncio.to_thread(almacen.marcar_correo, id_mensaje, estado)
    except ValueError as e:
        raise web.HTTPBadRequest(
            text=json.dumps({"error": str(e)}), content_type="application/json"
        )

    peticion.app[CLAVE_BUS].publicar("correo.marcado", correo=marcado)
    return web.json_response(marcado)


# --------------------------------------------------------------------------- #
# El chat escrito
# --------------------------------------------------------------------------- #


async def _chat_sesiones(peticion: web.Request) -> web.Response:
    return web.json_response(
        {"sesiones": await asyncio.to_thread(almacen.sesiones_chat)}
    )


async def _crear_sesion_chat(peticion: web.Request) -> web.Response:
    datos = await _cuerpo_json(peticion)
    sesion = await asyncio.to_thread(almacen.crear_sesion_chat, str(datos.get("titulo", "")))
    peticion.app[CLAVE_BUS].publicar("chat.sesion", sesion=sesion)
    return web.json_response(sesion, status=201)


async def _ver_sesion_chat(peticion: web.Request) -> web.Response:
    """Una conversación, con su semáforo. Es lo que sondean las caras mientras
    `turno` está `ocupado`: el texto de la respuesta va creciendo en el último
    mensaje, que llega con estado `escribiendo` hasta que se cierra el turno."""
    id_sesion = _id_de_ruta(peticion)
    sesion = await asyncio.to_thread(almacen.obtener_sesion_chat, id_sesion)
    if sesion is None:
        raise web.HTTPNotFound(
            text=json.dumps({"error": "No existe esa conversación"}),
            content_type="application/json",
        )
    mensajes = await asyncio.to_thread(almacen.mensajes_chat, id_sesion)
    return web.json_response({**sesion, "mensajes": mensajes})


async def _borrar_sesion_chat(peticion: web.Request) -> web.Response:
    id_sesion = _id_de_ruta(peticion)
    try:
        borrada = await asyncio.to_thread(almacen.borrar_sesion_chat, id_sesion)
    except ValueError as e:
        raise web.HTTPConflict(
            text=json.dumps({"error": str(e)}), content_type="application/json"
        )
    if not borrada:
        raise web.HTTPNotFound(
            text=json.dumps({"error": "No existe esa conversación"}),
            content_type="application/json",
        )
    peticion.app[CLAVE_BUS].publicar("chat.borrado", sesion={"id": id_sesion})
    return web.json_response({"ok": True})


async def _hablar_chat(peticion: web.Request) -> web.Response:
    """Encola un turno de conversación y contesta al momento.

    La cara queda sondeando `GET /chat/{id}`; aquí solo se apuntan los dos
    mensajes —el del usuario y el de Perseo, vacío y `escribiendo`— y se pone
    el trabajo en la cola, como todo lo demás.
    """
    id_sesion = _id_de_ruta(peticion)
    datos = await _cuerpo_json(peticion)
    texto = str(datos.get("texto", "")).strip()
    if not texto:
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "Falta 'texto'"}), content_type="application/json"
        )

    sesion = await asyncio.to_thread(almacen.obtener_sesion_chat, id_sesion)
    if sesion is None:
        raise web.HTTPNotFound(
            text=json.dumps({"error": "No existe esa conversación"}),
            content_type="application/json",
        )

    try:
        await asyncio.to_thread(almacen.marcar_turno_chat, id_sesion, "ocupado")
    except ValueError as e:
        raise web.HTTPConflict(
            text=json.dumps({"error": str(e)}), content_type="application/json"
        )

    try:
        id_usuario = await asyncio.to_thread(almacen.anadir_mensaje_chat, id_sesion, "usuario", texto)
        id_perseo = await asyncio.to_thread(
            almacen.anadir_mensaje_chat, id_sesion, "perseo", "", "escribiendo"
        )
        trabajo = await asyncio.to_thread(
            almacen.encolar,
            "chat",
            {"sesion": id_sesion, "mensaje": id_perseo, "texto": texto},
            "texto",
        )
    except Exception:
        # Sin turno no hay respuesta: si encolar falla, hay que devolver el
        # semáforo o la sesión quedaría ocupada para siempre.
        with contextlib.suppress(Exception):
            await asyncio.to_thread(almacen.marcar_turno_chat, id_sesion, "libre")
        raise

    peticion.app[CLAVE_BUS].publicar("trabajo.encolado", trabajo=trabajo)
    return web.json_response(
        {
            "sesion": id_sesion,
            "mensaje_usuario": id_usuario,
            "mensaje_id": id_perseo,
            "trabajo_id": trabajo["id"],
        },
        status=202,
    )


async def _habitos_espejo(peticion: web.Request) -> web.Response:
    """La ventana deja aquí su copia del seguimiento de hábitos.

    El seguimiento vive en el `localStorage` de la app, y esta ruta es cómo el
    núcleo se entera de él: sin ella, el chat escrito y los agentes son los
    únicos de la casa que no saben cómo van los hábitos del señor Persus.

    El cuerpo trae el texto YA redactado por la ventana. El núcleo no vuelve a
    contar nada (ver `habitos.py`): dos contabilidades del mismo dato acaban
    discrepando, y entonces ninguna de las dos vale.
    """
    cuerpo = await _cuerpo_json(peticion)
    texto = str(cuerpo.get("texto", "")).strip()
    if not texto:
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "Falta 'texto'"}), content_type="application/json"
        )
    cfg = peticion.app[CLAVE_CFG]
    foto = cuerpo.get("foto")
    copia = await asyncio.to_thread(
        habitos.guardar, cfg.directorio_datos, texto, foto if isinstance(foto, dict) else None
    )
    return web.json_response({"sellado": copia["sellado"]})


async def _habitos_ver(peticion: web.Request) -> web.Response:
    """Lo último que mandó la ventana, con el aviso delante si viene vieja."""
    cfg = peticion.app[CLAVE_CFG]
    return web.json_response(
        {"resumen": await asyncio.to_thread(habitos.resumen, cfg.directorio_datos)}
    )


async def _tareas_espejo(peticion: web.Request) -> web.Response:
    """La ventana deja aquí su copia del tablero de tareas.

    Gemela de `_habitos_espejo`: el corcho vive en el `localStorage` de la app y
    esta ruta es cómo el núcleo se entera de él. El cuerpo trae el texto YA
    redactado por la ventana, y el núcleo no vuelve a contar nada (ver
    `tareas.py`): dos contabilidades del mismo tablero acaban discrepando, y
    entonces ninguna de las dos vale.
    """
    cuerpo = await _cuerpo_json(peticion)
    texto = str(cuerpo.get("texto", "")).strip()
    if not texto:
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "Falta 'texto'"}), content_type="application/json"
        )
    cfg = peticion.app[CLAVE_CFG]
    foto = cuerpo.get("foto")
    copia = await asyncio.to_thread(
        tareas.guardar, cfg.directorio_datos, texto, foto if isinstance(foto, dict) else None
    )
    return web.json_response({"sellado": copia["sellado"]})


async def _tareas_ver(peticion: web.Request) -> web.Response:
    """Lo último que mandó la ventana, con el aviso delante si viene vieja."""
    cfg = peticion.app[CLAVE_CFG]
    return web.json_response(
        {"resumen": await asyncio.to_thread(tareas.resumen, cfg.directorio_datos)}
    )


async def _tareas_recoger(peticion: web.Request) -> web.Response:
    """La ventana recoge lo que Perseo le pidió hacer con el tablero.

    Va por POST y no por GET porque vacía la cola: una ruta que cambia el estado
    del servidor no puede ser una lectura, aunque lo que devuelva se parezca a
    una. Ver `tareas.recoger` para por qué se entrega sin acuse de recibo.
    """
    cfg = peticion.app[CLAVE_CFG]
    pendientes = await asyncio.to_thread(tareas.recoger, cfg.directorio_datos)
    return web.json_response({"ordenes": pendientes})


async def _biometria_estado(peticion: web.Request) -> web.Response:
    """Perfiles, progreso de aprendizaje y qué motores hay hoy.

    Va autenticado como todo: los nombres de los perfiles son gente real, y la
    lista de quién conoces no se le enseña a nadie sin token.
    """
    cfg = peticion.app[CLAVE_CFG]
    return web.json_response(
        await asyncio.to_thread(biometria.estado_completo, cfg.directorio_datos)
    )


async def _biometria_voz(peticion: web.Request) -> web.Response:
    """Un trozo de PCM 16k mono (base64) entra, un nombre o un progreso sale.

    Es la ruta que llama la app de voz con el mismo micrófono que ya alimenta
    a Gemini. Cuando aquí nace un perfil nuevo —un desconocido que por fin
    acumuló voz suficiente— se publica al bus, para que quien escuche sepa que
    hay alguien nuevo en la casa.
    """
    cuerpo = await _cuerpo_json(peticion)
    audio = str(cuerpo.get("audio", ""))
    if not audio:
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "Falta 'audio'"}), content_type="application/json"
        )

    cfg = peticion.app[CLAVE_CFG]
    resultado = await asyncio.to_thread(
        biometria.identificar_voz, cfg.directorio_datos, audio
    )
    if resultado.get("aprendido"):
        peticion.app[CLAVE_BUS].publicar(
            "biometria.perfil",
            nombre=resultado.get("nombre"),
            via="voz",
        )
    return web.json_response(resultado)


async def _biometria_cara(peticion: web.Request) -> web.Response:
    """Un JPEG (base64) entra; caras con nombre y caja salen.

    Igual que la voz: cuando una cara desconocida se fija como perfil, evento.
    """
    cuerpo = await _cuerpo_json(peticion)
    imagen = str(cuerpo.get("imagen", ""))
    if not imagen:
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "Falta 'imagen'"}), content_type="application/json"
        )

    cfg = peticion.app[CLAVE_CFG]
    resultado = await asyncio.to_thread(
        biometria.identificar_cara, cfg.directorio_datos, imagen
    )
    for cara in resultado.get("caras", []):
        if cara.get("aprendido"):
            peticion.app[CLAVE_BUS].publicar(
                "biometria.perfil", nombre=cara.get("nombre"), via="cara"
            )
    return web.json_response(resultado)


async def _biometria_enrolar(peticion: web.Request) -> web.Response:
    """Crea o refuerza un perfil con una muestra traída a propósito."""
    cuerpo = await _cuerpo_json(peticion)
    nombre = str(cuerpo.get("nombre", ""))
    audio = cuerpo.get("audio")
    imagen = cuerpo.get("imagen")
    if not nombre:
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "Falta 'nombre'"}), content_type="application/json"
        )
    if not audio and not imagen:
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "Hace falta 'audio' o 'imagen'"}),
            content_type="application/json",
        )

    cfg = peticion.app[CLAVE_CFG]
    resultado = await asyncio.to_thread(
        biometria.enrolar,
        cfg.directorio_datos,
        nombre,
        str(audio) if audio else None,
        str(imagen) if imagen else None,
    )
    if resultado.get("ok") and resultado.get("añadido"):
        peticion.app[CLAVE_BUS].publicar(
            "biometria.perfil", nombre=nombre, via="+".join(resultado["añadido"])
        )
    estado_http = 200 if resultado.get("ok") else 400
    return web.json_response(resultado, status=estado_http)


async def _biometria_renombrar(peticion: web.Request) -> web.Response:
    """Le pone nombre real a un «Desconocido N»."""
    cuerpo = await _cuerpo_json(peticion)
    cfg = peticion.app[CLAVE_CFG]
    resultado = await asyncio.to_thread(
        biometria.renombrar,
        cfg.directorio_datos,
        peticion.match_info["nombre"],
        str(cuerpo.get("nuevo_nombre", "")),
    )
    if resultado.get("ok"):
        peticion.app[CLAVE_BUS].publicar(
            "biometria.perfil", nombre=resultado.get("nombre"), via="renombrado"
        )
        # Ponerle nombre a un «Desconocido 3» es el momento en que esa voz pasa
        # a ser alguien. Los vectores no dicen nada a un humano; la nota sí, y
        # se puede corregir a mano. No bloquea la respuesta ni la tumba: si el
        # vault no está, se apunta y se sigue.
        asyncio.create_task(
            _anotar_persona(str(peticion.match_info["nombre"]), str(resultado["nombre"]))
        )
    estado_http = 200 if resultado.get("ok") else 400
    return web.json_response(resultado, status=estado_http)


async def _anotar_persona(antes: str, ahora: str) -> None:
    """Deja en `10_PERSEO/Personas/` que esta voz o esta cara ya tiene nombre."""
    from . import memoria

    try:
        await memoria.anotar_persona(antes, ahora)
    except Exception as e:  # noqa: BLE001 - un apunte que falla no rompe nada
        logger.warning("No se pudo anotar a %s en el vault: %s", ahora, e)


async def _biometria_borrar(peticion: web.Request) -> web.Response:
    """Borra el perfil y sus vectores. No hay copia: eso es lo pedido."""
    cfg = peticion.app[CLAVE_CFG]
    resultado = await asyncio.to_thread(
        biometria.borrar, cfg.directorio_datos, peticion.match_info["nombre"]
    )
    if resultado.get("ok"):
        peticion.app[CLAVE_BUS].publicar(
            "biometria.perfil", nombre=peticion.match_info["nombre"], via="borrado"
        )
    estado_http = 200 if resultado.get("ok") else 400
    return web.json_response(resultado, status=estado_http)


# La ruta `/clave-voz` vivía aquí y se fue con la pestaña Voz del móvil
# (T-9, 2026-08-21). Entregaba la clave de Gemini por la red para que el
# navegador del teléfono hablara directamente con el modelo, y estaba escrito
# que sería «la primera ruta que hay que quitar» el día que el núcleo se abriera
# a algo que no fuese el tailnet. Se ha quitado antes: ya no la usa nadie, y una
# clave que no se sirve no se puede filtrar. El modelo de voz lo elige la app de
# escritorio, que lee su clave del disco por Rust.


def _carga_proyectos(directorio_datos):
    """La lista para la pantalla, con el estado vivo de cada servicio.

    Una ficha que dijera «LANZAR» de una app ya en marcha mentiría, así que a
    cada proyecto de modo `servicio` se le pregunta si su puerto respira. Va
    todo dentro del hilo de trabajo: son sondeos locales de décimas.
    """
    salida = []
    for proyecto in proyectos.listar(directorio_datos):
        datos = proyecto.a_dict()
        if proyecto.modo == "servicio":
            datos["vivo"] = proyectos.puerto_responde(proyecto.destino)
        salida.append(datos)
    return salida


async def _listar_proyectos(peticion: web.Request) -> web.Response:
    """Los otros proyectos que se pueden abrir desde el panel."""
    cfg = peticion.app[CLAVE_CFG]
    carga = await asyncio.to_thread(_carga_proyectos, cfg.directorio_datos)
    return web.json_response(
        {
            "proyectos": carga,
            # Sin fichero no hay proyectos, y no es un fallo: la pantalla enseña
            # dónde se crea en vez de un hueco sin explicación.
            "fichero": str(cfg.directorio_datos / proyectos.NOMBRE_FICHERO),
        }
    )


async def _abrir_proyecto(peticion: web.Request) -> web.Response:
    """Abre uno. Por aquí llega **cuál**, nunca qué ejecutar: ver `proyectos`."""
    cfg = peticion.app[CLAVE_CFG]
    resultado = await asyncio.to_thread(
        proyectos.abrir, cfg.directorio_datos, peticion.match_info["id"]
    )
    if resultado.startswith("Error:"):
        raise web.HTTPBadRequest(
            text=json.dumps({"error": resultado}), content_type="application/json"
        )
    return web.json_response({"resultado": resultado})


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
        # Cancelar y **esperar** la tarea: sin el `await`, un latido que se
        # quedara a medias de escribir soltaba una excepción sin nadie que la
        # recogiera cuando el transporte ya estaba cerrado.
        tarea_latido.cancel()
        with contextlib.suppress(
            asyncio.CancelledError, ConnectionResetError, RuntimeError
        ):
            await tarea_latido

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
            web.get("/hokusai-bg.png", _arte),
            web.get("/perseo-avatar.jpg", _arte),
            web.get("/icono-180.png", _icono),
            web.get("/icono-512.png", _icono),
            web.get("/apple-touch-icon.png", _icono),
            web.get("/apple-touch-icon-precomposed.png", _icono),
            web.get("/salud", _salud),
            web.get("/estado", _estado),
            web.post("/sesion", _abrir_sesion),
            web.post("/mensaje", _mensaje),
            web.post("/trabajos", _crear_trabajo),
            web.get("/trabajos", _listar_trabajos),
            web.get("/trabajos/{id}", _ver_trabajo),
            web.get("/trabajos/{id}/actividad", _ver_actividad),
            web.post("/trabajos/{id}/cancelar", _cancelar_trabajo),
            web.post("/trabajos/{id}/{decision:aprobar|rechazar}", _responder_confirmacion),
            web.get("/correos", _listar_correos),
            web.post("/correos/{id}/estado", _marcar_correo),
            web.get("/chat", _chat_sesiones),
            web.post("/chat", _crear_sesion_chat),
            web.get("/chat/{id}", _ver_sesion_chat),
            web.delete("/chat/{id}", _borrar_sesion_chat),
            web.post("/chat/{id}/hablar", _hablar_chat),
            web.get("/proyectos", _listar_proyectos),
            web.post("/proyectos/{id}/abrir", _abrir_proyecto),
            web.get("/grafo", _pagina_grafo),
            web.get("/grafo/datos", _datos_grafo),
            web.post("/grafo/abrir", _abrir_nota_grafo),
            web.get("/confianza", _ver_confianza),
            web.post("/confianza", _cambiar_confianza),
            web.post("/habitos", _habitos_espejo),
            web.get("/habitos", _habitos_ver),
            web.post("/tareas", _tareas_espejo),
            web.get("/tareas", _tareas_ver),
            web.post("/tareas/recoger", _tareas_recoger),
            web.get("/biometria", _biometria_estado),
            web.post("/biometria/voz", _biometria_voz),
            web.post("/biometria/cara", _biometria_cara),
            web.post("/biometria/perfiles", _biometria_enrolar),
            web.post("/biometria/perfiles/{nombre}", _biometria_renombrar),
            web.delete("/biometria/perfiles/{nombre}", _biometria_borrar),
            web.get("/eventos", _eventos),
        ]
    )
    return app
