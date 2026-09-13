"""Las rutas de biometría: quién habla y quién sale por la cámara.

Salieron de `api.py` el 2026-09-12, y no por gusto: la regla de tamaño se puso
roja al añadir `GET /herramientas` y el fichero llevaba tiempo pidiéndolo. Van
juntas porque comparten lo que las hace distintas del resto de la API:

  · **Solo responden con el reconocimiento encendido.** Sin él, 503 y a otra
    cosa; no es un error, es que no está puesto.
  · **Los vectores no salen nunca.** Por aquí entra audio o imagen y sale un
    nombre. Lo que se guarda en `perfiles.json` son datos biométricos de gente
    real y se queda en el disco. Ver `docs/PRIVACIDAD.md`.
  · **Van autenticadas como todo**, incluida la de solo leer: la lista de a
    quién conoces es gente con nombre y apellidos.
"""

from __future__ import annotations

import asyncio
import logging

from aiohttp import web

from ..servicios import biometria
from .api_comun import CLAVE_BUS, CLAVE_CFG, cuerpo_json, fallo

logger = logging.getLogger(__name__)


async def biometria_estado(peticion: web.Request) -> web.Response:
    """Perfiles, progreso de aprendizaje y qué motores hay hoy.

    Va autenticado como todo: los nombres de los perfiles son gente real, y la
    lista de quién conoces no se le enseña a nadie sin token.
    """
    cfg = peticion.app[CLAVE_CFG]
    return web.json_response(
        await asyncio.to_thread(biometria.estado_completo, cfg.directorio_datos)
    )


async def biometria_voz(peticion: web.Request) -> web.Response:
    """Un trozo de PCM 16k mono (base64) entra, un nombre o un progreso sale.

    Es la ruta que llama la app de voz con el mismo micrófono que ya alimenta
    a Gemini. Cuando aquí nace un perfil nuevo —un desconocido que por fin
    acumuló voz suficiente— se publica al bus, para que quien escuche sepa que
    hay alguien nuevo en la casa.
    """
    cuerpo = await cuerpo_json(peticion)
    audio = str(cuerpo.get("audio", ""))
    if not audio:
        raise fallo(web.HTTPBadRequest, "Falta 'audio'")

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


async def biometria_cara(peticion: web.Request) -> web.Response:
    """Un JPEG (base64) entra; caras con nombre y caja salen.

    Igual que la voz: cuando una cara desconocida se fija como perfil, evento.
    """
    cuerpo = await cuerpo_json(peticion)
    imagen = str(cuerpo.get("imagen", ""))
    if not imagen:
        raise fallo(web.HTTPBadRequest, "Falta 'imagen'")

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


async def biometria_enrolar(peticion: web.Request) -> web.Response:
    """Crea o refuerza un perfil con una muestra traída a propósito."""
    cuerpo = await cuerpo_json(peticion)
    nombre = str(cuerpo.get("nombre", ""))
    audio = cuerpo.get("audio")
    imagen = cuerpo.get("imagen")
    if not nombre:
        raise fallo(web.HTTPBadRequest, "Falta 'nombre'")
    if not audio and not imagen:
        raise fallo(web.HTTPBadRequest, "Hace falta 'audio' o 'imagen'")

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


async def biometria_renombrar(peticion: web.Request) -> web.Response:
    """Le pone nombre real a un «Desconocido N»."""
    cuerpo = await cuerpo_json(peticion)
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
    from ..agentes import memoria

    try:
        await memoria.anotar_persona(antes, ahora)
    except Exception as e:  # noqa: BLE001 - un apunte que falla no rompe nada
        logger.warning("No se pudo anotar a %s en el vault: %s", ahora, e)


async def biometria_borrar(peticion: web.Request) -> web.Response:
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
