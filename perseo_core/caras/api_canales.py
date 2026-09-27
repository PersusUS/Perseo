"""Las rutas de lo que une a las caras: herramientas, hilo principal y ubicación.

Aparte de `api.py` por su techo de líneas, y juntas porque las tres existen para
lo mismo: que dé igual por dónde le hables a Perseo.

- `POST /herramientas/{nombre}` es por donde la voz ejecuta las herramientas
  que Rust no conoce (ADR 0008). Corre el mismo despacho que el chat escrito,
  con la puerta y la voz de quien habla puestas, así que la política ve lo
  mismo que si el trabajo lo hubiera encolado Rust.
- `GET /hilo` dice cuál es la conversación principal y lo último que hay en ella.
- `POST /ubicacion` guarda dónde está él, cuando él lo manda —un atajo del
  iPhone, por ejemplo—; `DELETE` la olvida.

Todas llevan token, como el resto: ninguna entra en `RUTAS_PUBLICAS`.
"""

from __future__ import annotations

import asyncio
import logging

from aiohttp import web

from ..agentes import chat_herramientas
from ..servicios import catalogo, hilo, recordatorios, ubicacion
from .api_comun import CLAVE_CFG, cuerpo_json, fallo

logger = logging.getLogger(__name__)


async def ejecutar_herramienta(peticion: web.Request) -> web.Response:
    nombre = peticion.match_info["nombre"]
    if catalogo.por_nombre(nombre) is None:
        raise fallo(web.HTTPNotFound, f"No hay ninguna herramienta «{nombre}» en el catálogo.")
    datos = await cuerpo_json(peticion)
    argumentos = datos.get("argumentos") or {}
    if not isinstance(argumentos, dict):
        raise fallo(web.HTTPBadRequest, "'argumentos' tiene que ser un objeto.")
    quien = datos.get("quien")
    try:
        texto = await chat_herramientas.ejecutar_desde(
            "voz", str(quien) if quien else None, nombre, argumentos
        )
    except chat_herramientas.ErrorHerramienta as e:
        raise fallo(web.HTTPUnprocessableEntity, str(e))
    return web.json_response({"texto": texto})


async def ver_hilo(peticion: web.Request) -> web.Response:
    cfg = peticion.app[CLAVE_CFG]
    sesion = await asyncio.to_thread(hilo.principal, cfg.directorio_datos)
    texto = await asyncio.to_thread(hilo.reciente, cfg.directorio_datos)
    return web.json_response({"sesion": sesion, "texto": texto})


async def guardar_ubicacion(peticion: web.Request) -> web.Response:
    cfg = peticion.app[CLAVE_CFG]
    datos = await cuerpo_json(peticion)
    try:
        dato = await asyncio.to_thread(
            ubicacion.guardar, cfg.directorio_datos, datos.get("latitud"), datos.get("longitud"),
            recordatorios.ahora_local(), datos.get("precision"), str(datos.get("fuente") or "atajo"),
        )
    except ValueError as e:
        raise fallo(web.HTTPBadRequest, str(e))
    return web.json_response(dato)


async def ver_ubicacion(peticion: web.Request) -> web.Response:
    cfg = peticion.app[CLAVE_CFG]
    return web.json_response({"ubicacion": await asyncio.to_thread(ubicacion.ultima, cfg.directorio_datos)})


async def olvidar_ubicacion(peticion: web.Request) -> web.Response:
    cfg = peticion.app[CLAVE_CFG]
    return web.json_response({"borrada": await asyncio.to_thread(ubicacion.borrar, cfg.directorio_datos)})
