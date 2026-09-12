"""Lo que comparten las rutas de la API, vivan en el fichero que vivan.

Tres claves y dos ayudantes. Están aquí y no en `api.py` porque desde que las
rutas de biometría se fueron a su propio fichero hay dos módulos que los
necesitan, y hacer que uno importe del otro sería un ciclo entre hermanos.
"""

from __future__ import annotations

import json
from typing import Any

from aiohttp import web

from ..infra import almacen
from ..infra.bus import Bus
from ..infra.router import Router

#: Lo que la aplicación lleva colgado. `AppKey` y no una cadena: con cadenas,
#: una errata se descubre en producción con un `KeyError` sin contexto.
CLAVE_CFG: web.AppKey[almacen.Configuracion] = web.AppKey("cfg")
CLAVE_BUS: web.AppKey[Bus] = web.AppKey("bus")
CLAVE_ROUTER: web.AppKey[Router] = web.AppKey("router")


def fallo(clase: type[web.HTTPException], mensaje: str) -> web.HTTPException:
    """Un error de la API, en JSON y no en la página HTML de aiohttp.

    Esto estaba escrito treinta y cuatro veces —tres líneas cada una— y el
    tercio de las veces con el `content_type` en una línea distinta, así que
    ningún grep encontraba las mismas. Quien consume esta API es una PWA y un
    puente en Rust: los dos hacen `json()` con lo que reciben, y un `<html>` de
    aiohttp ahí es un error de parseo en vez de un mensaje.
    """
    return clase(text=json.dumps({"error": mensaje}), content_type="application/json")


async def cuerpo_json(peticion: web.Request) -> dict[str, Any]:
    try:
        datos = await peticion.json()
    except json.JSONDecodeError:
        raise fallo(web.HTTPBadRequest, "El cuerpo no es JSON válido")
    if not isinstance(datos, dict):
        raise fallo(web.HTTPBadRequest, "Se esperaba un objeto JSON")
    return datos
