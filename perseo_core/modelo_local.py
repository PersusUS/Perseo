"""Llamadas al modelo local con gramática, en un solo sitio.

El router de la Fase A ya hablaba con Ollama, pero era el único que lo hacía y la
llamada vivía dentro de `agentes.py`. La Fase D trae un segundo consumidor —el
triaje de correo— y con dos ya conviene que la llamada esté escrita una vez: lo
que hay dentro no es obvio, y una copia divergente cuesta horas.

Lo que no es obvio, y por qué está aquí:

1. **`"think": false`.** La familia Qwen3 es de razonamiento híbrido, y el
   razonamiento choca con la decodificación restringida de `format`: la petición
   se queda colgada minutos y `content` llega vacío. Desde fuera parece que
   Ollama no está levantado. No romper esto es lo que hace utilizable a Qwen3 con
   gramática, y no perjudica a los modelos que no razonan.
2. **Nunca lanza.** Que el modelo local no esté disponible es el caso normal en
   un portátil, no un error del sistema. Se devuelve `None` y quien llama decide
   qué es lo seguro en su caso: el router encola, el triaje escala.
3. **La gramática garantiza la forma, no el contenido.** Un 4B devuelve siempre
   un JSON del esquema y a veces rellena mal un campo. Por eso todos los
   esquemas de este sistema tienen una salida de "no estoy seguro": el modelo
   tiene dónde escalar en vez de inventarse una respuesta.

Ver bitacora/05_PLAN_PERSEO_V2.md §3 y bitacora/06_HANDOFF.md §6, trampa 3.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

#: Espera máxima de una decisión local. Un 4B en la GPU tarda entre 1,5 y 3,6
#: segundos; 30 da margen para un arranque en frío del modelo.
ESPERA = 30


async def preguntar(
    sesion: aiohttp.ClientSession,
    url_ollama: str,
    modelo: str,
    esquema: dict[str, Any],
    sistema: str,
    usuario: str,
) -> dict[str, Any] | None:
    """Pide una respuesta con forma al modelo local. Devuelve `None` si no se pudo.

    `None` cubre los tres fallos posibles —Ollama caído, respuesta que no es 200,
    contenido que no es JSON— porque para quien llama son el mismo caso: hoy no
    hay decisión local y hay que tirar por el camino seguro.
    """
    cuerpo = {
        "model": modelo,
        "stream": False,
        "format": esquema,
        # No quitar. Ver la cabecera de este módulo.
        "think": False,
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": sistema},
            {"role": "user", "content": usuario},
        ],
    }

    try:
        async with sesion.post(f"{url_ollama}/api/chat", json=cuerpo) as respuesta:
            if respuesta.status != 200:
                logger.warning(
                    "El modelo local respondió %d: %.200s",
                    respuesta.status,
                    await respuesta.text(),
                )
                return None
            datos = await respuesta.json()
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.warning("Modelo local no disponible (%s).", e)
        return None

    crudo = (datos.get("message") or {}).get("content", "")
    try:
        decision = json.loads(crudo)
    except json.JSONDecodeError:
        # Con `format` puesto no debería ocurrir. Si ocurre, esta versión de
        # Ollama no está aplicando la gramática, y eso conviene verlo.
        logger.error("El modelo local devolvió algo que no es JSON: %.200s", crudo)
        return None

    if not isinstance(decision, dict):
        logger.error("El modelo local devolvió un %s en vez de un objeto.", type(decision).__name__)
        return None
    return decision
