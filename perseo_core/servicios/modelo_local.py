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

"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import aiohttp

from ..infra import almacen

logger = logging.getLogger(__name__)

#: Espera máxima de una decisión local. Un 4B en la GPU tarda entre 1,5 y 3,6
#: segundos; 30 da margen para un arranque en frío del modelo.
ESPERA = 30

#: Dónde vive el suplente. Se puede apuntar a otro sitio para verificarlo sin
#: cuota, igual que con Telegram o con Google.
def _url_gemini() -> str:
    import os

    return os.environ.get(
        "PERSEO_GEMINI_API", "https://generativelanguage.googleapis.com"
    ).rstrip("/")


@dataclass(frozen=True)
class Suplente:
    """El modelo de fuera que responde cuando el de casa no está.

    **Apagado salvo que se pida.** Es la única pieza del sistema que manda a un
    tercero el texto que se está clasificando: para el correo da igual —ese
    correo ya vive en Gmail— pero por el router pasa lo que le escribes a Perseo.
    Encenderlo es una decisión, no un valor por defecto sensato.

    Gemma sirve aquí y no sirve para la voz: sus modelos exponen
    `generateContent` y **no** `bidiGenerateContent`. No sustituye al modelo de
    voz, sustituye al de casa cuando Ollama está apagado.
    """

    clave: str
    modelo: str

    @property
    def utilizable(self) -> bool:
        return bool(self.clave and self.modelo)


async def preguntar(
    sesion: aiohttp.ClientSession,
    url_ollama: str,
    modelo: str,
    esquema: dict[str, Any],
    sistema: str,
    usuario: str,
    suplente: Suplente | None = None,
) -> dict[str, Any] | None:
    """Pide una respuesta con forma al modelo local. Devuelve `None` si no se pudo.

    `None` cubre los tres fallos posibles —Ollama caído, respuesta que no es 200,
    contenido que no es JSON— porque para quien llama son el mismo caso: hoy no
    hay decisión local y hay que tirar por el camino seguro.

    Si hay `suplente` configurado y el de casa no contesta, se pregunta fuera
    antes de rendirse. El orden no se invierte nunca: el local es gratis y no
    manda nada a ningún sitio.
    """
    decision = await _preguntar_ollama(sesion, url_ollama, modelo, esquema, sistema, usuario)
    if decision is not None:
        return decision
    if suplente is not None and suplente.utilizable:
        logger.info("El modelo local no contestó; se pregunta al suplente %s.", suplente.modelo)
        return await preguntar_suplente(sesion, suplente, esquema, sistema, usuario)
    return None


async def _preguntar_ollama(
    sesion: aiohttp.ClientSession,
    url_ollama: str,
    modelo: str,
    esquema: dict[str, Any],
    sistema: str,
    usuario: str,
) -> dict[str, Any] | None:
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


def acepta_esquema(modelo: str) -> bool:
    """¿Se le puede exigir la forma de la respuesta, o hay que pedirla por favor?

    Los `gemini-*` aceptan `responseSchema`; Gemma no. Se mira por el nombre
    porque es lo único que se sabe del modelo antes de llamarlo, y el coste de
    equivocarse es una petición perdida, no un fallo silencioso.
    """
    return "gemma" not in modelo.lower()


async def preguntar_suplente(
    sesion: aiohttp.ClientSession,
    suplente: Suplente,
    esquema: dict[str, Any],
    sistema: str,
    usuario: str,
) -> dict[str, Any] | None:
    """Lo mismo, pero contra la API de Gemini.

    Hay **dos formas de pedirlo** y se elige según el modelo, porque no todos
    aceptan lo mismo:

    1. **Con esquema, si el modelo lo acepta** (los `gemini-*`): `responseSchema`
       y `responseMimeType` obligan a que la respuesta SEA el objeto pedido, y
       las instrucciones van en `systemInstruction`, su sitio. Es lo más
       parecido a la gramática de Ollama que da esta API.
    2. **Con el esquema escrito dentro del texto** (Gemma, que no acepta ni una
       cosa ni la otra). La respuesta se lee entonces con tolerancia —un modelo
       grande la envuelve en ```json más veces de las que uno espera— y la forma
       se comprueba después.

    La diferencia no es cosmética: con Gemma, el 2026-08-24, el registro se
    llenó de «El suplente devolvió algo que no es JSON» seguido del razonamiento
    del modelo en voz alta. Un `flash-lite` con esquema no puede hacer eso.
    """
    url = f"{_url_gemini()}/v1beta/models/{suplente.modelo}:generateContent"
    if acepta_esquema(suplente.modelo):
        cuerpo = {
            "contents": [{"role": "user", "parts": [{"text": usuario}]}],
            "systemInstruction": {"parts": [{"text": sistema}]},
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseSchema": esquema,
            },
        }
    else:
        cuerpo = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                f"{sistema}\n\n"
                                "Responde SOLO con un objeto JSON que cumpla este esquema, "
                                "sin texto alrededor ni explicaciones:\n"
                                f"{json.dumps(esquema, ensure_ascii=False)}\n\n"
                                f"{usuario}"
                            )
                        }
                    ],
                }
            ],
            "generationConfig": {"temperature": 0},
        }

    try:
        async with sesion.post(
            url, params={"key": suplente.clave}, json=cuerpo
        ) as respuesta:
            # Se apunta en cuanto hay respuesta, sea cual sea: la petición ya ha
            # salido y Google ya la ha contado. Apuntar solo los 200 haría que el
            # panel dijera que queda cuota justo el día que se agota.
            await asyncio.to_thread(almacen.apuntar_uso, suplente.modelo)
            datos = await respuesta.json()
            if respuesta.status != 200:
                logger.warning(
                    "El suplente respondió %d: %.200s",
                    respuesta.status,
                    (datos.get("error") or {}).get("message", datos),
                )
                return None
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
        logger.warning("El suplente tampoco está disponible (%s).", e)
        return None

    candidatos = datos.get("candidates") or []
    partes = ((candidatos[0] if candidatos else {}).get("content") or {}).get("parts") or []
    crudo = "".join(str(p.get("text", "")) for p in partes)

    decision = primer_objeto(crudo)
    if decision is None:
        logger.error("El suplente devolvió algo que no es JSON: %.200s", crudo)
    return decision


#: Un objeto JSON dentro de un texto, con o sin vallas de ```json alrededor.
_VALLA = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def primer_objeto(texto: str) -> dict[str, Any] | None:
    """El primer objeto JSON que aparezca en un texto, o `None`.

    Sin gramática que lo garantice, un modelo grande contesta bien y **envuelto**:
    en vallas de código, con una frase delante, o las dos cosas. Exigir un JSON
    pelado desperdiciaría respuestas correctas.
    """
    if not texto:
        return None

    candidatos = [c.strip() for c in _VALLA.findall(texto)]
    candidatos.append(texto.strip())
    # Y como último recurso, desde la primera llave hasta la última.
    primera, ultima = texto.find("{"), texto.rfind("}")
    if 0 <= primera < ultima:
        candidatos.append(texto[primera : ultima + 1])

    for candidato in candidatos:
        try:
            decision = json.loads(candidato)
        except json.JSONDecodeError:
            continue
        if isinstance(decision, dict):
            return decision
    return None
