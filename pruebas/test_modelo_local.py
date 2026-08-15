"""La llamada al modelo local: nunca lanza, y no quita el `think: false`."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import aiohttp

from perseo_core import modelo_local


class RespuestaFalsa:
    def __init__(self, status: int, cuerpo: Any) -> None:
        self.status = status
        self._cuerpo = cuerpo

    async def __aenter__(self) -> "RespuestaFalsa":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def json(self) -> Any:
        return self._cuerpo

    async def text(self) -> str:
        return json.dumps(self._cuerpo)


class SesionFalsa:
    """Recuerda el cuerpo que se manda, que es lo que se quiere mirar."""

    def __init__(self, respuesta: Any = None, excepcion: Exception | None = None) -> None:
        self.respuesta = respuesta
        self.excepcion = excepcion
        self.enviado: dict[str, Any] = {}

    def post(self, url: str, json: dict[str, Any]) -> Any:  # noqa: A002
        self.enviado = json
        if self.excepcion is not None:
            raise self.excepcion
        return self.respuesta


def preguntar(sesion: Any) -> Any:
    return asyncio.run(
        modelo_local.preguntar(sesion, "http://ollama", "qwen3:4b", {"type": "object"}, "sistema", "hola")
    )


def test_el_cuerpo_lleva_think_false() -> None:
    """Sin esto, Qwen3 con gramática se queda colgado y `content` llega vacío."""
    sesion = SesionFalsa(RespuestaFalsa(200, {"message": {"content": "{}"}}))
    preguntar(sesion)
    assert sesion.enviado["think"] is False


def test_el_cuerpo_lleva_la_gramatica_y_temperatura_cero() -> None:
    sesion = SesionFalsa(RespuestaFalsa(200, {"message": {"content": "{}"}}))
    preguntar(sesion)
    assert sesion.enviado["format"] == {"type": "object"}
    assert sesion.enviado["options"]["temperature"] == 0
    assert sesion.enviado["stream"] is False


def test_una_respuesta_buena_se_devuelve_ya_leida() -> None:
    sesion = SesionFalsa(RespuestaFalsa(200, {"message": {"content": '{"clase": "ignorar"}'}}))
    assert preguntar(sesion) == {"clase": "ignorar"}


def test_ollama_caido_devuelve_nada_en_vez_de_lanzar() -> None:
    """Es el caso normal en un portátil, no un error del sistema."""
    sesion = SesionFalsa(excepcion=aiohttp.ClientError("no hay nadie"))
    assert preguntar(sesion) is None


def test_un_timeout_devuelve_nada() -> None:
    sesion = SesionFalsa(excepcion=asyncio.TimeoutError())
    assert preguntar(sesion) is None


def test_un_codigo_de_error_devuelve_nada() -> None:
    assert preguntar(SesionFalsa(RespuestaFalsa(500, {"error": "roto"}))) is None


def test_una_respuesta_que_no_es_json_devuelve_nada() -> None:
    """Si pasa, la versión de Ollama no está aplicando la gramática."""
    sesion = SesionFalsa(RespuestaFalsa(200, {"message": {"content": "pues verás, creo que"}}))
    assert preguntar(sesion) is None


def test_un_json_que_no_es_un_objeto_devuelve_nada() -> None:
    sesion = SesionFalsa(RespuestaFalsa(200, {"message": {"content": "[1, 2, 3]"}}))
    assert preguntar(sesion) is None


def test_una_respuesta_sin_mensaje_devuelve_nada() -> None:
    assert preguntar(SesionFalsa(RespuestaFalsa(200, {}))) is None
