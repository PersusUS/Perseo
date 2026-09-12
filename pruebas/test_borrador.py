"""Borradores de correo: lo que se escribe, y lo que no se puede enviar.

La garantía de este camino no está en el código, está en el permiso: el testigo
pide `gmail.compose`, que escribe borradores y **no** incluye `send`. Aunque el
modelo se empeñara, Google contestaría 403. Lo que se comprueba aquí es que el
código no se salga de eso y que el mensaje llegue bien formado — un borrador con
las tildes rotas es tan inútil como no tenerlo.
"""

from __future__ import annotations

import asyncio
import base64
from email import message_from_bytes
from typing import Any

import pytest

from perseo_core.agentes import correo
from perseo_core.infra import politica
from perseo_core.servicios import autorizar_google, google_api


# --------------------------------------------------------------------------- #
# El permiso
# --------------------------------------------------------------------------- #


def test_se_pide_compose_y_no_send() -> None:
    """`compose` escribe borradores; `send` manda correo y `modify` borra.

    Si alguien amplía esto, que sea a sabiendas: es la única cosa que separa
    "Perseo redacta" de "Perseo escribe en tu nombre a quien sea".
    """
    ambitos = " ".join(autorizar_google.AMBITOS)
    assert "gmail.compose" in ambitos
    assert "gmail.send" not in ambitos
    assert "gmail.modify" not in ambitos


def test_solo_lectura_para_lo_demas() -> None:
    ambitos = " ".join(autorizar_google.AMBITOS)
    assert "gmail.readonly" in ambitos
    assert "calendar.readonly" in ambitos
    # Del calendario no se pide nada que escriba: mover un evento sigue siendo
    # cosa suya.
    assert not any("calendar" in a and "readonly" not in a for a in autorizar_google.AMBITOS)


def test_redactar_no_hereda_el_libre_del_triaje() -> None:
    """`correo` entero es LIBRE por el triaje. Sin entrada propia, redactar
    escribiría en la cuenta sin quedar registrado en ninguna parte."""
    assert politica.nivel("correo", {"accion": "redactar"}) == politica.REVERSIBLE
    assert politica.nivel("correo", {"accion": "triar"}) == politica.LIBRE


# --------------------------------------------------------------------------- #
# El mensaje que se manda
# --------------------------------------------------------------------------- #


class _SesionFalsa:
    """Se queda con lo que se le manda, en vez de hablar con Google."""

    def __init__(self) -> None:
        self.url = ""
        self.cuerpo: dict[str, Any] = {}

    async def mandar(self, url: str, cuerpo: dict[str, Any]) -> dict[str, Any]:
        self.url = url
        self.cuerpo = cuerpo
        return {"id": "borrador-1", "message": {"id": "mensaje-1"}}


def _buzon_falso() -> tuple[google_api.BuzonGmail, _SesionFalsa]:
    buzon = google_api.BuzonGmail(
        google_api.Credenciales(client_id="a", client_secret="b", refresh_token="c")
    )
    sesion = _SesionFalsa()

    async def abrir() -> _SesionFalsa:
        return sesion

    buzon._abrir = abrir  # type: ignore[assignment]
    return buzon, sesion


def _leer_crudo(cuerpo: dict[str, Any]):
    return message_from_bytes(base64.urlsafe_b64decode(cuerpo["message"]["raw"]))


def test_el_borrador_va_a_la_ruta_de_borradores() -> None:
    buzon, sesion = _buzon_falso()
    asyncio.run(buzon.crear_borrador("ana@example.com", "Hola", "Qué tal."))
    assert sesion.url.endswith("/gmail/v1/users/me/drafts")


def test_el_mensaje_lleva_destinatario_asunto_y_cuerpo() -> None:
    buzon, sesion = _buzon_falso()
    asyncio.run(buzon.crear_borrador("ana@example.com", "Presupuesto", "Adjunto lo hablado."))
    mensaje = _leer_crudo(sesion.cuerpo)
    assert mensaje["To"] == "ana@example.com"
    assert mensaje["Subject"] == "Presupuesto"
    assert "Adjunto lo hablado." in mensaje.get_payload(decode=True).decode("utf-8")


def test_las_tildes_sobreviven() -> None:
    """Un borrador con la eñe rota es tan inútil como no tenerlo."""
    buzon, sesion = _buzon_falso()
    asyncio.run(
        buzon.crear_borrador("ana@example.com", "Reunión del miércoles", "Añado la señal.")
    )
    mensaje = _leer_crudo(sesion.cuerpo)
    assert "Añado la señal." in mensaje.get_payload(decode=True).decode("utf-8")


def test_no_se_escribe_el_remitente() -> None:
    """Lo pone Gmail con la cuenta del testigo; escribirlo solo sirve para
    equivocarse de dirección."""
    buzon, sesion = _buzon_falso()
    asyncio.run(buzon.crear_borrador("ana@example.com", "Hola", "Qué tal."))
    assert _leer_crudo(sesion.cuerpo)["From"] is None


def test_con_hilo_cuelga_de_la_conversacion() -> None:
    buzon, sesion = _buzon_falso()
    asyncio.run(buzon.crear_borrador("ana@example.com", "Re: cita", "Confirmado.", hilo="hilo-9"))
    assert sesion.cuerpo["message"]["threadId"] == "hilo-9"


def test_sin_hilo_no_se_manda_el_campo() -> None:
    """Un `threadId` vacío no es "sin hilo": Gmail lo rechaza."""
    buzon, sesion = _buzon_falso()
    asyncio.run(buzon.crear_borrador("ana@example.com", "Hola", "Qué tal."))
    assert "threadId" not in sesion.cuerpo["message"]


# --------------------------------------------------------------------------- #
# El agente
# --------------------------------------------------------------------------- #


def test_el_agente_exige_destinatario_y_texto() -> None:
    with pytest.raises(ValueError):
        asyncio.run(correo._redactar({"asunto": "Solo el asunto"}))


def test_un_buzon_que_no_sabe_redactar_lo_dice(monkeypatch: pytest.MonkeyPatch) -> None:
    """Con el buzón de mentira no hay borrador, y hay que decirlo: fingir que se
    escribió disimularía que el permiso no está."""

    class SinRedactar:
        pass

    monkeypatch.setattr(correo, "_buzon", SinRedactar())
    with pytest.raises(RuntimeError, match="gmail.compose"):
        asyncio.run(correo._redactar({"para": "ana@example.com", "texto": "Hola"}))


def test_el_titular_no_lleva_el_cuerpo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sale por Telegram: dice a quién, nunca lo que pone dentro."""

    class ConRedactar:
        async def crear_borrador(self, para, asunto, cuerpo, hilo=""):
            return {"id": "b-1", "mensaje": "m-1"}

    monkeypatch.setattr(correo, "_buzon", ConRedactar())
    resultado = asyncio.run(
        correo._redactar(
            {"para": "ana@example.com", "asunto": "Cita", "texto": "SECRETO DEL CUERPO"}
        )
    )
    assert "SECRETO DEL CUERPO" not in resultado["titular"]
    assert "ana@example.com" in resultado["titular"]
