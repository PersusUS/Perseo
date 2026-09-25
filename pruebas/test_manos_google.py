"""Correo y agenda con manos: enviar un borrador y apuntar citas, con lo que sale de casa parado.

Lo que se prueba es lo que protege: que enviar se pare aunque las
confirmaciones estén apagadas, que lo que diga la tarjeta sea lo que se envía
—o no se envíe nada—, y que invitar a alguien espere su sí.
"""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from perseo_core.agentes import agenda, chat_herramientas, correo
from perseo_core.infra import almacen, politica, router
from perseo_core.infra.bus import Bus
from perseo_core.infra.router import REGISTRO, NecesitaConfirmacion
from perseo_core.servicios import google_api, recordatorios


class BuzonDeBorradores:
    def __init__(self, para: str = "Ana <ana@correo.es>", asunto: str = "Re: La beca") -> None:
        self.real = {"para": para, "asunto": asunto}
        self.enviados: list[str] = []

    async def crear_borrador(self, para, asunto, cuerpo, hilo=""):
        return {"id": "r-7", "mensaje": "m-7"}

    async def leer_borrador(self, borrador):
        return dict(self.real)

    async def enviar_borrador(self, borrador):
        self.enviados.append(borrador)
        return {"id": "m-8", "hilo": "h-1"}


def _ejecutar_uno(trabajo: dict) -> None:
    asyncio.run(router.Trabajador(Bus())._ejecutar_uno(trabajo))


# --------------------------------------------------------------------------- #
# Enviar
# --------------------------------------------------------------------------- #


def test_el_borrador_le_dice_al_modelo_como_enviarlo(monkeypatch) -> None:
    buzon = BuzonDeBorradores()
    monkeypatch.setattr(correo, "_buzon", buzon)
    resultado = asyncio.run(
        REGISTRO["correo"]({"peticion": {"accion": "redactar", "para": "ana@correo.es", "asunto": "Re: La beca", "texto": "Hola"}})
    )
    assert "r-7" in resultado["texto"] and "enviar_borrador" in resultado["texto"]


def test_enviar_se_para_aunque_las_confirmaciones_esten_apagadas(db, monkeypatch) -> None:
    buzon = BuzonDeBorradores()
    monkeypatch.setattr(correo, "buzon", lambda: buzon)
    trabajo = almacen.encolar(
        "correo", {"accion": "enviar", "borrador": "r-7", "para": "ana@correo.es", "asunto": "Re: La beca"}
    )
    _ejecutar_uno(almacen.reclamar())

    parado = almacen.obtener(trabajo["id"])
    assert parado["estado"] == almacen.ESPERANDO
    assert parado["confirmacion"]["nivel"] == politica.EXTERIOR
    assert "ana@correo.es" in parado["confirmacion"]["resumen"] and "Re: La beca" in parado["confirmacion"]["resumen"]
    assert buzon.enviados == []


def test_el_modelo_no_aprueba_un_envio(db, monkeypatch) -> None:
    monkeypatch.setattr(correo, "buzon", lambda: BuzonDeBorradores())
    trabajo = almacen.encolar("correo", {"accion": "enviar", "borrador": "r-7", "para": "ana@correo.es", "asunto": "x"})
    _ejecutar_uno(almacen.reclamar())
    respuesta = asyncio.run(chat_herramientas._resolver_confirmacion({"id": trabajo["id"], "decision": "aprobar"}))
    assert "no se puede aprobar hablando" in respuesta


def test_tras_el_si_se_envia_si_es_el_mismo_borrador(db, monkeypatch) -> None:
    buzon = BuzonDeBorradores()
    monkeypatch.setattr(correo, "buzon", lambda: buzon)
    trabajo = almacen.encolar(
        "correo", {"accion": "enviar", "borrador": "r-7", "para": "ANA@correo.es", "asunto": "re:  la beca"}
    )
    _ejecutar_uno(almacen.reclamar())
    almacen.resolver_confirmacion(trabajo["id"], True)
    _ejecutar_uno(almacen.reclamar())

    assert buzon.enviados == ["r-7"]
    assert almacen.obtener(trabajo["id"])["estado"] == almacen.HECHO


def test_si_la_tarjeta_no_dice_lo_que_hay_en_gmail_no_se_envia_nada(db, monkeypatch) -> None:
    """El modelo escribió «para Ana» y el borrador va a otro: la tarjeta mentiría."""
    buzon = BuzonDeBorradores(para="otro@malo.es")
    monkeypatch.setattr(correo, "buzon", lambda: buzon)
    trabajo = almacen.encolar(
        "correo", {"accion": "enviar", "borrador": "r-7", "para": "ana@correo.es", "asunto": "Re: La beca"}
    )
    _ejecutar_uno(almacen.reclamar())
    almacen.resolver_confirmacion(trabajo["id"], True)
    _ejecutar_uno(almacen.reclamar())

    fallido = almacen.obtener(trabajo["id"])
    assert buzon.enviados == []
    assert fallido["estado"] == almacen.FALLIDO and "otro@malo.es" in fallido["error"]


# --------------------------------------------------------------------------- #
# Citas
# --------------------------------------------------------------------------- #


@pytest.fixture
def calendario(tmp_path: Path, monkeypatch) -> agenda.CalendarioFalso:
    falso = agenda.CalendarioFalso(tmp_path / "agenda.json")
    monkeypatch.setattr(agenda, "calendario", lambda: falso)
    return falso


def _manana() -> str:
    return (recordatorios.ahora_local() + timedelta(days=1)).replace(hour=21, minute=0).strftime("%Y-%m-%dT%H:%M")


def _apuntados(ruta: Path) -> list[dict[str, Any]]:
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else []


def test_una_cita_propia_se_apunta_sin_preguntar(calendario, tmp_path: Path) -> None:
    resultado = asyncio.run(REGISTRO["agenda"]({"peticion": {"accion": "crear", "titulo": "Cena", "inicio": _manana()}}))
    assert "Apuntado" in resultado["texto"] and "Cena" not in resultado["titular"]
    [cita] = _apuntados(tmp_path / "agenda.json")
    assert cita["titulo"] == "Cena" and cita["invitados"] == []


def test_invitar_a_alguien_espera_su_si(calendario, tmp_path: Path) -> None:
    peticion = {"accion": "crear", "titulo": "Cena", "inicio": _manana(), "invitados": "ana@correo.es, x"}
    with pytest.raises(NecesitaConfirmacion) as parada:
        asyncio.run(REGISTRO["agenda"]({"peticion": peticion}))
    assert parada.value.nivel == politica.EXTERIOR and "ana@correo.es" in parada.value.resumen
    assert _apuntados(tmp_path / "agenda.json") == []

    asyncio.run(REGISTRO["agenda"]({"peticion": peticion, "confirmacion": {"decision": "aprobado"}}))
    assert _apuntados(tmp_path / "agenda.json")[0]["invitados"] == ["ana@correo.es"]


def test_una_hora_pasada_no_se_apunta(calendario) -> None:
    resultado = asyncio.run(REGISTRO["agenda"]({"peticion": {"accion": "crear", "titulo": "Cena", "inicio": "2020-01-01T21:00"}}))
    assert "ya ha pasado" in resultado["texto"]


def test_crear_en_la_tabla_no_hereda_el_libre_de_leer() -> None:
    """`agenda` a secas es libre; crear escribe en su calendario y no puede heredarlo."""
    assert politica.nivel("agenda", {"accion": "crear"}) == politica.REVERSIBLE
    assert politica.nivel("agenda", {"accion": "invitar"}) == politica.EXTERIOR
    assert politica.nivel("correo", {"accion": "enviar"}) == politica.EXTERIOR


# --------------------------------------------------------------------------- #
# Lo que se le pide a Google
# --------------------------------------------------------------------------- #


class SesionQueApunta:
    def __init__(self, respuesta: dict[str, Any]) -> None:
        self.respuesta = respuesta
        self.mandados: list[tuple[str, dict[str, Any]]] = []
        self.pedidos: list[tuple[str, dict[str, Any] | None]] = []

    async def mandar(self, url, cuerpo):
        self.mandados.append((url, cuerpo))
        return self.respuesta

    async def pedir(self, url, parametros=None):
        self.pedidos.append((url, parametros))
        return self.respuesta


def _con_sesion(cliente, sesion) -> None:
    async def abrir():
        return sesion

    cliente._abrir = abrir


def test_enviar_borrador_es_drafts_send_con_su_id() -> None:
    buzon = google_api.BuzonGmail.__new__(google_api.BuzonGmail)
    sesion = SesionQueApunta({"id": "m-1", "threadId": "h-1"})
    _con_sesion(buzon, sesion)
    assert asyncio.run(buzon.enviar_borrador("r-7")) == {"id": "m-1", "hilo": "h-1"}
    [(url, cuerpo)] = sesion.mandados
    assert url.endswith("/gmail/v1/users/me/drafts/send") and cuerpo == {"id": "r-7"}


def test_leer_borrador_saca_para_y_asunto_de_las_cabeceras() -> None:
    buzon = google_api.BuzonGmail.__new__(google_api.BuzonGmail)
    cabeceras = [{"name": "To", "value": "ana@correo.es"}, {"name": "Subject", "value": "Re: La beca"}]
    _con_sesion(buzon, SesionQueApunta({"message": {"payload": {"headers": cabeceras}}}))
    assert asyncio.run(buzon.leer_borrador("r-7")) == {"para": "ana@correo.es", "asunto": "Re: La beca"}


def test_con_invitados_google_les_avisa_y_sin_ellos_no() -> None:
    calendario_ = google_api.CalendarioGoogle.__new__(google_api.CalendarioGoogle)
    calendario_._calendario = "primary"
    sesion = SesionQueApunta({"id": "e-1", "htmlLink": "https://calendar.google.com/x"})
    _con_sesion(calendario_, sesion)
    inicio = recordatorios.ahora_local()
    asyncio.run(calendario_.crear("Cena", inicio, inicio + timedelta(hours=1)))
    asyncio.run(calendario_.crear("Cena", inicio, inicio + timedelta(hours=1), invitados=("ana@correo.es",)))
    (sin, cuerpo_sin), (con, cuerpo_con) = sesion.mandados
    assert "sendUpdates" not in sin and "attendees" not in cuerpo_sin
    assert con.endswith("?sendUpdates=all") and cuerpo_con["attendees"] == [{"email": "ana@correo.es"}]
