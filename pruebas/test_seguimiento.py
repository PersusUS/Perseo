"""Seguimiento: avisar de lo que pedía algo y lleva días sin respuesta, y solo de eso."""

from __future__ import annotations

import asyncio
import email.utils
from datetime import datetime, timedelta, timezone

import pytest

from perseo_core.agentes import correo, seguimiento
from perseo_core.dominio.mensaje import Mensaje, es_automatico
from perseo_core.infra import almacen, disparadores
from perseo_core.infra.bus import Bus
from perseo_core.infra.router import REGISTRO
from perseo_core.servicios import recordatorios

from test_llamada_saliente import _marcador

AHORA = datetime(2026, 9, 24, 10, 0, tzinfo=timezone(timedelta(hours=2)))


def _fecha(dias: float) -> str:
    return email.utils.format_datetime(AHORA - timedelta(days=dias))


def _correo(id_: str, dias: float, clase: str = "requiere_accion", hecho: str = "") -> dict:
    return {"id": id_, "hilo": f"h{id_}", "remitente": "Ana", "asunto": f"asunto {id_}",
            "fecha": _fecha(dias), "clase": clase, "hecho": hecho}


def test_la_edad_sale_de_la_cabecera_date() -> None:
    assert seguimiento.edad(_fecha(3), AHORA).days == 3
    assert seguimiento.edad("2026-09-20T10:00:00", AHORA).days == 4
    assert seguimiento.edad("no es una fecha", AHORA) is None


def test_solo_lo_que_pide_algo_sin_marcar_y_entre_dos_y_catorce_dias() -> None:
    correos = [
        _correo("a", 3),
        _correo("b", 1),  # aún no: le queda tiempo
        _correo("c", 30),  # demasiado viejo: se decidió no seguirlo
        _correo("d", 3, clase="interesante"),
        _correo("e", 3, hecho="atendido"),
        _correo("f", 5),
    ]
    assert [c["id"] for c in seguimiento.candidatos(correos, AHORA, avisados={"f"})] == ["a"]


def test_lo_que_mando_una_maquina_no_llama() -> None:
    """La primera llamada de verdad nombró un evento de Luma, el CI y Stripe."""
    correos = [_correo("a", 3), {**_correo("b", 3), "automatico": True}]
    assert [c["id"] for c in seguimiento.candidatos(correos, AHORA, avisados=set())] == ["a"]


@pytest.mark.parametrize(
    "remitente",
    [
        "Jaime <usr-eF4akgGDvMGcFP7@user.luma-mail.com>",
        "Persus <notifications@github.com>",
        "Kickbacks <support@stripe.com>",
        "Banco <no-reply@banco.es>",
        "noreply+avisos@empresa.com",
    ],
)
def test_los_remitentes_de_maquina_se_reconocen_por_la_direccion(remitente: str) -> None:
    assert es_automatico(remitente)


@pytest.mark.parametrize(
    "remitente", ["Ana López <ana.lopez@gmail.com>", "profesor@us.es", "Info Pérez <jinfo@empresa.com>", "", "Ana"]
)
def test_una_persona_no_se_toma_por_maquina(remitente: str) -> None:
    assert not es_automatico(remitente)


def test_las_cabeceras_de_envio_delatan_un_boletin_aunque_la_direccion_parezca_de_persona() -> None:
    assert es_automatico("ana@tienda.com", {"List-Unsubscribe": "<mailto:baja@tienda.com>"})
    assert es_automatico("ana@tienda.com", {"Precedence": "bulk"})
    assert es_automatico("ana@tienda.com", {"Auto-Submitted": "auto-replied"})
    assert not es_automatico("ana@tienda.com", {"Auto-Submitted": "no", "Precedence": ""})


def test_un_mensaje_viejo_sin_la_marca_se_lee_como_de_persona() -> None:
    assert Mensaje.desde_dict({"id": "1", "remitente": "a", "asunto": "b"}).automatico is False
    assert Mensaje.desde_dict({"id": "1", "remitente": "a", "asunto": "b", "automatico": True}).automatico


class BuzonConHilos:
    def __init__(self, contestados: set[str]) -> None:
        self.contestados = contestados
        self.mirados: list[str] = []

    async def respondido(self, hilo: str) -> bool:
        self.mirados.append(hilo)
        return hilo in self.contestados


def _revisar(correos: list[dict], buzon, monkeypatch) -> dict:
    monkeypatch.setattr(correo, "buzon", lambda: buzon)
    return asyncio.run(REGISTRO["seguimiento"]({"id": 5, "peticion": {"correos": correos}}))


def test_lo_ya_contestado_en_gmail_se_marca_y_no_llama(db, monkeypatch) -> None:
    """Casi nadie marca en el panel: se contesta desde Gmail, y eso tiene que contar."""
    buzon = BuzonConHilos(contestados={"ha"})
    resultado = _revisar([{**_correo("a", 3), "dias": 3}], buzon, monkeypatch)
    assert resultado["callado"] and resultado["titular"] is None
    assert almacen.correos_marcados() == {"a": almacen.ATENDIDO}
    assert not _marcador().exists()


def test_lo_que_sigue_callado_suena_con_el_detalle_y_telegram_solo_cuenta(db, monkeypatch) -> None:
    buzon = BuzonConHilos(contestados={"hb"})
    resultado = _revisar(
        [{**_correo("a", 3), "dias": 3}, {**_correo("b", 4), "dias": 4}], buzon, monkeypatch
    )
    llamada = _marcador().read_text(encoding="utf-8")
    assert "asunto a" in llamada and "3 días" in llamada and "asunto b" not in llamada
    assert resultado["titular"] == "1 correo(s) que pedían algo siguen sin respuesta"
    assert "asunto" not in resultado["titular"]


def test_si_gmail_no_contesta_se_avisa_igual(db, monkeypatch) -> None:
    """Sin saber si contestó, se avisa: es el lado que no calla lo importante."""

    class BuzonRoto:
        async def respondido(self, hilo: str) -> bool:
            raise RuntimeError("Google respondió 500")

    resultado = _revisar([{**_correo("a", 3), "dias": 3}], BuzonRoto(), monkeypatch)
    assert resultado["titular"]


def _triado(id_: str, dias: float) -> None:
    """Un correo como lo deja el triaje en la cola: el lote y su clasificación."""
    trabajo = almacen.encolar("correo", {"accion": "triar", "mensajes": [
        {"id": id_, "remitente": "Ana", "asunto": "La beca", "fecha": _fecha(dias), "hilo": f"h{id_}"}
    ]}, "disparador")
    almacen.reclamar()
    almacen.completar(trabajo["id"], {"clasificados": [{"id": id_, "clase": "requiere_accion"}]})


@pytest.mark.parametrize("hora, encola", [(10, True), (3, False), (22, False)])
def test_el_disparador_encola_una_vez_y_a_horas_de_persona(cfg, db, monkeypatch, hora, encola) -> None:
    monkeypatch.setattr(seguimiento, "_avisados", None)
    monkeypatch.setattr(recordatorios, "ahora_local", lambda: AHORA.replace(hour=hora))
    _triado("m1", 3)
    ctx = disparadores.Contexto(cfg=cfg, bus=Bus())

    asyncio.run(seguimiento._vigilar_seguimiento(ctx))
    asyncio.run(seguimiento._vigilar_seguimiento(ctx))

    encolados = [t for t in almacen.listar() if t["agente"] == "seguimiento"]
    assert len(encolados) == (1 if encola else 0)
    if encola:
        assert encolados[0]["peticion"]["correos"][0]["hilo"] == "hm1"
