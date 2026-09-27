"""Llamar a un negocio: se para antes de marcar, se presenta como IA, y cuenta cómo fue."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from perseo_core.agentes import telefono
from perseo_core.infra import almacen, politica, router
from perseo_core.infra.bus import Bus
from perseo_core.servicios import twilio

from test_canales import ClienteFalso, _cuenta


class CerebroDeGuion:
    """Contesta con la jugada que toque según cuántas frases lleva la llamada."""

    def __init__(self, jugadas: list[tuple[str, dict[str, Any]]]) -> None:
        self.jugadas = jugadas
        self.sistemas: list[str] = []

    async def pensar(self, sistema, contents, declaraciones):
        self.sistemas.append(sistema)
        n = sum(1 for c in contents if c["role"] == "model")
        nombre, args = self.jugadas[min(n, len(self.jugadas) - 1)]
        return {"role": "model", "parts": [{"functionCall": {"name": nombre, "args": args}}]}

    async def cerrar(self):
        return None


@pytest.fixture
def montado(cfg, db):
    def montar(jugadas):
        cerebro = CerebroDeGuion(jugadas)
        falso = ClienteFalso(_cuenta())
        telefono.iniciar(cfg, cerebro=cerebro, cliente=falso)
        return cerebro, falso

    return montar


def test_marcar_se_para_y_la_tarjeta_dice_a_quien_y_para_que(db) -> None:
    trabajo = almacen.encolar(
        "telefono", {"accion": "negocio", "numero": "+34954000000", "objetivo": "mesa para 2 el viernes", "negocio": "Casa Prueba"}
    )
    asyncio.run(router.Trabajador(Bus())._ejecutar_uno(almacen.reclamar()))
    parado = almacen.obtener(trabajo["id"])
    assert parado["estado"] == almacen.ESPERANDO and parado["confirmacion"]["nivel"] == politica.EXTERIOR
    resumen = parado["confirmacion"]["resumen"]
    assert "+34954000000" in resumen and "Casa Prueba" in resumen and "mesa para 2" in resumen


def test_la_primera_frase_dice_que_es_una_ia_aunque_el_modelo_no_lo_diga(montado) -> None:
    montado([("decir", {"texto": "¿Tienen mesa para dos el viernes a las nueve?"})])
    llamada = telefono.nueva("negocio", numero="+34954000000", objetivo="mesa para 2", negocio="Casa Prueba")
    xml = asyncio.run(telefono.turno_negocio(llamada["id"], None, False))
    assert "asistente de inteligencia artificial" in xml and "mesa para dos" in xml
    assert telefono.leer(llamada["id"])["transcripcion"][0]["texto"].startswith("Hola, buenas.")


def test_el_modelo_no_puede_dar_datos_de_pago_porque_no_los_tiene(montado) -> None:
    """Lo que ve el modelo al teléfono: el encargo y la regla, nunca la bóveda."""
    cerebro, _ = montado([("decir", {"texto": "Hola"})])
    llamada = telefono.nueva("negocio", numero="+34954000000", objetivo="mesa para 2")
    asyncio.run(telefono.turno_negocio(llamada["id"], None, False))
    assert "NUNCA des datos de pago" in cerebro.sistemas[0] and "boveda" not in cerebro.sistemas[0]


def test_colgar_deja_el_resumen_y_se_despide(montado) -> None:
    montado([
        ("decir", {"texto": "¿Tienen mesa para dos el viernes?"}),
        ("colgar", {"despedida": "Perfecto, muchas gracias.", "resultado": "logrado", "resumen": "Mesa para 2, viernes 21:00, a nombre de Jesús."}),
    ])
    llamada = telefono.nueva("negocio", numero="+34954000000", objetivo="mesa para 2")
    asyncio.run(telefono.turno_negocio(llamada["id"], None, False))
    xml = asyncio.run(telefono.turno_negocio(llamada["id"], "Sí, a las nueve tenemos", False))
    guardada = telefono.leer(llamada["id"])
    assert "<Hangup/>" in xml and guardada["estado"] == "hecha" and guardada["resultado"] == "logrado"
    assert [linea["quien"] for linea in guardada["transcripcion"]] == ["perseo", "ellos", "perseo"]


def test_tres_silencios_y_se_cuelga(montado) -> None:
    montado([("decir", {"texto": "¿Hola?"})])
    llamada = telefono.nueva("negocio", numero="+34954000000", objetivo="x")
    for _ in range(2):
        assert "¿Me oye?" in asyncio.run(telefono.turno_negocio(llamada["id"], None, True))
    assert "<Hangup/>" in asyncio.run(telefono.turno_negocio(llamada["id"], None, True))
    assert telefono.leer(llamada["id"])["resultado"] == "no_logrado"


def test_si_comunica_lo_dice_el_aviso_de_twilio(montado) -> None:
    montado([("decir", {"texto": "Hola"})])
    llamada = telefono.nueva("negocio", numero="+34954000000", objetivo="x", sid="CA77")
    telefono.terminada("CA77", "busy")
    assert telefono.leer(llamada["id"])["resumen"] == "Comunicaba."


def test_el_agente_marca_espera_y_cuenta_como_fue(montado, monkeypatch) -> None:
    _, falso = montado([("decir", {"texto": "Hola"})])
    monkeypatch.setattr(telefono, "TOPE_LLAMADA", 5)

    async def prueba() -> dict[str, Any]:
        tarea = asyncio.create_task(router.REGISTRO["telefono"]({"peticion": {
            "accion": "negocio", "numero": "+34 954 00 00 00", "objetivo": "mesa para 2", "negocio": "Casa Prueba",
        }}))
        await asyncio.sleep(0.3)
        [(numero, ruta)] = falso.llamadas
        assert numero == "+34954000000" and ruta.startswith("/twilio/negocio/")
        llamada = telefono.leer(ruta.rsplit("/", 1)[1])
        llamada["transcripcion"] = [{"quien": "perseo", "texto": "Hola"}, {"quien": "ellos", "texto": "Sí"}]
        telefono._cerrar(llamada, "logrado", "Mesa para 2 el viernes a las 21:00.")
        return await tarea

    resultado = asyncio.run(prueba())
    assert resultado["resultado"] == "logrado" and "Mesa para 2" in resultado["texto"]
    assert resultado["titular"] == "Llamada terminada: hecho"  # sin lo hablado, por Telegram


def test_un_numero_sin_prefijo_no_se_marca(montado) -> None:
    _, falso = montado([("decir", {"texto": "Hola"})])
    with pytest.raises(ValueError, match="prefijo"):
        asyncio.run(router.REGISTRO["telefono"]({"peticion": {"accion": "negocio", "numero": "954000000", "objetivo": "x"}}))
    assert falso.llamadas == []


def test_sin_cuenta_de_twilio_se_dice_que_falta(cfg, db, monkeypatch) -> None:
    monkeypatch.setattr(twilio, "cargar", lambda d: None)
    telefono.iniciar(cfg, cerebro=CerebroDeGuion([("decir", {})]))
    with pytest.raises(RuntimeError, match="Twilio"):
        asyncio.run(router.REGISTRO["telefono"]({"peticion": {"accion": "movil"}}))
