"""Vigilancias: mirar una web cada tanto, callar mientras no pase nada, y avisar o actuar cuando pasa."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from perseo_core.agentes import recado, web
from perseo_core.infra.bus import Evento
from perseo_core.infra.router import REGISTRO
from perseo_core.servicios import boveda as boveda_mod
from perseo_core.servicios import vigilancias as v

from test_llamada_saliente import _correr_avisador, _marcador
from test_recado import CerebroGuion, ManosFalsas

AHORA = datetime(2026, 9, 24, 10, 0, tzinfo=timezone(timedelta(hours=2)))
ENTRADAS = "https://tickets.example/coldplay"


def _crear(dir_: Path, **extra) -> dict:
    return v.crear(dir_, "Entradas de Coldplay en tickets.example", "hay entradas a la venta", AHORA, **extra)


# --------------------------------------------------------------------------- #
# El servicio
# --------------------------------------------------------------------------- #


def test_la_primera_comprobacion_es_enseguida(tmp_path: Path) -> None:
    creada = _crear(tmp_path)
    tocan, caducadas = v.para_comprobar(tmp_path, AHORA)
    assert [t["id"] for t in tocan] == [creada["id"]] and caducadas == []


def test_al_encolarla_se_adelanta_la_siguiente(tmp_path: Path) -> None:
    """Si la comprobación tarda, la vuelta siguiente del disparador no encola otra encima."""
    _crear(tmp_path, cada_horas=2)
    v.para_comprobar(tmp_path, AHORA)
    assert v.para_comprobar(tmp_path, AHORA + timedelta(minutes=59)) == ([], [])
    assert len(v.para_comprobar(tmp_path, AHORA + timedelta(hours=2))[0]) == 1


def test_menos_de_una_hora_no_se_deja(tmp_path: Path) -> None:
    assert _crear(tmp_path, cada_horas=0.1)["cada_horas"] == v.HORAS_MINIMO


def test_no_mas_de_cinco_a_la_vez(tmp_path: Path) -> None:
    for _ in range(v.MAXIMO_ACTIVAS):
        _crear(tmp_path)
    with pytest.raises(ValueError, match="tope"):
        _crear(tmp_path)


def test_el_tope_diario_se_reparte_entre_todas(tmp_path: Path) -> None:
    """Cada comprobación son varias peticiones a Gemini, del mismo cubo que el chat."""
    for _ in range(v.MAXIMO_ACTIVAS):
        _crear(tmp_path, cada_horas=1)
    hechas = 0
    for hora in range(24):
        hechas += len(v.para_comprobar(tmp_path, AHORA.replace(hour=0) + timedelta(hours=hora, minutes=1))[0])
    assert hechas == v.TOPE_DIARIO
    # Al día siguiente, el cupo vuelve.
    assert v.para_comprobar(tmp_path, AHORA + timedelta(days=1))[0]


def test_la_que_caduca_se_dice_una_vez(tmp_path: Path) -> None:
    _crear(tmp_path, dias=1)
    tocan, caducadas = v.para_comprobar(tmp_path, AHORA + timedelta(days=2))
    assert tocan == [] and len(caducadas) == 1
    assert v.para_comprobar(tmp_path, AHORA + timedelta(days=3)) == ([], [])


def test_al_cumplirse_se_cierra(tmp_path: Path) -> None:
    creada = _crear(tmp_path)
    v.apuntar(tmp_path, creada["id"], True, "a la venta, 80 €", AHORA)
    assert v.activas(tmp_path) == []
    assert v.obtener(tmp_path, creada["id"])["estado"] == "cumplida"


def test_cancelar_por_el_principio(tmp_path: Path) -> None:
    _crear(tmp_path)
    assert v.cancelar(tmp_path, "entradas de coldplay")["estado"] == "cancelada"
    assert v.activas(tmp_path) == []


def test_sin_condicion_no_se_apunta(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cuándo avisar"):
        v.crear(tmp_path, "algo", "", AHORA)


# --------------------------------------------------------------------------- #
# El agente y su disparador
# --------------------------------------------------------------------------- #


def _gestionar(**peticion) -> dict:
    return asyncio.run(REGISTRO["vigilancias"]({"id": 1, "peticion": {"accion": "gestionar", **peticion}}))


def test_crear_listar_y_cancelar_por_la_herramienta(cfg) -> None:
    from perseo_core.agentes import vigilancias as agente

    agente.iniciar(cfg)
    creada = _gestionar(que="crear", objetivo="Mesa en Casa Prueba el viernes", condicion="hay mesa para 2")
    assert "Vigilando «Mesa en Casa Prueba el viernes»" in creada["texto"]
    assert "Mesa en Casa Prueba" in _gestionar(que="listar")["texto"]
    assert "Dejo de vigilar" in _gestionar(que="cancelar", objetivo="mesa en casa")["texto"]
    assert "ninguna" in _gestionar(que="listar")["texto"]


def test_el_disparador_encola_un_recado_en_modo_vigilar(cfg, db) -> None:
    from perseo_core.agentes import vigilancias as agente
    from perseo_core.infra import almacen, disparadores
    from perseo_core.infra.bus import Bus

    creada = v.crear(cfg.directorio_datos, "Entradas", "a la venta", datetime.now().astimezone())
    asyncio.run(agente._vigilar(disparadores.Contexto(cfg=cfg, bus=Bus())))
    trabajo = almacen.reclamar()
    assert trabajo["agente"] == "recado" and trabajo["origen"] == "disparador"
    assert trabajo["peticion"] == {"texto": "Entradas", "accion": "vigilar", "vigilancia": creada["id"]}


# --------------------------------------------------------------------------- #
# La comprobación, que es un recado
# --------------------------------------------------------------------------- #


@pytest.fixture
def vigilar(cfg, monkeypatch):
    boveda_mod.iniciar(cfg.directorio_datos, boveda_mod.CifradorDePruebas())

    async def comprobar(url, permitir_local=False, fijar=None):
        return url

    monkeypatch.setattr(web, "comprobar_url", comprobar)

    def montar(turnos, al_cumplirse="avisar"):
        creada = v.crear(
            cfg.directorio_datos, "Entradas de Coldplay", "hay entradas", datetime.now().astimezone(),
            al_cumplirse=al_cumplirse,
        )
        paginas = {ENTRADAS: [("e3", "heading", "Agotado", None), ("e4", "button", "Comprar 80 €", None)]}
        cerebro, manos = CerebroGuion(turnos), ManosFalsas(json.loads(json.dumps(paginas)))
        recado.iniciar(cfg, cerebro=cerebro, manos=manos)
        trabajo = {
            "id": 40, "agente": "recado", "origen": "disparador",
            "peticion": {"texto": creada["objetivo"], "accion": "vigilar", "vigilancia": creada["id"]},
        }
        return creada, cerebro, manos, trabajo

    yield montar
    asyncio.run(recado.detener())


def _correr(trabajo: dict) -> dict:
    return asyncio.run(REGISTRO["recado"](trabajo))


def test_sin_novedad_se_calla(vigilar, cfg) -> None:
    creada, cerebro, _, trabajo = vigilar([
        [("browser_navigate", {"url": ENTRADAS})],
        [("informar", {"cumple": False, "detalle": "sigue agotado"})],
    ])
    resultado = _correr(trabajo)
    assert resultado["estado"] == "sin_novedad" and resultado["callado"] and resultado["titular"] is None
    guardada = v.obtener(cfg.directorio_datos, creada["id"])
    assert guardada["estado"] == "activa" and guardada["ultima"]["detalle"] == "sigue agotado"
    # El modelo recibió la condición y la herramienta para contestarla.
    assert "hay entradas" in cerebro.vistos[0]


def test_al_cumplirse_avisa_sin_contar_el_que_por_telegram(vigilar, cfg) -> None:
    creada, _, manos, trabajo = vigilar([
        [("browser_navigate", {"url": ENTRADAS})],
        [("informar", {"cumple": True, "detalle": "a la venta, 80 €"})],
    ])
    resultado = _correr(trabajo)
    assert resultado["estado"] == "cumplida"
    assert resultado["titular"] == "Se ha cumplido una vigilancia"  # lo que va por un tercero
    assert "80 €" in resultado["aviso"]  # lo que dice la llamada, en casa
    assert v.obtener(cfg.directorio_datos, creada["id"])["estado"] == "cumplida"
    assert manos.hechas("browser_click") == []


def test_si_hay_que_hacerlo_sigue_y_lo_que_paga_espera(vigilar) -> None:
    from perseo_core.infra.router import NecesitaConfirmacion

    _, _, manos, trabajo = vigilar(
        [
            [("browser_navigate", {"url": ENTRADAS})],
            [("informar", {"cumple": True, "detalle": "a la venta"})],
            [("browser_click", {"target": "e4", "element": "comprar"})],
        ],
        al_cumplirse="hacer",
    )
    with pytest.raises(NecesitaConfirmacion):
        _correr(trabajo)
    assert manos.hechas("browser_click") == []


def test_una_web_que_no_se_deja_leer_no_se_come_la_cuota(vigilar, cfg) -> None:
    creada, cerebro, _, trabajo = vigilar([[("browser_snapshot", {})]] * 20)
    resultado = _correr(trabajo)
    assert resultado["estado"] == "sin_novedad" and cerebro.veces == recado.TOPE_PASOS_VIGILAR
    assert "no se pudo comprobar" in v.obtener(cfg.directorio_datos, creada["id"])["ultima"]["detalle"]


def test_una_vigilancia_cancelada_no_se_comprueba(vigilar, cfg) -> None:
    creada, cerebro, _, trabajo = vigilar([[("informar", {"cumple": True, "detalle": "x"})]])
    v.cancelar(cfg.directorio_datos, creada["id"])
    assert _correr(trabajo)["estado"] == "sin_vigilancia"
    assert cerebro.veces == 0


def test_informar_fuera_de_una_vigilancia_no_hace_nada(cfg, monkeypatch) -> None:
    boveda_mod.iniciar(cfg.directorio_datos, boveda_mod.CifradorDePruebas())
    cerebro = CerebroGuion([[("informar", {"cumple": True, "detalle": "x"})], [("terminar", {"resumen": "ok"})]])
    recado.iniciar(cfg, cerebro=cerebro, manos=ManosFalsas({}))
    try:
        assert _correr({"id": 3, "peticion": {"texto": "algo"}})["estado"] == "hecho"
        assert "solo vale en una vigilancia" in cerebro.vistos[-1]
    finally:
        asyncio.run(recado.detener())


# --------------------------------------------------------------------------- #
# El timbre
# --------------------------------------------------------------------------- #


def test_una_comprobacion_callada_no_hace_sonar_el_timbre() -> None:
    evento = Evento(
        tipo="trabajo.hecho",
        datos={"trabajo": {"id": 950, "agente": "recado", "peticion": {"texto": "x"},
                           "resultado": {"estado": "sin_novedad", "callado": True, "titular": None}}},
    )
    bus = _correr_avisador([evento])
    assert not _marcador().exists()
    assert "aviso.encargo" not in bus.publicados


def test_la_llamada_dice_el_aviso_entero() -> None:
    evento = Evento(
        tipo="trabajo.hecho",
        datos={"trabajo": {"id": 951, "agente": "recado", "peticion": {"texto": "Entradas"},
                           "resultado": {"titular": "Se ha cumplido una vigilancia",
                                         "aviso": "Se ha cumplido una vigilancia: a la venta, 80 €"}}},
    )
    _correr_avisador([evento])
    assert "a la venta, 80 €" in _marcador().read_text(encoding="utf-8")
