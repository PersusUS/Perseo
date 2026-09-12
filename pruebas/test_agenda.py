"""La agenda: lo que viene, una sola vez, y sin soltar el título."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from perseo_core.agentes import agenda
from perseo_core.infra import almacen


def dentro_de(minutos: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=minutos)).isoformat()


def escribir(ruta: Path, eventos: list[dict]) -> Path:
    ruta.write_text(json.dumps(eventos), encoding="utf-8")
    return ruta


def test_solo_entra_lo_que_cae_dentro_del_horizonte(tmp_path: Path) -> None:
    fichero = escribir(
        tmp_path / "agenda.json",
        [
            {"id": "pronto", "titulo": "A", "inicio": dentro_de(20)},
            {"id": "lejos", "titulo": "B", "inicio": dentro_de(60 * 24)},
        ],
    )
    eventos = asyncio.run(agenda.CalendarioFalso(fichero).proximos(timedelta(minutes=60)))
    assert [e.id for e in eventos] == ["pronto"]


def test_lo_que_ya_empezo_no_se_avisa(tmp_path: Path) -> None:
    """Avisar de algo a lo que llegas tarde es ruido, no información."""
    fichero = escribir(
        tmp_path / "agenda.json", [{"id": "pasado", "titulo": "A", "inicio": dentro_de(-30)}]
    )
    assert asyncio.run(agenda.CalendarioFalso(fichero).proximos(timedelta(minutes=60))) == []


def test_una_fecha_ilegible_no_rompe_nada(tmp_path: Path) -> None:
    fichero = escribir(
        tmp_path / "agenda.json",
        [
            {"id": "roto", "titulo": "A", "inicio": "el martes"},
            {"id": "bueno", "titulo": "B", "inicio": dentro_de(10)},
        ],
    )
    eventos = asyncio.run(agenda.CalendarioFalso(fichero).proximos(timedelta(minutes=60)))
    assert [e.id for e in eventos] == ["bueno"]


def test_los_eventos_salen_ordenados(tmp_path: Path) -> None:
    fichero = escribir(
        tmp_path / "agenda.json",
        [
            {"id": "b", "titulo": "B", "inicio": dentro_de(40)},
            {"id": "a", "titulo": "A", "inicio": dentro_de(10)},
        ],
    )
    eventos = asyncio.run(agenda.CalendarioFalso(fichero).proximos(timedelta(minutes=60)))
    assert [e.id for e in eventos] == ["a", "b"]


def test_un_calendario_corrupto_no_rompe(tmp_path: Path) -> None:
    fichero = tmp_path / "agenda.json"
    fichero.write_text("{no es json", encoding="utf-8")
    assert asyncio.run(agenda.CalendarioFalso(fichero).proximos(timedelta(minutes=60))) == []


def test_un_calendario_que_no_existe_no_rompe(tmp_path: Path) -> None:
    calendario = agenda.CalendarioFalso(tmp_path / "no_esta.json")
    assert asyncio.run(calendario.proximos(timedelta(minutes=60))) == []


def test_la_fecha_sin_zona_se_interpreta_como_local() -> None:
    evento = agenda.Evento(id="x", titulo="A", inicio="2026-08-16T10:00:00")
    assert evento.momento is not None
    assert evento.momento.tzinfo is timezone.utc


def test_una_fecha_con_z_se_entiende() -> None:
    evento = agenda.Evento(id="x", titulo="A", inicio="2026-08-16T10:00:00Z")
    assert evento.momento == datetime(2026, 8, 16, 10, 0, tzinfo=timezone.utc)


def test_una_fecha_imposible_devuelve_nada() -> None:
    assert agenda.Evento(id="x", titulo="A", inicio="pasado mañana").momento is None


def test_el_titular_no_lleva_el_titulo_del_evento() -> None:
    """Es contenido del calendario: se lee por el tailnet, no por Telegram."""
    evento = agenda.Evento(id="x", titulo="Revision medica", inicio=dentro_de(20))
    texto = agenda.titular([evento])
    assert texto.startswith("1 evento a las")
    assert "Revision medica" not in texto


def test_el_titular_en_plural() -> None:
    eventos = [
        agenda.Evento(id="a", titulo="A", inicio=dentro_de(10)),
        agenda.Evento(id="b", titulo="B", inicio=dentro_de(20)),
    ]
    assert agenda.titular(eventos).startswith("2 eventos")


def test_sin_eventos_no_hay_titular() -> None:
    assert agenda.titular([]) is None


def test_abrir_calendario_sin_configurar_devuelve_nada(cfg) -> None:
    assert agenda.abrir_calendario(cfg) is None


def test_el_agente_devuelve_los_eventos_y_el_titular() -> None:
    resultado = asyncio.run(
        agenda._agenda(
            {
                "peticion": {
                    "accion": "avisar",
                    "eventos": [{"id": "x", "titulo": "Cita", "inicio": dentro_de(15)}],
                }
            }
        )
    )
    assert len(resultado["eventos"]) == 1
    # El detalle se queda en la cola; el titular es lo que viaja.
    assert resultado["eventos"][0]["titulo"] == "Cita"
    assert "Cita" not in resultado["titular"]


def test_el_agente_sin_eventos_no_avisa() -> None:
    resultado = asyncio.run(agenda._agenda({"peticion": {"accion": "avisar", "eventos": []}}))
    assert resultado["titular"] is None


# -- La acción `proximos`: lo que hay, leído bajo demanda (tools del live) ---- #


def test_proximos_lee_del_calendario_de_verdad(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fichero = escribir(
        tmp_path / "agenda.json", [{"id": "x", "titulo": "Cita", "inicio": dentro_de(30)}]
    )
    monkeypatch.setenv("PERSEO_AGENDA", "falso")
    monkeypatch.setenv("PERSEO_AGENDA_FALSA", str(fichero))
    asyncio.run(agenda.detener())
    agenda.iniciar(almacen.cargar_configuracion())
    try:
        resultado = asyncio.run(agenda._agenda({"peticion": {"accion": "proximos"}}))
    finally:
        asyncio.run(agenda.detener())
    assert [e["id"] for e in resultado["eventos"]] == ["x"]
    assert resultado["horas"] == agenda.HORAS_POR_DEFECTO


def test_proximos_recorta_el_horizonte_al_techo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """«¿Qué tengo este mes?» no cabe en una respuesta hablada."""
    monkeypatch.setenv("PERSEO_AGENDA", "falso")
    monkeypatch.setenv("PERSEO_AGENDA_FALSA", str(tmp_path / "vacio.json"))
    asyncio.run(agenda.detener())
    agenda.iniciar(almacen.cargar_configuracion())
    try:
        resultado = asyncio.run(
            agenda._agenda({"peticion": {"accion": "proximos", "horas": 10_000}})
        )
    finally:
        asyncio.run(agenda.detener())
    assert resultado["horas"] == agenda.HORAS_MAXIMAS


def test_proximos_sin_calendario_es_un_error_claro(cfg) -> None:
    asyncio.run(agenda.detener())
    agenda.iniciar(cfg)
    try:
        asyncio.run(agenda._agenda({"peticion": {"accion": "proximos"}}))
        raise AssertionError("debía fallar: no hay calendario configurado")
    except RuntimeError as e:
        assert "PERSEO_AGENDA" in str(e)
    finally:
        asyncio.run(agenda.detener())


def test_proximos_sin_iniciar_avisa_de_lo_que_falta() -> None:
    asyncio.run(agenda.detener())
    try:
        asyncio.run(agenda._agenda({"peticion": {"accion": "proximos"}}))
        raise AssertionError("debía fallar: el agente no está iniciado")
    except RuntimeError as e:
        assert "iniciar" in str(e)
