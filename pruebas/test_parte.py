"""El parte del día: todo lo de hoy en una respuesta, y lo que falta, dicho.

Lo que no puede pasar: que una sección que no se pudo leer desaparezca sin más
—un parte sin agenda parece un día libre— y que por Telegram viaje el detalle
en vez de los recuentos.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from perseo_core.agentes import agenda, parte
from perseo_core.dominio.evento import Evento
from perseo_core.infra import disparadores, politica
from perseo_core.infra.bus import Bus
from perseo_core.servicios import google_api, recordatorios

ZONA = timezone(timedelta(hours=2))
AHORA = datetime(2026, 9, 23, 8, 30, tzinfo=ZONA)


class CalendarioFalso:
    def __init__(self, eventos=None, error=None) -> None:
        self.eventos = eventos or []
        self.error = error

    async def proximos(self, horizonte):
        if self.error:
            raise self.error
        return self.eventos


def _componer(cfg, monkeypatch, calendario) -> dict:
    monkeypatch.setattr(agenda, "calendario", lambda: calendario)
    return asyncio.run(parte.componer(cfg, AHORA))


def test_el_parte_junta_agenda_recordatorios_tareas_y_habitos(cfg, monkeypatch) -> None:
    recordatorios.crear(cfg.directorio_datos, "Pastilla", AHORA + timedelta(hours=2))
    recordatorios.crear(cfg.directorio_datos, "Mañana no", AHORA + timedelta(days=1))
    cita = Evento(id="1", titulo="Dentista", inicio="2026-09-23T08:00:00Z")
    r = _componer(cfg, monkeypatch, CalendarioFalso([cita]))

    assert "Agenda de hoy (1)" in r["texto"]
    # Google da la hora en UTC; el parte la dice en la de aquí.
    assert "10:00 Dentista" in r["texto"]
    assert "Pastilla" in r["texto"] and "Mañana no" not in r["texto"]
    assert "Tareas:" in r["texto"] and "Hábitos:" in r["texto"]


def test_sin_permiso_de_google_la_agenda_lo_dice(cfg, monkeypatch) -> None:
    error = google_api.TestigoCaducado("Google ya no acepta el permiso guardado.")
    r = _componer(cfg, monkeypatch, CalendarioFalso(error=error))
    assert "Agenda: no se puede mirar" in r["texto"]
    assert "agenda sin mirar" in r["titular"]


def test_sin_calendario_tambien_se_dice(cfg, monkeypatch) -> None:
    r = _componer(cfg, monkeypatch, None)
    assert "no hay calendario configurado" in r["texto"]


def test_por_telegram_solo_van_recuentos(cfg, monkeypatch) -> None:
    cita = Evento(id="1", titulo="Cita secreta", inicio="2026-09-23T10:00:00+02:00")
    r = _componer(cfg, monkeypatch, CalendarioFalso([cita]))
    assert r["titular"].startswith("Parte de hoy: 1 cita(s)")
    assert "secreta" not in r["titular"]


def test_el_parte_solo_lee() -> None:
    assert politica.nivel("parte", {"accion": "dar"}) == politica.LIBRE


# --------------------------------------------------------------------------- #
# El de la mañana
# --------------------------------------------------------------------------- #


def test_toca_entre_la_hora_y_tres_horas_despues() -> None:
    assert parte.toca(AHORA, "08:30")
    assert parte.toca(AHORA + timedelta(hours=2, minutes=59), "08:30")
    assert not parte.toca(AHORA - timedelta(minutes=1), "08:30")
    assert not parte.toca(AHORA + timedelta(hours=3), "08:30")


def test_sin_hora_el_disparador_se_retira(cfg) -> None:
    ctx = disparadores.Contexto(cfg=replace(cfg, parte_hora=""), bus=Bus())
    with pytest.raises(disparadores.Retirarse):
        asyncio.run(parte._vigilar_parte(ctx))


def test_una_hora_rara_tambien_retira(cfg) -> None:
    ctx = disparadores.Contexto(cfg=replace(cfg, parte_hora="pronto"), bus=Bus())
    with pytest.raises(disparadores.Retirarse, match="no es una hora"):
        asyncio.run(parte._vigilar_parte(ctx))


def test_el_de_la_manana_sale_una_vez_al_dia(cfg, monkeypatch) -> None:
    encolados = []

    async def encolar(self, agente_nombre, peticion):
        encolados.append(agente_nombre)
        return {"id": len(encolados)}

    monkeypatch.setattr(disparadores.Contexto, "encolar", encolar)
    monkeypatch.setattr(parte, "_dados", None)
    monkeypatch.setattr(recordatorios, "ahora_local", lambda: AHORA + timedelta(minutes=5))
    ctx = disparadores.Contexto(cfg=replace(cfg, parte_hora="08:30"), bus=Bus())

    asyncio.run(parte._vigilar_parte(ctx))
    asyncio.run(parte._vigilar_parte(ctx))
    assert encolados == ["parte"]
    # Y sobrevive a un reinicio: la marca está en disco.
    monkeypatch.setattr(parte, "_dados", None)
    asyncio.run(parte._vigilar_parte(ctx))
    assert encolados == ["parte"]
    assert (Path(cfg.directorio_datos) / "parte_dados.json").exists()
