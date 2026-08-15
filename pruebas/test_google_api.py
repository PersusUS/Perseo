"""Gmail y Calendar: lo que se lee del JSON y lo que se deja fuera."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from perseo_core import agenda, correo, google_api


def escribir(ruta: Path, datos: dict) -> Path:
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    return ruta


def test_las_credenciales_se_leen_planas(tmp_path: Path) -> None:
    ruta = escribir(
        tmp_path / "google.json",
        {"client_id": "id", "client_secret": "s", "refresh_token": "r"},
    )
    assert google_api.Credenciales.desde_fichero(ruta).refresh_token == "r"


def test_las_credenciales_se_leen_dentro_de_installed(tmp_path: Path) -> None:
    """Es como las escupe la consola de Google: no hay que editarlas a mano."""
    ruta = escribir(
        tmp_path / "google.json",
        {"installed": {"client_id": "id", "client_secret": "s", "refresh_token": "r"}},
    )
    assert google_api.Credenciales.desde_fichero(ruta).client_id == "id"


def test_unas_credenciales_incompletas_se_rechazan(tmp_path: Path) -> None:
    ruta = escribir(tmp_path / "google.json", {"client_id": "id"})
    with pytest.raises(google_api.SinCredenciales, match="refresh_token"):
        google_api.Credenciales.desde_fichero(ruta)


def test_un_fichero_que_no_existe_lo_dice(tmp_path: Path) -> None:
    with pytest.raises(google_api.SinCredenciales):
        google_api.Credenciales.desde_fichero(tmp_path / "no_esta.json")


def test_un_fichero_corrupto_lo_dice(tmp_path: Path) -> None:
    ruta = tmp_path / "google.json"
    ruta.write_text("{no es json", encoding="utf-8")
    with pytest.raises(google_api.SinCredenciales):
        google_api.Credenciales.desde_fichero(ruta)


def test_la_cabecera_se_busca_sin_distinguir_mayusculas() -> None:
    cabeceras = [{"name": "from", "value": "a@b.c"}, {"name": "Subject", "value": "Hola"}]
    assert google_api._cabecera(cabeceras, "From") == "a@b.c"
    assert google_api._cabecera(cabeceras, "subject") == "Hola"
    assert google_api._cabecera(cabeceras, "Cc") == ""


def test_la_consulta_de_gmail_deja_fuera_los_chats() -> None:
    """En Gmail los chats también son mensajes, y no son correo."""
    assert "is:unread" in google_api.BuzonGmail.CONSULTA
    assert "-in:chats" in google_api.BuzonGmail.CONSULTA


def test_gmail_como_buzon_sin_credenciales_no_rompe(cfg, monkeypatch) -> None:
    """No configurado no es lo mismo que roto: se avisa y se sigue."""
    from dataclasses import replace

    assert correo.abrir_buzon(replace(cfg, correo_buzon="gmail")) is None


def test_google_como_calendario_sin_credenciales_no_rompe(cfg) -> None:
    from dataclasses import replace

    assert agenda.abrir_calendario(replace(cfg, agenda_origen="google")) is None


def test_gmail_como_buzon_con_credenciales(cfg, tmp_path: Path) -> None:
    from dataclasses import replace

    ruta = escribir(
        tmp_path / "google.json",
        {"client_id": "id", "client_secret": "s", "refresh_token": "r"},
    )
    buzon = correo.abrir_buzon(
        replace(cfg, correo_buzon="gmail", google_credenciales=str(ruta))
    )
    assert isinstance(buzon, google_api.BuzonGmail)


def test_calendario_google_con_credenciales(cfg, tmp_path: Path) -> None:
    from dataclasses import replace

    ruta = escribir(
        tmp_path / "google.json",
        {"client_id": "id", "client_secret": "s", "refresh_token": "r"},
    )
    calendario = agenda.abrir_calendario(
        replace(cfg, agenda_origen="google", google_credenciales=str(ruta))
    )
    assert isinstance(calendario, google_api.CalendarioGoogle)


def test_un_buzon_desconocido_sigue_sin_correo(cfg) -> None:
    from dataclasses import replace

    assert correo.abrir_buzon(replace(cfg, correo_buzon="hotmail")) is None


def test_el_testigo_se_considera_caducado_con_margen() -> None:
    """No usarlo justo en el segundo en que expira evita un 401 evitable."""
    assert google_api.MARGEN_TESTIGO > 0


def test_no_se_piden_mil_mensajes_de_golpe() -> None:
    assert 0 < google_api.TOPE_MENSAJES <= 100


def test_las_urls_se_pueden_apuntar_a_otro_sitio(monkeypatch) -> None:
    """Es lo que permite verificar sin cuenta de Google."""
    monkeypatch.setenv("PERSEO_GOOGLE_GMAIL", "http://127.0.0.1:9/")
    assert google_api.URL_GMAIL() == "http://127.0.0.1:9"


def test_por_defecto_apuntan_a_google(monkeypatch) -> None:
    monkeypatch.delenv("PERSEO_GOOGLE_GMAIL", raising=False)
    monkeypatch.delenv("PERSEO_GOOGLE_OAUTH", raising=False)
    assert google_api.URL_GMAIL().endswith("googleapis.com")
    assert "oauth2" in google_api.URL_TESTIGO()


class _SesionFalsa:
    def __init__(self, respuestas: list[dict]) -> None:
        self._respuestas = respuestas
        self.pedidas: list[tuple[str, dict]] = []

    async def pedir(self, url: str, parametros: dict | None = None) -> dict:
        self.pedidas.append((url, parametros or {}))
        return self._respuestas.pop(0)


def test_el_calendario_deja_fuera_los_eventos_de_todo_el_dia(monkeypatch) -> None:
    """Avisar de uno «a las 00:00» no dice nada útil."""
    import asyncio

    calendario = google_api.CalendarioGoogle(
        google_api.Credenciales("id", "s", "r")
    )
    falsa = _SesionFalsa(
        [
            {
                "items": [
                    {"id": "a", "summary": "Con hora", "start": {"dateTime": "2026-08-16T10:00:00Z"}},
                    {"id": "b", "summary": "Todo el dia", "start": {"date": "2026-08-16"}},
                ]
            }
        ]
    )
    monkeypatch.setattr(calendario, "_abrir", lambda: _corutina(falsa))

    eventos = asyncio.run(calendario.proximos(timedelta(hours=2)))
    assert [e.id for e in eventos] == ["a"]
    # Y se piden las citas concretas, no las series repetidas.
    assert falsa.pedidas[0][1]["singleEvents"] == "true"


async def _corutina(valor):
    return valor
