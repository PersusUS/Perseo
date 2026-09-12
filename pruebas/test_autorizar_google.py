"""El consentimiento de Google: lo que se pide, y lo que se guarda.

El paseo entero —navegador, vuelta al bucle local y canje del código— se recorre
en `verificar_google.py` contra un Google de mentira. Aquí está lo que se puede
romper sin que nadie se entere: pedir un ámbito de más, o dejar un fichero que
luego `google_api` no sepa leer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from perseo_core.servicios import autorizar_google, google_api


def escribir(ruta: Path, datos: dict) -> Path:
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    return ruta


def test_se_lee_el_json_tal_cual_lo_da_la_consola(tmp_path: Path) -> None:
    """Con las claves dentro de `installed`, para no editarlo a mano."""
    ruta = escribir(
        tmp_path / "google.json", {"installed": {"client_id": "id", "client_secret": "s"}}
    )
    assert autorizar_google.leer_cliente(ruta) == ("id", "s")


def test_tambien_se_lee_plano(tmp_path: Path) -> None:
    ruta = escribir(tmp_path / "google.json", {"client_id": "id", "client_secret": "s"})
    assert autorizar_google.leer_cliente(ruta) == ("id", "s")


def test_aqui_el_refresh_token_no_hace_falta(tmp_path: Path) -> None:
    """Es lo que se viene a buscar: exigirlo dejaría el ayudante sin sentido."""
    ruta = escribir(tmp_path / "google.json", {"client_id": "id", "client_secret": "s"})
    with pytest.raises(google_api.SinCredenciales):
        google_api.Credenciales.desde_fichero(ruta)
    assert autorizar_google.leer_cliente(ruta)


def test_un_fichero_incompleto_dice_que_falta(tmp_path: Path) -> None:
    ruta = escribir(tmp_path / "google.json", {"client_id": "id"})
    with pytest.raises(autorizar_google.SinConsentimiento, match="client_secret"):
        autorizar_google.leer_cliente(ruta)


def test_un_fichero_que_no_existe_dice_de_donde_sale(tmp_path: Path) -> None:
    with pytest.raises(autorizar_google.SinConsentimiento, match="consola"):
        autorizar_google.leer_cliente(tmp_path / "no_esta.json")


def test_un_fichero_corrupto_lo_dice(tmp_path: Path) -> None:
    ruta = tmp_path / "google.json"
    ruta.write_text("{no es json", encoding="utf-8")
    with pytest.raises(autorizar_google.SinConsentimiento):
        autorizar_google.leer_cliente(ruta)


def test_se_pide_leer_y_escribir_borradores_y_nada_mas() -> None:
    """Con este testigo no se puede **enviar** un correo aunque alguien lo intente.

    `gmail.compose` se añadió el 2026-08-16 y es el ámbito más pequeño que
    escribe un borrador: no incluye `send`, que manda, ni `modify`, que además
    borra. Del calendario se sigue pidiendo solo lectura.
    """
    assert autorizar_google.AMBITOS == (
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/calendar.readonly",
        "https://www.googleapis.com/auth/gmail.compose",
    )
    url = autorizar_google.url_de_consentimiento("id", "http://127.0.0.1:1234/")
    assert "gmail.readonly" in url and "calendar.readonly" in url
    assert "gmail.compose" in url
    assert "gmail.send" not in url and "gmail.modify" not in url


def test_la_url_pide_lo_que_hace_falta_para_un_refresh_token() -> None:
    """Sin `offline` no lo dan, y sin `consent` dejan de darlo a la segunda vez."""
    url = autorizar_google.url_de_consentimiento("id", "http://127.0.0.1:1234/")
    assert "access_type=offline" in url
    assert "prompt=consent" in url
    assert "response_type=code" in url


def test_la_direccion_de_vuelta_viaja_escapada() -> None:
    url = autorizar_google.url_de_consentimiento("id", "http://127.0.0.1:1234/")
    assert "redirect_uri=http%3A%2F%2F127.0.0.1%3A1234%2F" in url


def test_lo_guardado_es_lo_que_google_api_sabe_leer(tmp_path: Path) -> None:
    """Los dos extremos del fichero, atados: si uno cambia, esto se entera."""
    ruta = tmp_path / "datos" / "google.json"
    autorizar_google.guardar(ruta, "id", "secreto", "refresco")

    credenciales = google_api.Credenciales.desde_fichero(ruta)
    assert credenciales.client_id == "id"
    assert credenciales.client_secret == "secreto"
    assert credenciales.refresh_token == "refresco"


def test_guardar_crea_el_directorio_si_no_esta(tmp_path: Path) -> None:
    ruta = tmp_path / "sin" / "crear" / "google.json"
    autorizar_google.guardar(ruta, "id", "s", "r")
    assert ruta.is_file()
