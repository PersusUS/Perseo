"""La bóveda: que el modelo nunca vea un valor, y que un valor no salga de su sitio.

Se prueba con `CifradorDePruebas`, que no cifra, para que corra igual en el CI
de Linux; el DPAPI de verdad tiene su propia prueba, solo en Windows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from perseo_core.servicios import boveda as b


@pytest.fixture
def boveda(tmp_path: Path) -> b.Boveda:
    caja = b.Boveda(tmp_path / "boveda.json", b.CifradorDePruebas())
    caja.guardar("resy", "cuenta", ["resy.com"], {"usuario": "ana@correo.es", "clave": "S3creta!"})
    caja.guardar("visa", "tarjeta", ["resy.com"], {"numero": "4111111111111111", "cvc": "737"}, 50)
    return caja


def test_listar_no_lleva_ni_un_valor(boveda: b.Boveda) -> None:
    listado = json.dumps(boveda.listar(), ensure_ascii=False)
    assert "S3creta!" not in listado and "4111111111111111" not in listado and "737" not in listado
    assert "{{boveda:resy.clave}}" in listado
    assert "resy.com" in listado


def test_en_el_fichero_no_hay_nada_en_claro(tmp_path: Path, boveda: b.Boveda) -> None:
    crudo = (tmp_path / "boveda.json").read_text(encoding="utf-8")
    assert "S3creta!" not in crudo and "4111111111111111" not in crudo


def test_sustituye_en_su_sitio_y_en_sus_subdominios(boveda: b.Boveda) -> None:
    texto = "{{boveda:resy.usuario}} / {{boveda:resy.clave}}"
    assert boveda.sustituir(texto, "resy.com") == "ana@correo.es / S3creta!"
    assert boveda.sustituir(texto, "www.resy.com") == "ana@correo.es / S3creta!"


@pytest.mark.parametrize("anfitrion", ["resy.com.malo.es", "noresy.com", "malo.es", ""])
def test_fuera_de_su_sitio_no_se_sustituye(boveda: b.Boveda, anfitrion: str) -> None:
    """El robo más sencillo: una página hostil que pide teclear la clave en su formulario."""
    with pytest.raises(b.ErrorBoveda, match="solo vale en resy.com"):
        boveda.sustituir("{{boveda:resy.clave}}", anfitrion)


def test_una_referencia_que_no_existe_lo_dice(boveda: b.Boveda) -> None:
    with pytest.raises(b.ErrorBoveda):
        boveda.sustituir("{{boveda:gmail.clave}}", "resy.com")
    with pytest.raises(b.ErrorBoveda):
        boveda.usos("{{boveda:resy.pin}}")


def test_tapar_cambia_cada_valor_por_su_referencia(boveda: b.Boveda) -> None:
    pagina = 'textbox "Clave" [ref=e7]: S3creta!\ntextbox "Email": ana@correo.es\nCVC 737'
    tapada = boveda.tapar(pagina)
    assert "S3creta!" not in tapada and "ana@correo.es" not in tapada and "737" not in tapada
    assert "{{boveda:resy.clave}}" in tapada and "{{boveda:visa.cvc}}" in tapada


def test_tapar_encuentra_la_tarjeta_partida_en_grupos(boveda: b.Boveda) -> None:
    for forma in ("4111 1111 1111 1111", "4111-1111-1111-1111"):
        assert "{{boveda:visa.numero}}" in boveda.tapar(f"Tarjeta: {forma}")


def test_tapar_encuentra_la_clave_dentro_de_una_url(boveda: b.Boveda) -> None:
    """Un formulario que manda por GET deja la clave en la dirección siguiente."""
    tapada = boveda.tapar("- Page URL: https://resy.com/entrar?clave=S3creta%21")
    assert "S3creta" not in tapada


def test_usos_distingue_la_tarjeta_y_su_tope(boveda: b.Boveda) -> None:
    usos = boveda.usos("{{boveda:visa.numero}} y {{boveda:resy.clave}}")
    tipos = {(u.nombre, u.tipo, u.limite_euros) for u in usos}
    assert ("visa", "tarjeta", 50) in tipos and ("resy", "cuenta", None) in tipos


def test_una_entrada_sin_sitios_no_se_guarda(tmp_path: Path) -> None:
    """Sin sitios valdría en cualquier web, que es lo que la regla existe para impedir."""
    caja = b.Boveda(tmp_path / "b.json", b.CifradorDePruebas())
    with pytest.raises(b.ErrorBoveda, match="sitio"):
        caja.guardar("resy", "cuenta", [], {"clave": "x"})


def test_un_fichero_roto_no_se_trata_como_vacio(tmp_path: Path) -> None:
    """Devolver vacía y guardar encima perdería todas las entradas."""
    ruta = tmp_path / "b.json"
    ruta.write_text("{roto", encoding="utf-8")
    caja = b.Boveda(ruta, b.CifradorDePruebas())
    with pytest.raises(b.ErrorBoveda):
        caja.guardar("resy", "cuenta", ["resy.com"], {"clave": "x"})
    assert ruta.read_text(encoding="utf-8") == "{roto"


def test_borrar_quita_la_entrada(boveda: b.Boveda) -> None:
    assert boveda.borrar("resy")
    assert not boveda.borrar("resy")
    assert [e["nombre"] for e in boveda.listar()] == ["visa"]


@pytest.mark.skipif(sys.platform != "win32", reason="DPAPI solo existe en Windows")
def test_dpapi_de_verdad_ida_y_vuelta(tmp_path: Path) -> None:
    caja = b.Boveda(tmp_path / "b.json")
    caja.guardar("resy", "cuenta", ["resy.com"], {"clave": "Contraseña con ñ"})
    assert "Contraseña" not in (tmp_path / "b.json").read_text(encoding="utf-8")
    assert caja.sustituir("{{boveda:resy.clave}}", "resy.com") == "Contraseña con ñ"


@pytest.mark.skipif(sys.platform == "win32", reason="fuera de Windows no hay DPAPI")
def test_fuera_de_windows_no_guarda_en_claro(tmp_path: Path) -> None:
    caja = b.Boveda(tmp_path / "b.json")
    with pytest.raises(b.ErrorBoveda, match="DPAPI"):
        caja.guardar("resy", "cuenta", ["resy.com"], {"clave": "x"})
