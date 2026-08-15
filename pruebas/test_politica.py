"""La política de §7: qué nivel tiene cada cosa y cuándo se pregunta."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from perseo_core import politica


def test_leer_es_libre() -> None:
    assert politica.nivel("memoria", {"accion": "buscar"}) == politica.LIBRE
    assert politica.nivel("memoria", {"accion": "leer"}) == politica.LIBRE
    assert politica.nivel("web", {"accion": "leer"}) == politica.LIBRE
    assert politica.nivel("correo", {"accion": "triar"}) == politica.LIBRE


def test_escribir_es_reversible() -> None:
    assert politica.nivel("memoria", {"accion": "anotar"}) == politica.REVERSIBLE
    assert politica.nivel("dev", {"texto": "arregla esto"}) == politica.REVERSIBLE


def test_teclear_a_ciegas_es_irreversible() -> None:
    """Va sobre la ventana que tenga el foco, y esa puede ser cualquiera."""
    assert politica.nivel("pc", {"accion": "escribir_teclado"}) == politica.IRREVERSIBLE
    assert politica.nivel("pc", {"accion": "atajo_teclado"}) == politica.IRREVERSIBLE


def test_abrir_una_app_no_lo_es() -> None:
    assert politica.nivel("pc", {"accion": "abrir_app"}) == politica.LIBRE


def test_lo_que_no_esta_en_la_tabla_pregunta() -> None:
    """La decisión que sostiene el módulo entero."""
    assert politica.nivel("agente_del_futuro", {}) == politica.IRREVERSIBLE
    assert politica.nivel("memoria", {"accion": "borrar"}) == politica.IRREVERSIBLE
    assert politica.nivel("pc", {"accion": "formatear"}) == politica.IRREVERSIBLE


def test_el_simulacro_se_queda_fuera() -> None:
    """Lleva su propia confirmación dentro; con la política encima, dos."""
    assert politica.nivel("simulacro", {"accion": "lo que sea"}) == politica.LIBRE


def test_la_accion_manda_sobre_el_agente() -> None:
    assert politica.nivel("pc", {"accion": "abrir_app"}) != politica.nivel(
        "pc", {"accion": "escribir_teclado"}
    )


def test_sin_peticion_se_mira_el_agente_a_secas() -> None:
    assert politica.nivel("eco") == politica.LIBRE
    assert politica.nivel("dev") == politica.REVERSIBLE


def test_de_entrada_no_hay_confianza() -> None:
    assert not politica.hay_confianza()
    assert politica.confianza_hasta() is None


def test_la_confianza_baja_lo_irreversible() -> None:
    assert politica.pide_confirmacion("pc", {"accion": "escribir_teclado"})
    politica.activar_confianza(30)
    assert not politica.pide_confirmacion("pc", {"accion": "escribir_teclado"})


def test_la_confianza_no_cambia_lo_libre() -> None:
    politica.activar_confianza(30)
    assert not politica.pide_confirmacion("memoria", {"accion": "buscar"})


def test_la_confianza_tiene_tope() -> None:
    hasta = politica.activar_confianza(99999)
    margen = hasta - datetime.now(timezone.utc)
    assert margen <= timedelta(minutes=politica.MAX_MINUTOS_CONFIANZA)


def test_pedir_cero_minutos_da_al_menos_uno() -> None:
    hasta = politica.activar_confianza(0)
    assert hasta > datetime.now(timezone.utc)


def test_una_confianza_caducada_no_vale(tmp_path: Path) -> None:
    politica.iniciar(tmp_path)
    caducada = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    (tmp_path / "confianza.txt").write_text(caducada, encoding="utf-8")

    assert not politica.hay_confianza()
    # Y el fichero se quita de en medio, para que el disco no mienta.
    assert not (tmp_path / "confianza.txt").exists()


def test_un_fichero_ilegible_se_trata_como_sin_confianza(tmp_path: Path) -> None:
    politica.iniciar(tmp_path)
    (tmp_path / "confianza.txt").write_text("esto no es una fecha", encoding="utf-8")
    assert not politica.hay_confianza()


def test_apagar_la_confianza() -> None:
    politica.activar_confianza(30)
    politica.desactivar_confianza()
    assert not politica.hay_confianza()


def test_el_resumen_dice_que_es_sin_soltar_el_detalle() -> None:
    """Sale por Telegram, donde solo va el titular."""
    resumen = politica.resumir("pc", {"accion": "escribir_teclado", "parametro": "mi contraseña"})
    assert "irreversible" in resumen
    assert "pc" in resumen and "escribir_teclado" in resumen
    assert "contraseña" not in resumen
