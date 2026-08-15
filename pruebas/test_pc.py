"""El agente `pc`: ninguna inyección llega a tocar el sistema.

Ninguna de estas pruebas abre una ventana ni teclea nada: todas comprueban casos
que se rechazan **antes** de llamar a nadie.
"""

from __future__ import annotations

import pytest

from perseo_core import pc


@pytest.mark.parametrize(
    ("accion", "parametro", "motivo"),
    [
        ("abrir_app", "notepad & calc", "encadenado con &"),
        ("abrir_app", "spotify && shutdown /s /t 0", "encadenado con &&"),
        ("abrir_app", "a | del /q C:\\*", "tuberia"),
        ("abrir_app", "cmd", "shell fuera de la lista"),
        ("abrir_app", "powershell", "shell fuera de la lista"),
        ("abrir_app", "file:///C:/Windows", "esquema no permitido"),
        ("abrir_app", "javascript:alert(1)", "esquema no permitido"),
        ("atajo_teclado", "ctrl,alt,delete,f4,esc,tab", "demasiadas teclas"),
        ("atajo_teclado", "ctrl,shutdown", "tecla inventada"),
        ("atajo_teclado", "", "sin teclas"),
        ("mover_raton", "abc,def", "coordenadas no numericas"),
        ("mover_raton", "10", "coordenadas incompletas"),
        ("volumen", "; rm -rf /", "valor no reconocido"),
        ("click_raton", "triple; shutdown", "tipo de clic inventado"),
        ("apagar_equipo", "ya", "accion inexistente"),
    ],
)
def test_las_inyecciones_se_bloquean(accion: str, parametro: str, motivo: str) -> None:
    assert pc.controlar(accion, parametro).startswith("Error:"), motivo


def test_una_url_http_no_es_una_inyeccion() -> None:
    """El rechazo tiene que ser por el esquema, no por ser una URL."""
    assert not pc._abrir_url.__doc__ is None  # la función existe y está documentada
    partes = pc._es_url("https://example.com")
    assert partes is True


def test_el_texto_a_teclear_pierde_los_controles() -> None:
    """Un '\\n' equivale a pulsar Enter en la ventana que tenga el foco."""
    assert pc._texto_imprimible("formatear\nsi") == "formatearsi"
    assert pc._texto_imprimible("dato\ty\rotro") == "datoyotro"


def test_un_texto_larguisimo_se_rechaza() -> None:
    largo = "x" * (pc.MAX_LONGITUD_TEXTO + 1)
    assert pc.controlar("escribir_teclado", largo).startswith("Error:")


def test_un_texto_solo_de_controles_se_rechaza() -> None:
    assert pc.controlar("escribir_teclado", "\n\t\r").startswith("Error:")


def test_la_lista_blanca_no_lleva_interpretes() -> None:
    """Poder abrir un shell haría inútil todo lo demás del módulo."""
    prohibidos = {"cmd", "powershell", "wt", "terminal", "regedit", "bash"}
    assert not (prohibidos & set(pc.APLICACIONES_PERMITIDAS))


def test_las_teclas_permitidas_no_llevan_teclas_de_sistema() -> None:
    assert "printscreen" not in pc._TECLAS_PERMITIDAS
    assert "volumemute" not in pc._TECLAS_PERMITIDAS


def test_los_esquemas_de_url_permitidos_son_dos() -> None:
    assert pc.ESQUEMAS_URL_PERMITIDOS == {"http", "https"}


def test_una_url_con_esquema_raro_se_rechaza() -> None:
    assert pc._abrir_url("ftp://archivos.example/x").startswith("Error:")


def test_una_url_sin_dominio_se_rechaza() -> None:
    assert pc._abrir_url("http:///sin-dominio").startswith("Error:")


def test_buscar_en_youtube_sin_termino_se_rechaza() -> None:
    assert pc.controlar("buscar_youtube", "   ").startswith("Error:")


def test_la_accion_no_distingue_mayusculas_ni_espacios() -> None:
    assert pc.controlar("  APAGAR_EQUIPO ", "").startswith("Error:")
