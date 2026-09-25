"""Qué ventana se trae al frente tras abrir algo: la nueva, y si no nace ninguna, la del programa."""

from __future__ import annotations

from perseo_core.servicios import ventanas


def test_la_ventana_nueva_del_programa_gana_a_las_que_ya_estaban() -> None:
    antes = {1, 2}
    ahora = [(1, "chrome.exe"), (2, "Code.exe"), (3, "chrome.exe")]
    assert ventanas.elegir(antes, ahora, ("chrome.exe",)) == 3


def test_si_el_programa_ya_estaba_abierto_se_trae_su_ventana() -> None:
    """WhatsApp abierto no crea ventana: solo se activa, y se quedaba parpadeando."""
    antes = {1, 2}
    ahora = [(1, "Code.exe"), (2, "WhatsApp.Root.exe")]
    assert ventanas.elegir(antes, ahora, ("WhatsApp.exe", "WhatsApp.Root.exe")) == 2


def test_el_nombre_del_ejecutable_no_distingue_mayusculas() -> None:
    assert ventanas.elegir(set(), [(7, "Notepad.exe")], ("notepad.exe",)) == 7


def test_una_ventana_nueva_de_otro_ejecutable_vale_si_no_hay_nada_del_programa() -> None:
    """Steam u Office nacen de un lanzador con otro nombre."""
    assert ventanas.elegir({1}, [(1, "Code.exe"), (5, "ApplicationFrameHost.exe")], ("calc.exe",)) == 5


def test_sin_nada_nuevo_ni_del_programa_no_se_toca_nada() -> None:
    assert ventanas.elegir({1}, [(1, "Code.exe")], ("chrome.exe",)) is None
    assert ventanas.elegir({1}, [(1, "Code.exe")], ()) is None


def test_fuera_de_windows_no_hace_nada(monkeypatch) -> None:
    monkeypatch.setattr(ventanas.os, "name", "posix")
    assert ventanas.traer_al_frente(set(), ("chrome.exe",), plazo=0.1) is False
