"""La marca de presencia: lo que sustituyó a buscar el nombre del binario.

Esta pieza rompía en silencio: el detector buscaba `temp-app.exe` en el
`tasklist`, así que renombrar el paquete Rust dejaba de encontrar la aplicación
—y cada aplauso abría una segunda instancia— sin que nada avisara. Ahora se
pregunta por un PID, y esto lo pincha.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "commands"))

import presencia  # noqa: E402


def test_el_proceso_de_estas_pruebas_esta_vivo() -> None:
    assert presencia._proceso_vivo(os.getpid())


def test_un_pid_imposible_no_esta_vivo() -> None:
    # Ni Windows ni Linux reparten identificadores tan altos.
    assert not presencia._proceso_vivo(4_000_000_000)


@pytest.mark.parametrize("pid", [0, -1])
def test_un_pid_absurdo_no_esta_vivo(pid: int) -> None:
    assert not presencia._proceso_vivo(pid)


def test_un_pid_que_no_cabe_en_el_sistema_no_revienta() -> None:
    """En Linux, `os.kill` con un PID por encima de INT_MAX lanza OverflowError.

    Lo encontró el CI: en Windows pasaba y en Linux tumbaba a quien preguntara.
    Un número que no cabe no es un proceso, es basura en el fichero.
    """
    assert not presencia._proceso_vivo(2**62)


def test_sin_marca_no_hay_pid(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(presencia, "rutas", lambda: [tmp_path / presencia.FICHERO])
    assert presencia.pid_anunciado() is None
    assert not presencia.app_viva()


def test_se_lee_el_pid_de_la_marca(tmp_path: Path, monkeypatch) -> None:
    marca = tmp_path / presencia.FICHERO
    marca.write_text("4242\n", encoding="utf-8")
    monkeypatch.setattr(presencia, "rutas", lambda: [marca])
    assert presencia.pid_anunciado() == 4242


def test_una_marca_ilegible_no_rompe(tmp_path: Path, monkeypatch) -> None:
    marca = tmp_path / presencia.FICHERO
    marca.write_text("no soy un numero", encoding="utf-8")
    monkeypatch.setattr(presencia, "rutas", lambda: [marca])
    assert presencia.pid_anunciado() is None


def test_la_app_esta_viva_si_el_pid_lo_esta(tmp_path: Path, monkeypatch) -> None:
    marca = tmp_path / presencia.FICHERO
    marca.write_text(str(os.getpid()), encoding="utf-8")
    monkeypatch.setattr(presencia, "rutas", lambda: [marca])
    assert presencia.app_viva()


def test_una_marca_rancia_no_engana(tmp_path: Path, monkeypatch) -> None:
    """Lo que se comprueba es el proceso, no el fichero.

    Si la app muere de mala manera el fichero se queda; creerle haría que el
    aplauso dejara de lanzar Perseo para siempre.
    """
    marca = tmp_path / presencia.FICHERO
    marca.write_text("4000000000", encoding="utf-8")
    monkeypatch.setattr(presencia, "rutas", lambda: [marca])
    assert not presencia.app_viva()


def test_limpiar_borra_solo_la_rancia(tmp_path: Path, monkeypatch) -> None:
    marca = tmp_path / presencia.FICHERO
    monkeypatch.setattr(presencia, "rutas", lambda: [marca])

    marca.write_text(str(os.getpid()), encoding="utf-8")
    presencia.limpiar_marca_rancia()
    assert marca.exists()

    marca.write_text("4000000000", encoding="utf-8")
    presencia.limpiar_marca_rancia()
    assert not marca.exists()


def test_se_mira_en_mas_de_un_sitio() -> None:
    """La aplicación escribe en el árbol de fuentes y en su configuración."""
    assert len(presencia.rutas()) >= 2


def test_el_nombre_del_fichero_es_el_mismo_que_en_rust() -> None:
    """Si uno de los dos cambia sin el otro, la detección se rompe en silencio."""
    rust = (
        Path(__file__).resolve().parent.parent
        / "RealTime"
        / "src-tauri"
        / "src"
        / "presencia.rs"
    ).read_text(encoding="utf-8")
    assert f'pub const FICHERO: &str = "{presencia.FICHERO}";' in rust
