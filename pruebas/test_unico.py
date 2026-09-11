"""La cerradura de una pieza: uno y nada más, aunque falle quien lo comprueba.

El 2026-08-26 había TRES detectores de aplausos sobre el mismo micrófono, y no
por un fallo del detector: `perseo.arrancar_detector()` pregunta si hay otro con
una consulta WMI que a veces falla, y cuando falla contesta «no está» y arranca
uno más — cada diez minutos, que es cada cuánto lo intenta `PerseoRevivir`.
La cerradura existe para que esa pregunta pueda fallar sin consecuencias.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "commands"))

import unico  # noqa: E402


@pytest.fixture()
def nombre() -> str:
    """Un nombre de cerradura de usar y tirar, distinto en cada prueba."""
    import uuid

    return f"PerseoPrueba-{uuid.uuid4().hex[:12]}"


def test_el_primero_la_toma(nombre: str) -> None:
    cerrojo = unico.tomar(nombre)
    assert cerrojo is not None
    cerrojo.soltar()


def test_el_segundo_se_queda_fuera(nombre: str) -> None:
    """Es lo único que se le pide: que el segundo NO pueda."""
    primero = unico.tomar(nombre)
    try:
        assert unico.tomar(nombre) is None
    finally:
        primero.soltar()


def test_al_soltarla_vuelve_a_estar_libre(nombre: str) -> None:
    """Si no, un detector que se cuelga deja la pieza muerta para siempre — que
    es exactamente el fallo que tendría un fichero de PID sin borrar."""
    primero = unico.tomar(nombre)
    primero.soltar()
    segundo = unico.tomar(nombre)
    assert segundo is not None
    segundo.soltar()


def test_soltarla_dos_veces_no_revienta(nombre: str) -> None:
    cerrojo = unico.tomar(nombre)
    cerrojo.soltar()
    cerrojo.soltar()


def test_sin_cerradura_se_sigue_adelante(monkeypatch, nombre: str) -> None:
    """Quedarse sin detector de aplausos por no poder crear un mutex sería
    cambiar un problema por otro peor: si el sistema falla, se deja pasar."""

    def revienta(_nombre: str):
        raise OSError("no hay mutex hoy")

    monkeypatch.setattr(unico, "_tomar_en_windows", revienta)
    monkeypatch.setattr(unico, "_tomar_en_posix", revienta)
    with pytest.raises(OSError):
        unico._tomar_en_windows(nombre)
    # Y aun así, la puerta de arriba no deja caer el fallo al que llama.
    monkeypatch.setattr(unico, "_tomar_en_windows", lambda n: unico.Cerrojo(None))
    monkeypatch.setattr(unico, "_tomar_en_posix", lambda n: unico.Cerrojo(None))
    cerrojo = unico.tomar(nombre)
    assert cerrojo is not None
    cerrojo.soltar()


def test_el_vigilante_se_retira_si_ya_hay_otro(tmp_path: Path, monkeypatch) -> None:
    """Dos vigilantes son dos núcleos peleándose por el 8787, y el bucle de
    arranques fallidos que sale de ahí no lo para nadie."""
    import vigilante

    monkeypatch.setenv("PERSEO_CORE_DATOS", str(tmp_path))
    monkeypatch.setattr(vigilante.unico, "tomar", lambda _nombre: None)
    assert vigilante.vigilar() == 0
    escrito = (tmp_path / "vigilante.log").read_text(encoding="utf-8")
    assert "otro vigilante" in escrito
