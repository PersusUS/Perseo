"""El agente `dev`: de la raíz no se sale, y las listas son las que son."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from perseo_core import almacen, dev


@pytest.fixture()
def dev_falso(cfg: almacen.Configuracion, tmp_path: Path, monkeypatch):
    """Motor de mentira y una raíz de usar y tirar."""
    raiz = tmp_path / "proyecto"
    (raiz / "dentro").mkdir(parents=True)
    monkeypatch.setenv("PERSEO_DEV_MOTOR", "falso")
    monkeypatch.setenv("PERSEO_DEV_RAIZ", str(raiz))
    nueva = almacen.cargar_configuracion()

    monkeypatch.setattr(dev, "_motor", None)
    dev.iniciar(nueva)
    yield raiz
    dev.detener()


def test_con_motor_falso_hay_motor(dev_falso) -> None:
    assert isinstance(dev._motor, dev.MotorFalso)


def test_sin_directorio_el_encargo_va_a_la_raiz(dev_falso: Path) -> None:
    assert dev.resolver_raiz("") == dev_falso.resolve()


def test_un_subdirectorio_vale(dev_falso: Path) -> None:
    assert dev.resolver_raiz("dentro") == (dev_falso / "dentro").resolve()


@pytest.mark.parametrize("intento", ["..", "../..", "dentro/../..", "C:\\Windows"])
def test_de_la_raiz_no_se_sale(dev_falso, intento: str) -> None:
    """Un encargo puede venir de un correo."""
    with pytest.raises(ValueError):
        dev.resolver_raiz(intento)


def test_un_directorio_que_no_existe_se_rechaza(dev_falso) -> None:
    with pytest.raises(ValueError):
        dev.resolver_raiz("no_existe")


def test_git_push_esta_denegado() -> None:
    """Publicar es del usuario, no del agente."""
    assert any("git push" in h for h in dev.HERRAMIENTAS_DENEGADAS)


def test_borrar_esta_denegado() -> None:
    assert any(h.startswith("Bash(rm ") for h in dev.HERRAMIENTAS_DENEGADAS)
    assert any("git reset --hard" in h for h in dev.HERRAMIENTAS_DENEGADAS)


def test_no_hay_un_bash_abierto_entre_las_permitidas() -> None:
    """Un `Bash` a secas haría inútiles las denegadas."""
    assert "Bash" not in dev.HERRAMIENTAS_PERMITIDAS
    assert all(h.startswith("Bash(") or "(" not in h for h in dev.HERRAMIENTAS_PERMITIDAS)


def test_hay_tope_de_vueltas() -> None:
    assert 0 < dev.MAX_VUELTAS <= 100


def test_el_encargo_pasa_por_el_motor(dev_falso) -> None:
    resultado = asyncio.run(dev._dev({"peticion": {"texto": "arregla el bug"}}))
    assert "arregla el bug" in resultado["texto"]
    assert resultado["titular"]
    assert resultado["sesion"] == "falsa"
    assert dev._motor.encargos == ["arregla el bug"]


def test_el_titular_concuerda_en_singular(dev_falso) -> None:
    resultado = asyncio.run(dev._dev({"peticion": {"texto": "algo"}}))
    assert "1 vuelta)" in resultado["titular"]


def test_un_encargo_sin_texto_se_rechaza(dev_falso) -> None:
    with pytest.raises(ValueError):
        asyncio.run(dev._dev({"peticion": {"texto": "   "}}))


def test_un_encargo_fuera_de_la_raiz_se_rechaza(dev_falso) -> None:
    with pytest.raises(ValueError):
        asyncio.run(dev._dev({"peticion": {"texto": "algo", "directorio": "../.."}}))


def test_sin_motor_el_encargo_falla_con_un_error_util(dev_falso, monkeypatch) -> None:
    monkeypatch.setattr(dev, "_motor", None)
    with pytest.raises(RuntimeError, match="PERSEO_DEV_MOTOR"):
        asyncio.run(dev._dev({"peticion": {"texto": "algo"}}))


def test_un_resultado_fallido_del_motor_falla_el_trabajo(dev_falso, monkeypatch) -> None:
    class MotorQueFalla:
        async def ejecutar(self, instruccion, raiz, tope, sesion=""):
            return dev.Resultado(texto="no pude", ok=False)

    monkeypatch.setattr(dev, "_motor", MotorQueFalla())
    with pytest.raises(RuntimeError, match="no pude"):
        asyncio.run(dev._dev({"peticion": {"texto": "algo"}}))
