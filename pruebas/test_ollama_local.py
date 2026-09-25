"""`perseo on` y el Ollama que contesta en el 11434.

Que el puerto conteste no dice que sirva. En el PC del señor Persus hay dos
Ollama: el de Windows, con `qwen3:4b`, y uno dentro de WSL Ubuntu que solo tiene
`qwen2.5-coder:1.5b`. Con el de WSL encendido, `wslrelay.exe` se queda el
puerto, `perseo on` decía «[ya estaba]», no arrancaba el de Windows, y el triaje
del correo fallaba pidiendo un modelo que ese Ollama no tiene.

Sin red y sin subprocesos: lo que contesta el 11434 y quién tiene el puerto son
dobles.
"""

from __future__ import annotations

import asyncio
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from perseo_core.caras import estado
from perseo_core.infra.configuracion import cargar_configuracion

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "commands"))

import ollama_local  # noqa: E402

#: Lo que tiene el Ollama de WSL de esta máquina, tal cual lo lista.
MODELOS_DE_WSL = ["qwen2.5-coder:1.5b"]


@pytest.fixture()
def perseo(datos: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """El lanzador, sin poder arrancar nada ni preguntar a Windows de verdad.

    `datos` deja el directorio aislado y sin `PERSEO_MODELO_ROUTER` en el
    entorno: si quien corre las pruebas lo tiene puesto, dejarían de probar lo
    que creen que prueban.

    Preguntar por el dueño del puerto cuesta un segundo de PowerShell, y solo
    se paga cuando hay algo que avisar: la prueba que lo quiera, lo pone.
    """
    if sys.platform != "win32":
        pytest.skip("perseo.py importa el registro de Windows")
    import perseo as modulo

    def no_arrancar(argumentos: list[str]) -> None:
        raise AssertionError(f"no tenía que arrancarse nada: {argumentos}")

    def no_preguntar(puerto: int = ollama_local.PUERTO) -> str | None:
        raise AssertionError("con el modelo delante no hay que preguntar por el puerto")

    monkeypatch.setattr(modulo, "_sin_consola", no_arrancar)
    monkeypatch.setattr(ollama_local, "quien_tiene_el_puerto", no_preguntar)
    return modulo


def _contesta(monkeypatch: pytest.MonkeyPatch, modelos: list[str] | None) -> None:
    monkeypatch.setattr(ollama_local, "modelos", lambda: modelos)


def _puerto_de(monkeypatch: pytest.MonkeyPatch, dueno: str | None) -> None:
    monkeypatch.setattr(ollama_local, "quien_tiene_el_puerto", lambda puerto=0: dueno)


# --------------------------------------------------------------------------- #
# Lo que dice `perseo on`
# --------------------------------------------------------------------------- #


def test_el_ollama_de_wsl_no_pasa_por_el_de_windows(
    perseo: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """El caso que se vio: contesta, pero sin `qwen3:4b`, y el puerto es de WSL."""
    _contesta(monkeypatch, MODELOS_DE_WSL)
    _puerto_de(monkeypatch, "wslrelay")

    assert perseo.arrancar_ollama() is False
    salida = capsys.readouterr().out
    assert "[ya estaba]" not in salida
    assert "qwen3:4b" in salida
    assert "wslrelay" in salida
    assert "WSL" in salida


def test_con_el_modelo_ya_estaba(
    perseo: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _contesta(monkeypatch, ["qwen3:4b", *MODELOS_DE_WSL])

    assert perseo.arrancar_ollama() is False
    salida = capsys.readouterr().out
    assert "[ya estaba]" in salida
    assert "[ojo]" not in salida


def test_el_aviso_nombra_el_modelo_configurado(
    perseo: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """El lanzador pide el modelo a `<datos>`, no al de fábrica."""
    monkeypatch.setenv("PERSEO_MODELO_ROUTER", "llama3:8b")
    _contesta(monkeypatch, ["qwen3:4b"])
    _puerto_de(monkeypatch, "ollama")

    perseo.arrancar_ollama()
    assert "ollama pull llama3:8b" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# El aviso: el arreglo depende de quién tenga el puerto
# --------------------------------------------------------------------------- #


def test_al_ollama_de_windows_sin_el_modelo_se_le_pide_bajarlo(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Aquí no hay nada que parar: falta un `ollama pull`."""
    _puerto_de(monkeypatch, "ollama")

    ollama_local.avisar_sin_modelo("qwen3:4b")
    salida = capsys.readouterr().out
    assert "ollama pull qwen3:4b" in salida
    assert "WSL" not in salida


def test_sin_saber_quien_tiene_el_puerto_se_avisa_igual(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Si PowerShell falla, el aviso del modelo no puede caerse con él."""
    _puerto_de(monkeypatch, None)

    ollama_local.avisar_sin_modelo("qwen3:4b")
    salida = capsys.readouterr().out
    assert "qwen3:4b" in salida
    assert "Get-NetTCPConnection" in salida


# --------------------------------------------------------------------------- #
# Que el lanzador y el núcleo digan lo mismo
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "variable, fichero",
    [
        (None, None),
        (None, {"PERSEO_MODELO_ROUTER": "llama3:8b"}),
        ("qwen3:8b", {"PERSEO_MODELO_ROUTER": "llama3:8b"}),
        (None, {"PERSEO_CORREO": "gmail"}),
    ],
)
def test_el_modelo_sale_de_donde_lo_saca_el_nucleo(
    datos: Path,
    monkeypatch: pytest.MonkeyPatch,
    variable: str | None,
    fichero: dict[str, str] | None,
) -> None:
    """El lanzador no importa el núcleo y repite su lectura; esto la ata."""
    if variable is not None:
        monkeypatch.setenv("PERSEO_MODELO_ROUTER", variable)
    if fichero is not None:
        (datos / "entorno.json").write_text(json.dumps(fichero), encoding="utf-8-sig")

    assert ollama_local.modelo_router(datos) == cargar_configuracion().modelo_router


class _Respuesta:
    def __init__(self, datos: Any) -> None:
        self.status = 200
        self._datos = datos

    async def json(self) -> Any:
        return self._datos

    async def __aenter__(self) -> "_Respuesta":
        return self

    async def __aexit__(self, *_: object) -> bool:
        return False


class _Sesion:
    def __init__(self, respuesta: _Respuesta) -> None:
        self._respuesta = respuesta

    def get(self, *_: object, **__: object) -> _Respuesta:
        return self._respuesta


@pytest.mark.parametrize(
    "nombres, vale",
    [
        (["qwen3:4b"], True),
        (["qwen3:8b"], True),
        (["qwen3"], True),
        (MODELOS_DE_WSL, False),
        (["qwen3-coder:30b"], False),
        ([], False),
    ],
)
def test_el_modelo_se_busca_con_la_regla_de_la_pantalla_de_estado(
    datos: Path, nombres: list[str], vale: bool
) -> None:
    """Si el panel pintara verde lo que `perseo on` avisa, no habría a cuál creer."""
    cfg = cargar_configuracion()
    sesion = _Sesion(_Respuesta({"models": [{"name": n} for n in nombres]}))
    pieza = asyncio.run(estado._ollama(cfg, sesion))  # type: ignore[arg-type]

    assert ollama_local.tiene_el_modelo(nombres, cfg.modelo_router) is vale
    assert (pieza.estado == estado.OK) is vale


# --------------------------------------------------------------------------- #
# Lo que se lee del 11434
# --------------------------------------------------------------------------- #


class _Cuerpo:
    def __init__(self, cuerpo: bytes) -> None:
        self.status = 200
        self._cuerpo = cuerpo

    def read(self) -> bytes:
        return self._cuerpo

    def __enter__(self) -> "_Cuerpo":
        return self

    def __exit__(self, *_: object) -> bool:
        return False


def _urlopen(monkeypatch: pytest.MonkeyPatch, respuesta: Any) -> None:
    def falso(*_: object, **__: object) -> Any:
        if isinstance(respuesta, Exception):
            raise respuesta
        return respuesta

    monkeypatch.setattr(urllib.request, "urlopen", falso)


def test_se_leen_los_nombres_de_la_lista(monkeypatch: pytest.MonkeyPatch) -> None:
    cuerpo = json.dumps({"models": [{"name": "qwen2.5-coder:1.5b", "size": 986062089}]})
    _urlopen(monkeypatch, _Cuerpo(cuerpo.encode("utf-8")))
    assert ollama_local.modelos() == MODELOS_DE_WSL


def test_nadie_en_el_puerto_no_es_una_lista_vacia(monkeypatch: pytest.MonkeyPatch) -> None:
    """`None` es «arráncalo»; `[]` es «hay alguien y no sirve». No se confunden."""
    _urlopen(monkeypatch, urllib.error.URLError(ConnectionRefusedError()))
    assert ollama_local.modelos() is None


def test_un_200_que_no_es_json_cuenta_como_vivo_sin_modelos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Algo contesta, y no es un Ollama que sirva: toca aviso, no arrancar otro."""
    _urlopen(monkeypatch, _Cuerpo(b"<html>otra cosa</html>"))
    assert ollama_local.modelos() == []
