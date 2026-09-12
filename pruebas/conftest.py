"""Piezas compartidas por las pruebas unitarias.

Estas pruebas son **el otro lado** de los verificadores de `perseo_core/`. Los
verificadores levantan el núcleo de verdad y comprueban el sistema entero, que es
lo que da confianza pero tarda minutos; esto comprueba cada pieza por separado en
milisegundos, sin red, sin subprocesos y sin tocar nada real.

Las dos cosas hacen falta: un verificador dice que el conjunto funciona, y una
prueba unitaria dice **qué** se ha roto cuando deja de funcionar.

    python -m pytest
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from perseo_core.infra import almacen, politica  # noqa: E402
from perseo_core.infra.configuracion import Configuracion, cargar_configuracion  # noqa: E402


@pytest.fixture()
def datos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Un directorio de datos aislado, con el entorno apuntando ahí.

    Se limpian también las variables del sistema: si quien ejecuta las pruebas
    tiene `PERSEO_CORREO` puesto en su terminal, las pruebas dejarían de probar
    lo que creen que prueban.
    """
    for variable in (
        "PERSEO_CORE_HOST",
        "PERSEO_CORE_PUERTO",
        "PERSEO_TOKEN",
        "PERSEO_TELEGRAM_TOKEN",
        "PERSEO_TELEGRAM_CHAT",
        "PERSEO_DISPARADORES",
        "PERSEO_CORREO",
        "PERSEO_CORREO_FALSO",
        "PERSEO_AGENDA",
        "PERSEO_AGENDA_FALSA",
        "PERSEO_DEV_MOTOR",
        "PERSEO_DEV_RAIZ",
        "PERSEO_WEB",
        "PERSEO_WEB_LOCAL",
        "OBSIDIAN_VAULT_PATH",
        "PERSEO_VAULT",
        "PERSEO_VAULT_REST",
        "PERSEO_VAULT_CLAVE",
        "PERSEO_MODELO_SUPLENTE",
        "GEMINI_API_KEY",
        "PERSEO_URL_BASE",
        "PERSEO_OLLAMA",
        "PERSEO_MODELO_ROUTER",
    ):
        monkeypatch.delenv(variable, raising=False)

    directorio = tmp_path / "datos"
    directorio.mkdir()
    monkeypatch.setenv("PERSEO_CORE_DATOS", str(directorio))
    return directorio


@pytest.fixture()
def cfg(datos: Path) -> Configuracion:
    return cargar_configuracion()


@pytest.fixture()
def db(cfg: Configuracion) -> Iterator[Configuracion]:
    """Base de datos abierta sobre el directorio temporal, y cerrada al salir.

    `almacen` guarda la conexión en una global, así que dejarla abierta filtraría
    estado de una prueba a la siguiente — que es la clase de fallo que aparece
    solo cuando se ejecutan todas juntas.
    """
    almacen.cerrar()
    almacen.abrir(cfg)
    try:
        yield cfg
    finally:
        almacen.cerrar()


@pytest.fixture()
def vault(tmp_path: Path) -> Path:
    carpeta = tmp_path / "vault"
    carpeta.mkdir()
    return carpeta


@pytest.fixture(autouse=True)
def politica_limpia(tmp_path: Path) -> Iterator[None]:
    """La política guarda el fichero de confianza en una global: se aísla."""
    carpeta = tmp_path / "politica"
    carpeta.mkdir()
    politica.iniciar(carpeta)
    yield
    politica.desactivar_confianza()
