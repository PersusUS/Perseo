"""Lo que pasa cuando la telemetría y la presencia **no** pueden contestar.

La pantalla de estado se refresca sola desde el móvil. Un fallo aquí no puede
tumbar la respuesta entera: lo que se pierde es un número, y lo que se protege es
saber qué está encendido. Estas pruebas comprueban justo los caminos malos, que
son los que nadie ejecuta hasta el día que fallan.
"""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from perseo_core import almacen, estado


def test_sin_psutil_la_telemetria_lo_dice_y_no_lanza(monkeypatch: pytest.MonkeyPatch) -> None:
    """`psutil` entró hoy como dependencia. En la Raspberry del plan puede no
    estar, y el núcleo tiene que arrancar igual."""
    de_verdad = builtins.__import__

    def sin_psutil(nombre, *args, **kwargs):
        if nombre == "psutil":
            raise ImportError("no está")
        return de_verdad(nombre, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", sin_psutil)
    datos = estado.telemetria()
    assert datos["disponible"] is False
    assert "psutil" in datos["motivo"]


def test_si_psutil_revienta_tampoco_lanza(monkeypatch: pytest.MonkeyPatch) -> None:
    import psutil

    def explota():
        raise RuntimeError("el contador se ha ido")

    monkeypatch.setattr(psutil, "virtual_memory", explota)
    datos = estado.telemetria()
    assert datos["disponible"] is False


def test_la_telemetria_da_numeros_creibles() -> None:
    datos = estado.telemetria()
    if not datos["disponible"]:
        pytest.skip("sin psutil en esta máquina")
    assert 0 <= datos["cpu"] <= 100
    assert 0 <= datos["memoria"]["porcentaje"] <= 100
    assert datos["memoria"]["usado"] <= datos["memoria"]["total"]
    assert "de" in datos["disco"]["legible"]


def test_la_red_da_velocidad_y_no_el_total_desde_el_arranque() -> None:
    """Un total desde que arrancó Windows es un número enorme que no dice nada."""
    if not estado.telemetria()["disponible"]:
        pytest.skip("sin psutil en esta máquina")
    segunda = estado.telemetria()
    assert segunda["red"]["subida"] >= 0
    assert "/s" in segunda["red"]["legible"]


def test_la_presencia_con_la_base_cerrada_no_revienta(cfg: almacen.Configuracion) -> None:
    """El caso de un reinicio a medias: la pantalla pide estado antes de que la
    base esté abierta."""
    import asyncio

    almacen.cerrar()
    datos = asyncio.run(estado.presencia(cfg))
    assert datos["haciendo"] is None
    assert datos["correo"] == {}


def test_la_presencia_cuenta_solo_lo_que_falta_por_resolver(db, datos: Path) -> None:
    """Un correo marcado deja de contar; lo ignorable no contó nunca."""
    import asyncio

    trabajo = almacen.encolar("correo", {"accion": "triar"}, origen="disparador")
    almacen.reclamar()
    almacen.completar(trabajo["id"], {
        "clasificados": [
            {"id": "m1", "clase": "requiere_accion"},
            {"id": "m2", "clase": "requiere_accion"},
            {"id": "m3", "clase": "ignorar"},
        ]
    })

    sin_marcar = asyncio.run(estado.presencia(cfg=almacen.cargar_configuracion()))
    assert sin_marcar["correo"] == {"requiere_accion": 2}

    almacen.marcar_correo("m1", almacen.ATENDIDO)
    marcado = asyncio.run(estado.presencia(cfg=almacen.cargar_configuracion()))
    assert marcado["correo"] == {"requiere_accion": 1}
