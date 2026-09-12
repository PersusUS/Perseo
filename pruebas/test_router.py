"""El trabajador: la política antes de ejecutar, y el registro de router."""

from __future__ import annotations

import asyncio

import pytest

from perseo_core.infra import almacen, politica, router
from perseo_core.infra.bus import Bus


def test_los_agentes_del_sistema_estan_registrados() -> None:
    # Importar el arranque es lo que los da de alta.
    from perseo_core import __main__  # noqa: F401

    assert {"correo", "agenda", "memoria", "pc", "dev", "web"} <= set(router.REGISTRO)


def test_registrar_dos_veces_el_mismo_nombre_es_un_error() -> None:
    with pytest.raises(ValueError):

        @router.registrar("eco")
        async def _otro(trabajo):  # pragma: no cover - no llega a ejecutarse
            ...


def test_aprobado_solo_cuando_hay_un_si() -> None:
    assert router.aprobado({"confirmacion": {"decision": "aprobado"}})
    assert not router.aprobado({"confirmacion": {"decision": "rechazado"}})
    assert not router.aprobado({"confirmacion": None})
    assert not router.aprobado({})


def _ejecutar_uno(trabajo: dict) -> Bus:
    bus = Bus()
    trabajador = router.Trabajador(bus)
    asyncio.run(trabajador._ejecutar_uno(trabajo))
    return bus


def test_un_trabajo_libre_se_ejecuta(db) -> None:
    trabajo = almacen.encolar("eco", {"texto": "hola"})
    reclamado = almacen.reclamar()
    _ejecutar_uno(reclamado)

    assert almacen.obtener(trabajo["id"])["estado"] == almacen.HECHO


def test_un_trabajo_irreversible_se_para_antes_de_ejecutarse(db) -> None:
    """Lo que espera no puede haber hecho ya la mitad."""
    trabajo = almacen.encolar("pc", {"accion": "escribir_teclado", "parametro": "hola"})
    reclamado = almacen.reclamar()
    _ejecutar_uno(reclamado)

    parado = almacen.obtener(trabajo["id"])
    assert parado["estado"] == almacen.ESPERANDO
    assert "irreversible" in parado["confirmacion"]["resumen"]


def test_lo_que_no_esta_clasificado_tambien_se_para(db) -> None:
    trabajo = almacen.encolar("memoria", {"accion": "borrar"})
    reclamado = almacen.reclamar()
    _ejecutar_uno(reclamado)

    assert almacen.obtener(trabajo["id"])["estado"] == almacen.ESPERANDO


def test_con_confianza_lo_irreversible_pasa(db, monkeypatch) -> None:
    politica.activar_confianza(30)

    ejecutado = []

    async def falso(trabajo):
        ejecutado.append(trabajo["id"])
        return {"texto": "hecho"}

    monkeypatch.setitem(router.REGISTRO, "pc", falso)
    almacen.encolar("pc", {"accion": "escribir_teclado", "parametro": "hola"})
    _ejecutar_uno(almacen.reclamar())

    assert len(ejecutado) == 1


def test_tras_aprobar_el_trabajo_pasa_la_politica(db, monkeypatch) -> None:
    """El agente se ejecuta desde el principio, y esta vez ve el sí."""
    ejecutado = []

    async def falso(trabajo):
        ejecutado.append(trabajo["id"])
        return {"texto": "hecho"}

    monkeypatch.setitem(router.REGISTRO, "pc", falso)
    trabajo = almacen.encolar("pc", {"accion": "escribir_teclado", "parametro": "x"})
    _ejecutar_uno(almacen.reclamar())
    almacen.resolver_confirmacion(trabajo["id"], True)
    _ejecutar_uno(almacen.reclamar())

    assert len(ejecutado) == 1
    assert almacen.obtener(trabajo["id"])["estado"] == almacen.HECHO


def test_un_agente_desconocido_falla_el_trabajo(db) -> None:
    trabajo = almacen.encolar("eco", {})
    reclamado = almacen.reclamar()
    reclamado["agente"] = "inventado"
    _ejecutar_uno(reclamado)

    fallido = almacen.obtener(trabajo["id"])
    assert fallido["estado"] == almacen.FALLIDO
    assert "desconocido" in fallido["error"]


def test_un_agente_que_lanza_falla_el_trabajo(db, monkeypatch) -> None:
    async def falso(trabajo):
        raise RuntimeError("se rompió")

    monkeypatch.setitem(router.REGISTRO, "eco", falso)
    trabajo = almacen.encolar("eco", {})
    _ejecutar_uno(almacen.reclamar())

    fallido = almacen.obtener(trabajo["id"])
    assert fallido["estado"] == almacen.FALLIDO
    assert "se rompió" in fallido["error"]


def test_necesita_confirmacion_deja_el_trabajo_esperando(db) -> None:
    trabajo = almacen.encolar("simulacro", {"accion": "borrar la papelera", "detalle": "37 ficheros"})
    _ejecutar_uno(almacen.reclamar())

    esperando = almacen.obtener(trabajo["id"])
    assert esperando["estado"] == almacen.ESPERANDO
    assert "borrar la papelera" in esperando["confirmacion"]["resumen"]


def test_la_ruta_del_router_encola_ante_la_duda() -> None:
    """`no_seguro` se trata como encolar: que quede registrado y visible."""
    assert router.Ruta(destino="no_seguro", agente="eco", motivo="").hay_que_encolar
    assert router.Ruta(destino="encolar", agente="eco", motivo="").hay_que_encolar
    assert not router.Ruta(destino="responder", agente="eco", motivo="").hay_que_encolar


def test_el_esquema_del_router_deja_dudar() -> None:
    destinos = router.ESQUEMA_RUTA["properties"]["destino"]["enum"]
    assert "no_seguro" in destinos
