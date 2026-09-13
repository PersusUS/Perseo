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


@pytest.fixture
def armado(monkeypatch):
    """Enciende las confirmaciones mientras dure la prueba.

    Están apagadas de fábrica desde el 2026-09-12 —ver `politica.CONFIRMACIONES`—,
    pero lo que estas pruebas comprueban es el **cableado**: que el trabajador
    consulta la política antes de ejecutar y respeta lo que diga. Ese cableado
    tiene que seguir probado aunque hoy no se use, porque es lo que hará falta
    el día que alguien vuelva a armarlo, y un camino sin pruebas se pudre
    callado.
    """
    monkeypatch.setattr(politica, "CONFIRMACIONES", True)


def test_un_trabajo_irreversible_se_para_antes_de_ejecutarse(db, armado) -> None:
    """Lo que espera no puede haber hecho ya la mitad."""
    trabajo = almacen.encolar("pc", {"accion": "escribir_teclado", "parametro": "hola"})
    reclamado = almacen.reclamar()
    _ejecutar_uno(reclamado)

    parado = almacen.obtener(trabajo["id"])
    assert parado["estado"] == almacen.ESPERANDO
    assert "irreversible" in parado["confirmacion"]["resumen"]


def test_lo_que_no_esta_clasificado_tambien_se_para(db, armado) -> None:
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


def test_tras_aprobar_el_trabajo_pasa_la_politica(db, armado, monkeypatch) -> None:
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


def test_apagadas_no_se_para_nada(db, monkeypatch) -> None:
    """Con el interruptor en su sitio de fábrica, lo irreversible sale entero.

    Es la decisión del 2026-09-12 dicha en una prueba. Si algún día alguien
    vuelve a armar el guardia sin querer, esta es la que se pone roja.
    """
    assert politica.CONFIRMACIONES is False

    ejecutado = []

    async def falso(trabajo):
        ejecutado.append(trabajo["id"])
        return {"texto": "hecho"}

    monkeypatch.setitem(router.REGISTRO, "pc", falso)
    trabajo = almacen.encolar("pc", {"accion": "escribir_teclado", "parametro": "hola"})
    _ejecutar_uno(almacen.reclamar())

    assert ejecutado == [trabajo["id"]]
    assert almacen.obtener(trabajo["id"])["estado"] == almacen.HECHO


def test_apagadas_tampoco_para_a_una_visita(db, monkeypatch) -> None:
    """La otra mitad: ni siquiera quien no es el dueño se para.

    Va escrito porque es lo que más sorprende de la decisión, y porque la regla
    de la visita sigue entera en `politica.pide_confirmacion`: lo que la desactiva
    es el interruptor, no un olvido.
    """
    ejecutado = []

    async def falso(trabajo):
        ejecutado.append(trabajo["id"])
        return {"texto": "hecho"}

    monkeypatch.setitem(router.REGISTRO, "pc", falso)
    trabajo = almacen.encolar(
        "pc", {"accion": "escribir_teclado", "parametro": "hola"}, quien="Una visita"
    )
    _ejecutar_uno(almacen.reclamar())

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
