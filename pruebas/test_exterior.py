"""Lo que sale de casa se para, aunque las confirmaciones estén apagadas (ADR 0007).

Es el único nivel que no obedece al interruptor de `politica.CONFIRMACIONES`, y
el único —con lo crítico— cuyo sí no puede darlo el modelo: la pregunta de un
recado lleva dentro el nombre de un botón que escribió quien hizo la web.
"""

from __future__ import annotations

import asyncio

from perseo_core.agentes import chat_herramientas
from perseo_core.infra import almacen, politica, router
from perseo_core.infra.bus import Bus


def test_exterior_se_para_con_las_confirmaciones_apagadas() -> None:
    assert politica.CONFIRMACIONES is False
    assert politica.hay_que_parar("recado", {"accion": "exterior"})
    # Y el resto sigue como decidió el ADR 0005: sin preguntar.
    assert not politica.hay_que_parar("pc", {"accion": "escribir_teclado"})
    assert not politica.hay_que_parar("recado", {"accion": "hacer"})


def test_exterior_no_lo_tapa_la_confianza_ni_un_si_reciente() -> None:
    politica.activar_confianza(30)
    assert politica.pide_confirmacion("recado", {"accion": "exterior"})
    politica.recordar_aprobacion("recado", {"accion": "exterior"})
    assert politica.pide_confirmacion("recado", {"accion": "exterior"})


def test_el_resumen_lo_dice() -> None:
    assert "sale de casa" in politica.resumir("recado", {"accion": "exterior"})


def _ejecutar_uno(trabajo: dict) -> None:
    asyncio.run(router.Trabajador(Bus())._ejecutar_uno(trabajo))


def test_la_pregunta_guarda_su_nivel(db, monkeypatch) -> None:
    """Un recado entero es reversible; lo que para a mitad es exterior, y eso queda escrito."""

    async def falso(trabajo):
        raise router.NecesitaConfirmacion("¿Pulsar «Pagar 45 €»?", nivel=politica.EXTERIOR)

    monkeypatch.setitem(router.REGISTRO, "recado", falso)
    trabajo = almacen.encolar("recado", {"texto": "reserva", "accion": "hacer"})
    _ejecutar_uno(almacen.reclamar())

    parado = almacen.obtener(trabajo["id"])
    assert parado["estado"] == almacen.ESPERANDO
    assert parado["confirmacion"]["nivel"] == politica.EXTERIOR
    assert politica.nivel(parado["agente"], parado["peticion"]) == politica.REVERSIBLE
    assert politica.nivel_de_la_pregunta(parado) == politica.EXTERIOR


def test_sin_nivel_guardado_manda_la_tabla(db) -> None:
    trabajo = almacen.encolar("pc", {"accion": "escribir_teclado"})
    assert politica.nivel_de_la_pregunta(trabajo) == politica.IRREVERSIBLE


def test_el_modelo_no_aprueba_lo_exterior(db, monkeypatch) -> None:
    async def falso(trabajo):
        raise router.NecesitaConfirmacion("¿Pulsar «Pagar»?", nivel=politica.EXTERIOR)

    monkeypatch.setitem(router.REGISTRO, "recado", falso)
    trabajo = almacen.encolar("recado", {"texto": "reserva"})
    _ejecutar_uno(almacen.reclamar())

    respuesta = asyncio.run(
        chat_herramientas._resolver_confirmacion({"id": trabajo["id"], "decision": "aprobar"})
    )
    assert "no se puede aprobar hablando" in respuesta
    assert almacen.obtener(trabajo["id"])["estado"] == almacen.ESPERANDO


def test_rechazarlo_hablando_si_vale(db, monkeypatch) -> None:
    """Decir que no nunca saca nada de casa: ese camino se queda abierto."""

    async def falso(trabajo):
        raise router.NecesitaConfirmacion("¿Pulsar «Pagar»?", nivel=politica.EXTERIOR)

    monkeypatch.setitem(router.REGISTRO, "recado", falso)
    trabajo = almacen.encolar("recado", {"texto": "reserva"})
    _ejecutar_uno(almacen.reclamar())

    asyncio.run(chat_herramientas._resolver_confirmacion({"id": trabajo["id"], "decision": "rechazar"}))
    assert almacen.obtener(trabajo["id"])["estado"] == almacen.RECHAZADO
