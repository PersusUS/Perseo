"""El bus: publicar nunca bloquea, aunque haya un suscriptor dormido."""

from __future__ import annotations

import asyncio

from perseo_core.bus import Bus, Evento


def test_publicar_sin_suscriptores_no_rompe() -> None:
    bus = Bus()
    evento = bus.publicar("trabajo.encolado", trabajo={"id": 1})
    assert evento.tipo == "trabajo.encolado"
    assert bus.suscriptores == 0


def test_el_suscriptor_recibe_lo_que_se_publica() -> None:
    async def guion() -> Evento:
        bus = Bus()
        async with bus.suscribir() as eventos:
            bus.publicar("trabajo.hecho", trabajo={"id": 7})
            return await asyncio.wait_for(eventos.__anext__(), timeout=2)

    evento = asyncio.run(guion())
    assert evento.tipo == "trabajo.hecho"
    assert evento.datos["trabajo"]["id"] == 7


def test_darse_de_baja_al_salir_del_contexto() -> None:
    async def guion() -> int:
        bus = Bus()
        async with bus.suscribir():
            assert bus.suscriptores == 1
        return bus.suscriptores

    assert asyncio.run(guion()) == 0


def test_una_cola_llena_pierde_lo_viejo_y_no_bloquea() -> None:
    """Un móvil con la pantalla apagada no puede hacer crecer la memoria."""

    async def guion() -> Evento:
        bus = Bus(tope_cola=3)
        async with bus.suscribir() as eventos:
            for n in range(10):
                bus.publicar("progreso", n=n)
            # Lo que queda son los últimos, no los primeros.
            return await asyncio.wait_for(eventos.__anext__(), timeout=2)

    primero_que_queda = asyncio.run(guion())
    assert primero_que_queda.datos["n"] >= 7


def test_dos_suscriptores_reciben_los_dos() -> None:
    async def guion() -> tuple[str, str]:
        bus = Bus()
        async with bus.suscribir() as unos, bus.suscribir() as otros:
            bus.publicar("aviso")
            uno = await asyncio.wait_for(unos.__anext__(), timeout=2)
            otro = await asyncio.wait_for(otros.__anext__(), timeout=2)
            return uno.tipo, otro.tipo

    assert asyncio.run(guion()) == ("aviso", "aviso")


def test_el_evento_lleva_marca_de_tiempo_utc() -> None:
    evento = Evento(tipo="x")
    assert evento.momento.endswith("Z")
    assert evento.a_dict()["tipo"] == "x"
