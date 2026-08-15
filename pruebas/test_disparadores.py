"""Los disparadores: la marca de agua, el que falla y el que se retira."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from perseo_core import almacen, disparadores
from perseo_core.bus import Bus


def test_la_marca_de_agua_empieza_estrenando(tmp_path: Path) -> None:
    vistos = disparadores.Vistos(ruta=tmp_path / "vistos.json").cargar()
    assert vistos.estrenando
    assert vistos.ids == set()


def test_anotar_deja_de_estrenar_y_guarda(tmp_path: Path) -> None:
    ruta = tmp_path / "vistos.json"
    vistos = disparadores.Vistos(ruta=ruta).cargar()
    vistos.anotar(["a", "b"])

    assert not vistos.estrenando
    assert set(json.loads(ruta.read_text(encoding="utf-8"))) == {"a", "b"}


def test_la_marca_de_agua_sobrevive_al_reinicio(tmp_path: Path) -> None:
    """Si viviera en memoria, reiniciar volvería a avisar de todo."""
    ruta = tmp_path / "vistos.json"
    disparadores.Vistos(ruta=ruta).cargar().anotar(["a"])

    otra_vez = disparadores.Vistos(ruta=ruta).cargar()
    assert not otra_vez.estrenando
    assert otra_vez.sin_ver(["a", "b"]) == ["b"]


def test_un_fichero_corrupto_no_tumba_nada(tmp_path: Path) -> None:
    ruta = tmp_path / "vistos.json"
    ruta.write_text("{no es json", encoding="utf-8")
    vistos = disparadores.Vistos(ruta=ruta).cargar()
    assert vistos.ids == set()
    # Existe, así que no se estrena: perder la marca avisa dos veces, y ya está.
    assert not vistos.estrenando


def test_sin_ver_ignora_los_vacios(tmp_path: Path) -> None:
    vistos = disparadores.Vistos(ruta=tmp_path / "v.json").cargar()
    assert vistos.sin_ver(["", "a"]) == ["a"]


def test_la_marca_de_agua_no_crece_sin_fin(tmp_path: Path) -> None:
    vistos = disparadores.Vistos(ruta=tmp_path / "v.json", tope=10).cargar()
    vistos.anotar([f"id-{n:03d}" for n in range(50)])
    assert len(vistos.ids) == 10


def test_encolar_desde_el_contexto_marca_el_origen(db, monkeypatch) -> None:
    """Es lo que distingue en la web lo que has pedido tú de lo que salió solo."""
    bus = Bus()
    ctx = disparadores.Contexto(cfg=db, bus=bus)
    trabajo = asyncio.run(ctx.encolar("eco", {"texto": "x"}))

    assert trabajo["origen"] == "disparador"
    assert almacen.obtener(trabajo["id"])["estado"] == almacen.PENDIENTE


def _planificador(cfg, nombres: tuple[str, ...]) -> disparadores.Planificador:
    from dataclasses import replace

    return disparadores.Planificador(replace(cfg, disparadores=nombres), Bus())


def test_un_disparador_desconocido_no_arranca_nada(cfg) -> None:
    plan = _planificador(cfg, ("no_existe",))
    asyncio.run(plan.ejecutar())  # termina solo, sin lanzar


def test_el_que_se_retira_deja_de_dar_vueltas(cfg, monkeypatch) -> None:
    """Un disparador sin configurar no es un error: es algo que hoy no está."""
    vueltas = []

    async def revisar(ctx):
        vueltas.append(1)
        raise disparadores.Retirarse("no hay de dónde tirar")

    monkeypatch.setitem(
        disparadores.REGISTRO,
        "prueba",
        disparadores.Disparador(nombre="prueba", intervalo=0.01, revisar=revisar),
    )
    asyncio.run(_planificador(cfg, ("prueba",)).ejecutar())
    assert vueltas == [1]


def test_una_vuelta_que_falla_no_tumba_el_bucle(cfg, monkeypatch) -> None:
    vueltas = []

    async def revisar(ctx):
        vueltas.append(1)
        if len(vueltas) < 3:
            raise RuntimeError("la red se cayó")
        raise disparadores.Retirarse("ya está")

    monkeypatch.setitem(
        disparadores.REGISTRO,
        "prueba",
        disparadores.Disparador(nombre="prueba", intervalo=0.01, revisar=revisar),
    )
    asyncio.run(_planificador(cfg, ("prueba",)).ejecutar())
    assert len(vueltas) == 3


def test_la_configuracion_manda_sobre_el_intervalo_del_registro(cfg, monkeypatch) -> None:
    from dataclasses import replace

    async def revisar(ctx):
        raise disparadores.Retirarse("x")

    disparador = disparadores.Disparador(nombre="prueba", intervalo=999, revisar=revisar)
    monkeypatch.setitem(disparadores.REGISTRO, "prueba", disparador)

    plan = disparadores.Planificador(
        replace(cfg, disparadores=("prueba",), intervalos={"prueba": 7.0}), Bus()
    )
    assert plan._intervalo(disparador) == 7.0


def test_registrar_dos_veces_el_mismo_nombre_es_un_error() -> None:
    with pytest.raises(ValueError):

        @disparadores.registrar("correo", intervalo=1)
        async def _otro(ctx):  # pragma: no cover - no llega a ejecutarse
            ...
