"""El cliente MCP: protocolo, política por servidor y el agente.

Contra un servidor de mentira que habla el protocolo de verdad (JSON-RPC sobre
stdio), con ruido incluido —notificaciones sueltas y peticiones ajenas— porque
los servidores reales hacen eso y el cliente no puede quedarse colgado.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from perseo_core import mcp, politica

MENTIRA = Path(__file__).resolve().parent / "servidor_mcp_mentira.py"


@pytest.fixture(autouse=True)
def limpiar():
    asyncio.run(mcp.detener())
    yield
    asyncio.run(mcp.detener())


def definicion(**extra) -> dict:
    base = {
        "comando": [sys.executable, str(MENTIRA)],
        "nivel": "reversible",
        "herramientas": [],
        "env": {},
        "tope_segundos": 60.0,
    }
    base.update(extra)
    return base


def escribir_config(datos: Path, contenido: dict) -> Path:
    ruta = datos / "mcp.json"
    ruta.write_text(json.dumps(contenido), encoding="utf-8")
    return ruta


# -- La configuración -------------------------------------------------------- #


def test_sin_fichero_no_hay_servidores(tmp_path: Path) -> None:
    assert mcp.cargar_servidores(tmp_path) == {}


def test_un_comando_en_cadena_se_descarta(tmp_path: Path) -> None:
    """Una cadena nunca vale: por ahí entran las comillas y los &&."""
    ruta = tmp_path / "mcp.json"
    ruta.write_text(json.dumps({"x": {"comando": "npx -y algo"}}), encoding="utf-8")
    assert mcp.cargar_servidores(tmp_path) == {}


def test_el_nivel_raro_cae_en_irreversible(tmp_path: Path) -> None:
    escribir_config(tmp_path, {"x": definicion(nivel="a lo loco")})
    cargados = mcp.cargar_servidores(tmp_path)
    assert cargados["x"]["nivel"] == politica.IRREVERSIBLE


def test_los_valores_por_defecto(tmp_path: Path) -> None:
    escribir_config(tmp_path, {"x": {"comando": ["algo"]}})
    cargado = mcp.cargar_servidores(tmp_path)["x"]
    assert cargado["nivel"] == politica.IRREVERSIBLE
    assert cargado["herramientas"] == []
    assert cargado["env"] == {}
    assert cargado["tope_segundos"] == mcp.TOPE_POR_DEFECTO


# -- El cliente contra el servidor de mentira -------------------------------- #


async def _servidor_vivo() -> tuple[mcp.ServidorMcp, dict]:
    mcp.definiciones.clear()
    mcp.definiciones["mentira"] = definicion()
    servidor = mcp.ServidorMcp("mentira", mcp.definiciones["mentira"])
    await servidor.arrancar()
    return servidor, mcp.definiciones["mentira"]


def test_arrancar_lista_las_herramientas() -> None:
    async def guion() -> None:
        servidor, _ = await _servidor_vivo()
        try:
            assert [h["name"] for h in servidor.herramientas] == ["eco", "tarda"]
        finally:
            await servidor.detener()

    asyncio.run(guion())


def test_llamar_devuelve_texto_y_acepta_objetos() -> None:
    async def guion() -> None:
        servidor, _ = await _servidor_vivo()
        try:
            respuesta = await servidor.llamar("eco", {"saludo": "hola"})
            assert "hola" in respuesta
        finally:
            await servidor.detener()

    asyncio.run(guion())


def test_la_herramienta_desconocida_es_error_propio() -> None:
    async def guion() -> None:
        servidor, _ = await _servidor_vivo()
        try:
            with pytest.raises(mcp.ErrorMcp):
                await servidor.llamar("no_existo", {})
        finally:
            await servidor.detener()

    asyncio.run(guion())


def test_la_lista_de_permitidas_manda() -> None:
    """Si el día de mañana el servidor estrena una herramienta peligrosa, no
    pasa nada hasta que alguien la escriba en `mcp.json`."""

    async def guion() -> None:
        mcp.definiciones.clear()
        mcp.definiciones["mentira"] = definicion(herramientas=["tarda"])
        servidor = mcp.ServidorMcp("mentira", mcp.definiciones["mentira"])
        await servidor.arrancar()
        try:
            with pytest.raises(mcp.ErrorMcp):
                await servidor.llamar("eco", {})
        finally:
            await servidor.detener()

    asyncio.run(guion())


def test_un_servidor_que_no_contesta_muere_a_plazo(tmp_path: Path) -> None:
    """El plazo mata al proceso; la siguiente llamada lo resucita."""

    async def guion() -> None:
        mcp.definiciones.clear()
        mcp.definiciones["mentira"] = definicion(tope_segundos=5)
        servidor = mcp.ServidorMcp("mentira", mcp.definiciones["mentira"])
        await servidor.arrancar()
        try:
            with pytest.raises(mcp.ErrorMcp):
                await servidor.llamar("tarda", {})
            assert not servidor.vivo
        finally:
            await servidor.detener()

    asyncio.run(guion())


# -- El gancho de la política -------------------------------------------------


def test_el_nivel_del_servidor_llega_a_la_politica(cfg) -> None:
    escribir_config(cfg.directorio_datos, {"vault": definicion(), "libre": definicion(nivel="libre")})

    async def guion() -> None:
        await mcp.iniciar(cfg)

    asyncio.run(guion())
    try:
        assert politica.nivel("mcp", {"servidor": "vault"}) == politica.REVERSIBLE
        assert politica.nivel("mcp", {"servidor": "libre"}) == politica.LIBRE
        # Un servidor sin entrada cae en lo desconocido: irreversible.
        assert politica.nivel("mcp", {"servidor": "intruso"}) == politica.IRREVERSIBLE
        # Y otro agente no se entera del gancho.
        assert politica.nivel("pc", {"accion": "abrir_app"}) == politica.LIBRE
    finally:
        asyncio.run(mcp.detener())


# -- El agente ----------------------------------------------------------------


def _agente_con_servidor(cfg) -> None:
    escribir_config(cfg.directorio_datos, {"mentira": definicion()})
    asyncio.run(mcp.iniciar(cfg))


def test_listar_dice_nombres_y_niveles(cfg) -> None:
    _agente_con_servidor(cfg)
    resultado = asyncio.run(mcp._mcp({"peticion": {"accion": "servidores"}}))
    nombres = [s["nombre"] for s in resultado["servidores"]]
    assert nombres == ["mentira"]
    assert {h["nombre"] for h in resultado["servidores"][0]["herramientas"]} == {"eco", "tarda"}


def test_llamar_por_el_agente(cfg) -> None:
    _agente_con_servidor(cfg)
    resultado = asyncio.run(
        mcp._mcp(
            {
                "peticion": {
                    "accion": "llamar",
                    "servidor": "mentira",
                    "herramienta": "eco",
                    "argumentos": {"dato": "42"},
                }
            }
        )
    )
    assert "42" in resultado["texto"]
    assert resultado["titular"].startswith("mentira.eco()")


def test_llamar_sin_servidor_es_error_claro(cfg) -> None:
    _agente_con_servidor(cfg)
    with pytest.raises(mcp.ErrorMcp):
        asyncio.run(
            mcp._mcp({"peticion": {"accion": "llamar", "servidor": "fantasma", "herramienta": "x"}})
        )


def test_sin_nada_configurado_se_contesta_con_gracia(cfg) -> None:
    asyncio.run(mcp.iniciar(cfg))
    resultado = asyncio.run(mcp._mcp({"peticion": {"accion": "servidores"}}))
    assert resultado["servidores"] == []
