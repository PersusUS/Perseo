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

from perseo_core.infra import politica
from perseo_core.servicios import mcp, mcp_transportes, mcp_argumentos

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
    assert cargado["tope_segundos"] == mcp_transportes.TOPE_POR_DEFECTO


# -- El cliente contra el servidor de mentira -------------------------------- #


async def _servidor_vivo() -> tuple[mcp_transportes.ServidorMcp, dict]:
    mcp.definiciones.clear()
    mcp.definiciones["mentira"] = definicion()
    servidor = mcp_transportes.ServidorMcp("mentira", mcp.definiciones["mentira"])
    await servidor.arrancar()
    return servidor, mcp.definiciones["mentira"]


def test_arrancar_lista_las_herramientas() -> None:
    async def guion() -> None:
        servidor, _ = await _servidor_vivo()
        try:
            assert [h["name"] for h in servidor.herramientas] == ["eco", "tarda", "estricto"]
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
            with pytest.raises(mcp_transportes.ErrorMcp):
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
        servidor = mcp_transportes.ServidorMcp("mentira", mcp.definiciones["mentira"])
        await servidor.arrancar()
        try:
            with pytest.raises(mcp_transportes.ErrorMcp):
                await servidor.llamar("eco", {})
        finally:
            await servidor.detener()

    asyncio.run(guion())


def test_el_nombre_en_espanol_llega_traducido() -> None:
    """`comando` contra un servidor que espera `command` era una llamada perdida.

    Era el fallo #333: el modelo piensa en español y ese idioma se le colaba
    hasta el nombre del parámetro.
    """

    async def guion() -> None:
        servidor, _ = await _servidor_vivo()
        try:
            respuesta = await servidor.llamar(
                "estricto", {"comando": "echo hola", "ruta": "C:/"}
            )
            assert '"command": "echo hola"' in respuesta
            assert '"path": "C:/"' in respuesta
            assert "comando" not in respuesta
        finally:
            await servidor.detener()

    asyncio.run(guion())


def test_lo_que_falta_se_dice_con_la_firma_y_sin_viajar() -> None:
    """Un requerido que no viene se contesta aquí, con los nombres buenos."""

    async def guion() -> None:
        servidor, _ = await _servidor_vivo()
        try:
            with pytest.raises(mcp_transportes.ErrorMcp) as fallo:
                await servidor.llamar("estricto", {"command": "echo"})
        finally:
            await servidor.detener()
        mensaje = str(fallo.value)
        assert "path" in mensaje
        assert "requerido" in mensaje

    asyncio.run(guion())


def test_el_rechazo_del_servidor_trae_la_firma_pegada() -> None:
    """El -32602 llega como error de JSON-RPC; la pista tiene que ir con él."""

    async def guion() -> None:
        servidor, _ = await _servidor_vivo()
        # Se salta la comprobación de casa para llegar al rechazo del servidor.
        servidor.herramientas = [{"name": "estricto", "description": "", "inputSchema": {}}]
        try:
            with pytest.raises(mcp_transportes.ErrorMcp) as fallo:
                await servidor.llamar("estricto", {})
        finally:
            await servidor.detener()
        assert "Input validation error" in str(fallo.value)

    asyncio.run(guion())


def test_el_fichero_pone_lo_que_el_modelo_no_sabe() -> None:
    """La raíz del vault no la sabe ningún modelo: la escribe una persona."""

    async def guion() -> None:
        mcp.definiciones.clear()
        mcp.definiciones["mentira"] = definicion(
            argumentos_por_defecto={"estricto": {"path": "C:/Users/ejemplo/Documents/vault"}}
        )
        servidor = mcp_transportes.ServidorMcp("mentira", mcp.definiciones["mentira"])
        await servidor.arrancar()
        try:
            respuesta = await servidor.llamar("estricto", {"command": "buscar"})
            assert "C:/Users/ejemplo/Documents/vault" in respuesta
        finally:
            await servidor.detener()

    asyncio.run(guion())


def test_el_catalogo_ensena_los_parametros() -> None:
    """Sin la firma, el modelo adivina los nombres — y adivina en español."""
    firma = mcp_argumentos._firma(
        {
            "name": "estricto",
            "description": "Prueba.",
            "inputSchema": {
                "type": "object",
                "properties": {"command": {"type": "string"}, "path": {"type": "string"}},
                "required": ["command"],
            },
        }
    )
    assert firma.startswith("estricto(command, [path])")


def test_un_servidor_que_no_contesta_muere_a_plazo(tmp_path: Path) -> None:
    """El plazo mata al proceso; la siguiente llamada lo resucita."""

    async def guion() -> None:
        mcp.definiciones.clear()
        mcp.definiciones["mentira"] = definicion(tope_segundos=5)
        servidor = mcp_transportes.ServidorMcp("mentira", mcp.definiciones["mentira"])
        await servidor.arrancar()
        try:
            with pytest.raises(mcp_transportes.ErrorMcp):
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


def test_una_herramienta_puede_tener_su_propio_nivel(cfg) -> None:
    """Un servidor entero no es un nivel: mirar no cuesta lo que escribir."""
    escribir_config(
        cfg.directorio_datos,
        {
            "windows": definicion(
                nivel="irreversible",
                niveles_herramienta={"Snapshot": "libre", "Registry": "que sé yo"},
            )
        },
    )
    asyncio.run(mcp.iniciar(cfg))
    try:
        peticion = {"servidor": "windows", "herramienta": "Snapshot"}
        assert politica.nivel("mcp", peticion) == politica.LIBRE
        # Un nivel raro se descarta y esa herramienta vuelve a la del servidor.
        raro = {"servidor": "windows", "herramienta": "Registry"}
        assert politica.nivel("mcp", raro) == politica.IRREVERSIBLE
    finally:
        asyncio.run(mcp.detener())


def test_el_powershell_que_solo_mira_no_para_a_pedir_un_si(cfg) -> None:
    """Listar el escritorio no es irreversible; borrarlo sí."""
    escribir_config(cfg.directorio_datos, {"windows": definicion(nivel="irreversible")})
    asyncio.run(mcp.iniciar(cfg))

    def nivel_de(comando: str) -> str:
        return politica.nivel(
            "mcp",
            {
                "servidor": "windows",
                "herramienta": "PowerShell",
                "argumentos": {"command": comando},
            },
        )

    try:
        assert nivel_de("Get-ChildItem -Directory C:/Users/ejemplo/Desktop") == politica.REVERSIBLE
        assert nivel_de("dir | Select-Object Name") == politica.REVERSIBLE
        # Lo que escribe, borra o se esconde detrás de una tubería, pregunta.
        assert nivel_de("Get-ChildItem | Remove-Item") == politica.IRREVERSIBLE
        assert nivel_de("Remove-Item C:/algo -Recurse") == politica.IRREVERSIBLE
        assert nivel_de("Get-Content notas.txt > salida.txt") == politica.IRREVERSIBLE
        assert nivel_de("Get-ChildItem; Remove-Item x") == politica.IRREVERSIBLE
        assert nivel_de("") == politica.IRREVERSIBLE
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
    assert {h["nombre"] for h in resultado["servidores"][0]["herramientas"]} == {
        "eco",
        "tarda",
        "estricto",
    }
    # Y con la firma: los parámetros viajan en el catálogo y en el texto.
    estricto = next(
        h for h in resultado["servidores"][0]["herramientas"] if h["nombre"] == "estricto"
    )
    assert [p["nombre"] for p in estricto["parametros"]] == ["command", "path"]
    assert "estricto(command, path)" in resultado["texto"]


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


def test_los_argumentos_en_texto_tambien_valen(cfg) -> None:
    """Un modelo de voz manda a veces el objeto ya escrito como JSON."""
    _agente_con_servidor(cfg)
    resultado = asyncio.run(
        mcp._mcp(
            {
                "peticion": {
                    "accion": "llamar",
                    "servidor": "mentira",
                    "herramienta": "eco",
                    "argumentos": '{"dato": "42"}',
                }
            }
        )
    )
    assert "42" in resultado["texto"]
    assert resultado["titular"].startswith("mentira.eco()")


def test_llamar_sin_servidor_es_error_claro(cfg) -> None:
    _agente_con_servidor(cfg)
    with pytest.raises(mcp_transportes.ErrorMcp):
        asyncio.run(
            mcp._mcp({"peticion": {"accion": "llamar", "servidor": "fantasma", "herramienta": "x"}})
        )


def test_sin_nada_configurado_se_contesta_con_gracia(cfg) -> None:
    asyncio.run(mcp.iniciar(cfg))
    resultado = asyncio.run(mcp._mcp({"peticion": {"accion": "servidores"}}))
    assert resultado["servidores"] == []


# --------------------------------------------------------------------------- #
# Servidores MCP remotos (2026-08-24)
# --------------------------------------------------------------------------- #


def _escribir_mcp(datos: Path, contenido: dict) -> dict:
    (datos / "mcp.json").write_text(json.dumps(contenido), encoding="utf-8")
    return mcp.cargar_servidores(datos)


def test_un_servidor_con_url_vale(datos: Path) -> None:
    """Lo de fuera también cuenta: GitHub, Notion y compañía viven en una URL."""
    servidores = _escribir_mcp(datos, {"lejos": {"url": "https://mcp.ejemplo.com/mcp"}})
    assert servidores["lejos"]["url"] == "https://mcp.ejemplo.com/mcp"
    assert servidores["lejos"]["comando"] == []


def test_un_remoto_abre_el_cliente_de_http(datos: Path) -> None:
    servidores = _escribir_mcp(datos, {"lejos": {"url": "https://mcp.ejemplo.com/mcp"}})
    servidor = mcp_transportes.abrir_servidor("lejos", servidores["lejos"])
    assert isinstance(servidor, mcp_transportes.ServidorMcpRemoto)


def test_lo_de_siempre_sigue_siendo_un_proceso_hijo(datos: Path) -> None:
    servidores = _escribir_mcp(datos, {"cerca": {"comando": ["python", "servidor.py"]}})
    servidor = mcp_transportes.abrir_servidor("cerca", servidores["cerca"])
    assert isinstance(servidor, mcp_transportes.ServidorMcp)


@pytest.mark.parametrize(
    "url, vale",
    [
        ("https://mcp.ejemplo.com/mcp", True),
        ("http://127.0.0.1:8912/mcp", True),
        ("http://localhost:8912/mcp", True),
        ("http://mcp.ejemplo.com/mcp", False),
        ("ftp://mcp.ejemplo.com", False),
        ("no-es-una-url", False),
    ],
)
def test_en_claro_solo_contra_casa(url: str, vale: bool) -> None:
    """Por ahí viajan argumentos que Perseo compone leyendo correos y pantallas."""
    assert mcp_transportes._url_aceptable(url) is vale


def test_un_remoto_en_claro_y_lejos_se_descarta(datos: Path) -> None:
    servidores = _escribir_mcp(datos, {"malo": {"url": "http://mcp.ejemplo.com/mcp"}})
    assert servidores == {}


def test_las_cabeceras_del_testigo_se_conservan(datos: Path) -> None:
    servidores = _escribir_mcp(
        datos,
        {"lejos": {"url": "https://mcp.ejemplo.com/mcp", "cabeceras": {"Authorization": "Bearer x"}}},
    )
    assert servidores["lejos"]["cabeceras"]["Authorization"] == "Bearer x"


def test_un_remoto_sin_haber_saludado_no_esta_vivo(datos: Path) -> None:
    servidores = _escribir_mcp(datos, {"lejos": {"url": "https://mcp.ejemplo.com/mcp"}})
    assert mcp_transportes.abrir_servidor("lejos", servidores["lejos"]).vivo is False
