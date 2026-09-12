"""La pantalla de estado: qué se pinta en verde, en ámbar y en rojo.

Lo que se comprueba aquí no es el aspecto, es **la clasificación**: que algo sin
configurar salga como apagado y no como fallo, que un fallo salga con el arreglo
al lado, y que un sondeo que revienta no tumbe la respuesta entera. Un panel que
está siempre en rojo se deja de mirar el segundo día, y entonces no sirve de
nada tenerlo.

Sin red: los sondeos que salen fuera se sustituyen por dobles.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import aiohttp
import pytest

from perseo_core.caras import estado
from perseo_core.infra import politica
from perseo_core.infra.configuracion import Configuracion, cargar_configuracion


@pytest.fixture(autouse=True)
def sin_memoria() -> None:
    """Cada prueba parte sin nada recordado de la anterior."""
    estado.olvidar()


def configuracion(monkeypatch: pytest.MonkeyPatch, datos: Path, **variables: str) -> Configuracion:
    for nombre, valor in variables.items():
        monkeypatch.setenv(nombre, valor)
    return cargar_configuracion()


# --------------------------------------------------------------------------- #
# Dobles de aiohttp
# --------------------------------------------------------------------------- #


class _Respuesta:
    def __init__(self, status: int, datos: Any) -> None:
        self.status = status
        self._datos = datos

    async def json(self) -> Any:
        return self._datos

    async def text(self) -> str:
        return str(self._datos)

    async def __aenter__(self) -> "_Respuesta":
        return self

    async def __aexit__(self, *_: object) -> bool:
        return False


class _Rota:
    """Un `get` que no llega a ninguna parte, como Ollama apagado."""

    async def __aenter__(self) -> "_Rota":
        raise aiohttp.ClientConnectionError("conexión rechazada")

    async def __aexit__(self, *_: object) -> bool:
        return False


class _Sesion:
    def __init__(self, respuesta: Any) -> None:
        self._respuesta = respuesta

    def get(self, url: str, **_: object) -> Any:
        return self._respuesta


# --------------------------------------------------------------------------- #
# Topes de cuota
# --------------------------------------------------------------------------- #


def test_gemma_tiene_su_tope() -> None:
    assert estado.tope_diario("gemma-4-31b-it") == 14400


def test_flash_lite_gana_a_flash() -> None:
    """El orden de la tabla importa: `flash-lite` contiene `flash`.

    Si se comprobara al revés, un modelo con 500 peticiones al día se pintaría
    con el tope de 20 y la barra saldría llena el primer día.
    """
    assert estado.tope_diario("gemini-3.5-flash-lite") == 500
    assert estado.tope_diario("gemini-3.5-flash") == 20


def test_un_modelo_desconocido_no_se_inventa_tope() -> None:
    assert estado.tope_diario("un-modelo-nuevo") is None


def test_la_cuota_incluye_el_suplente_aunque_no_se_haya_usado(
    monkeypatch: pytest.MonkeyPatch, datos: Path
) -> None:
    """Sin la fila, el panel no diría nada el día que aún no se ha gastado nada."""
    cfg = configuracion(monkeypatch, datos, PERSEO_MODELO_SUPLENTE="gemma-4-31b-it")
    cuota = estado._cuota(cfg, {})
    assert [s["modelo"] for s in cuota["servicios"]] == ["gemma-4-31b-it"]
    assert cuota["servicios"][0]["usadas"] == 0


def test_la_cuota_cuenta_lo_gastado(monkeypatch: pytest.MonkeyPatch, datos: Path) -> None:
    cfg = configuracion(monkeypatch, datos, PERSEO_MODELO_SUPLENTE="gemma-4-31b-it")
    cuota = estado._cuota(cfg, {"gemma-4-31b-it": 7})
    assert cuota["servicios"][0] == {"modelo": "gemma-4-31b-it", "usadas": 7, "tope": 14400}


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #


def test_ollama_apagado_sale_en_rojo_con_el_arreglo(cfg: Configuracion) -> None:
    pieza = asyncio.run(estado._ollama(cfg, _Sesion(_Rota())))
    assert pieza.estado == estado.MALO
    assert "ollama serve" in pieza.arreglo


def test_ollama_en_pie_con_su_modelo(cfg: Configuracion) -> None:
    sesion = _Sesion(_Respuesta(200, {"models": [{"name": cfg.modelo_router}]}))
    pieza = asyncio.run(estado._ollama(cfg, sesion))
    assert pieza.estado == estado.OK
    assert cfg.modelo_router in pieza.detalle


def test_ollama_en_pie_sin_el_modelo_avisa_de_como_traerlo(cfg: Configuracion) -> None:
    """Es ámbar y no rojo: el servidor está, lo que falta se baja en un comando."""
    sesion = _Sesion(_Respuesta(200, {"models": [{"name": "llama3:8b"}]}))
    pieza = asyncio.run(estado._ollama(cfg, sesion))
    assert pieza.estado == estado.AVISO
    assert pieza.arreglo == f"ollama pull {cfg.modelo_router}"


def test_a_ollama_le_vale_otra_etiqueta_del_mismo_modelo(
    monkeypatch: pytest.MonkeyPatch, datos: Path
) -> None:
    """`qwen3:4b` y `qwen3:8b` son el mismo modelo con otro tamaño.

    Pedir la etiqueta exacta pintaría un aviso a quien tiene descargada otra
    variante que funciona igual de bien.
    """
    cfg = configuracion(monkeypatch, datos, PERSEO_MODELO_ROUTER="qwen3:4b")
    sesion = _Sesion(_Respuesta(200, {"models": [{"name": "qwen3:8b"}]}))
    assert asyncio.run(estado._ollama(cfg, sesion)).estado == estado.OK


def test_ollama_que_responde_mal_no_es_lo_mismo_que_apagado(cfg: Configuracion) -> None:
    pieza = asyncio.run(estado._ollama(cfg, _Sesion(_Respuesta(500, "boom"))))
    assert pieza.estado == estado.MALO
    assert "500" in pieza.detalle


# --------------------------------------------------------------------------- #
# Lo que se sabe sin preguntar a nadie
# --------------------------------------------------------------------------- #


def test_el_suplente_viene_apagado(cfg: Configuracion) -> None:
    """Apagado, no roto: mandar el texto fuera es una decisión, no un defecto."""
    pieza = estado._suplente(cfg)
    assert pieza.estado == estado.APAGADO
    assert "tercero" in pieza.arreglo


def test_el_suplente_sin_clave_avisa(monkeypatch: pytest.MonkeyPatch, datos: Path) -> None:
    cfg = configuracion(monkeypatch, datos, PERSEO_MODELO_SUPLENTE="gemma-4-31b-it")
    assert estado._suplente(cfg).estado == estado.AVISO


def test_el_suplente_completo(monkeypatch: pytest.MonkeyPatch, datos: Path) -> None:
    cfg = configuracion(
        monkeypatch, datos, PERSEO_MODELO_SUPLENTE="gemma-4-31b-it", GEMINI_API_KEY="x"
    )
    assert estado._suplente(cfg).estado == estado.OK


def test_telegram_sin_configurar(cfg: Configuracion) -> None:
    assert estado._telegram(cfg).estado == estado.APAGADO


def test_telegram_con_enlace_que_no_sale_de_la_maquina(
    monkeypatch: pytest.MonkeyPatch, datos: Path
) -> None:
    """El fallo que no se parece a su causa: el enlace abre en blanco en el móvil."""
    cfg = configuracion(
        monkeypatch,
        datos,
        PERSEO_TELEGRAM_TOKEN="t",
        PERSEO_TELEGRAM_CHAT="1",
        PERSEO_URL_BASE="http://127.0.0.1:8787",
    )
    pieza = estado._telegram(cfg)
    assert pieza.estado == estado.AVISO
    assert "tailscale" in pieza.arreglo.lower()


def test_telegram_bien(monkeypatch: pytest.MonkeyPatch, datos: Path) -> None:
    cfg = configuracion(
        monkeypatch,
        datos,
        PERSEO_TELEGRAM_TOKEN="t",
        PERSEO_TELEGRAM_CHAT="1",
        PERSEO_URL_BASE="http://100.64.0.1:8787",
    )
    assert estado._telegram(cfg).estado == estado.OK


def test_el_buzon_de_mentira_no_pasa_por_verde(
    monkeypatch: pytest.MonkeyPatch, datos: Path
) -> None:
    """Verificar con imitaciones está bien; creerse que es el buzón de verdad, no."""
    cfg = configuracion(monkeypatch, datos, PERSEO_CORREO="falso")
    assert estado._correo(cfg).estado == estado.AVISO


def test_sin_correo_es_apagado(cfg: Configuracion) -> None:
    assert estado._correo(cfg).estado == estado.APAGADO


def test_el_motor_de_dev_simulado_avisa(monkeypatch: pytest.MonkeyPatch, datos: Path) -> None:
    cfg = configuracion(monkeypatch, datos, PERSEO_DEV_MOTOR="falso")
    assert estado._dev(cfg).estado == estado.AVISO


def test_el_navegador_simulado_avisa(monkeypatch: pytest.MonkeyPatch, datos: Path) -> None:
    cfg = configuracion(monkeypatch, datos, PERSEO_WEB="falso")
    assert estado._web(cfg).estado == estado.AVISO


def test_las_confirmaciones_apagadas_se_dicen() -> None:
    """El estado de fabrica desde el 2026-09-12, y lo que mas importa que no mienta.

    Un panel se lee de un vistazo y no se comprueba: si esta pieza saliera en
    verde diciendo que lo irreversible pide un si, seria peor que no tenerla.
    """
    assert not politica.CONFIRMACIONES
    pieza = estado._confianza()
    assert pieza.estado == estado.AVISO
    assert "Apagadas" in pieza.detalle
    assert pieza.arreglo, "una pieza en aviso dice como se arregla"


def test_las_confirmaciones_puestas_son_lo_normal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(politica, "CONFIRMACIONES", True)
    assert estado._confianza().estado == estado.OK


def test_el_modo_confianza_encendido_se_ve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Que lo irreversible no pregunte tiene que estar a la vista mientras dure."""
    monkeypatch.setattr(politica, "CONFIRMACIONES", True)
    politica.activar_confianza(30)
    pieza = estado._confianza()
    assert pieza.estado == estado.AVISO
    assert "sin preguntar" in pieza.detalle


def test_apagadas_el_modo_confianza_no_cambia_la_pieza() -> None:
    """Encender la confianza con el interruptor apagado no cambia nada, y se dice.

    Es lo que justifica que el panel esconda el boton: ver `Panel.tsx`.
    """
    politica.activar_confianza(30)
    assert "Apagadas" in estado._confianza().detalle


# --------------------------------------------------------------------------- #
# Memoria y Google
# --------------------------------------------------------------------------- #


def test_el_vault_en_ficheros_esta_bien(
    monkeypatch: pytest.MonkeyPatch, datos: Path, vault: Path
) -> None:
    """Ficheros es el respaldo por defecto, no un estado degradado."""
    cfg = configuracion(monkeypatch, datos, OBSIDIAN_VAULT_PATH=str(vault))
    pieza = asyncio.run(estado._vault(cfg))
    assert pieza.estado == estado.OK
    assert str(vault) in pieza.detalle


def test_el_plugin_pedido_sin_clave_avisa_de_que_se_sigue_en_ficheros(
    monkeypatch: pytest.MonkeyPatch, datos: Path
) -> None:
    cfg = configuracion(monkeypatch, datos, PERSEO_VAULT="rest")
    pieza = asyncio.run(estado._vault(cfg))
    assert pieza.estado == estado.AVISO
    assert "ficheros" in pieza.detalle


def test_google_sin_pedir_no_toca_la_red(cfg: Configuracion) -> None:
    """Si nadie ha pedido Gmail ni Calendar, no se gasta ni una petición."""
    pieza = asyncio.run(estado._google(cfg))
    assert pieza.estado == estado.APAGADO


def test_google_sin_credenciales_dice_como_conseguirlas(
    monkeypatch: pytest.MonkeyPatch, datos: Path
) -> None:
    cfg = configuracion(monkeypatch, datos, PERSEO_CORREO="gmail")
    pieza = asyncio.run(estado._google(cfg))
    assert pieza.estado == estado.MALO
    assert "autorizar_google" in pieza.arreglo


# --------------------------------------------------------------------------- #
# Las dos reglas de la cabecera del módulo
# --------------------------------------------------------------------------- #


def test_un_sondeo_que_revienta_no_tumba_la_respuesta() -> None:
    """Regla 1: lo que salga mal se convierte en una pieza roja, no en un 500."""

    async def explota() -> estado.Pieza:
        raise ZeroDivisionError("algo muy inesperado")

    pieza = asyncio.run(estado._sin_caerse("x", "X", explota))
    assert pieza.estado == estado.MALO
    assert "inesperado" in pieza.detalle


def test_lo_recien_preguntado_no_se_vuelve_a_preguntar() -> None:
    """Regla 2: la pantalla se refresca sola y los sondeos son caros."""
    vueltas = 0

    async def sondear() -> estado.Pieza:
        nonlocal vueltas
        vueltas += 1
        return estado.Pieza("x", "X", estado.OK, "")

    async def guion() -> None:
        for _ in range(3):
            await estado._recordando("x", 60, sondear)

    asyncio.run(guion())
    assert vueltas == 1


def test_olvidar_vuelve_a_preguntar() -> None:
    vueltas = 0

    async def sondear() -> estado.Pieza:
        nonlocal vueltas
        vueltas += 1
        return estado.Pieza("x", "X", estado.OK, "")

    async def guion() -> None:
        await estado._recordando("x", 60, sondear)
        estado.olvidar()
        await estado._recordando("x", 60, sondear)

    asyncio.run(guion())
    assert vueltas == 2


# --------------------------------------------------------------------------- #
# El panel entero
# --------------------------------------------------------------------------- #


class _RouterFalso:
    disponible = True


def test_el_panel_trae_todo_lo_que_pinta_la_pantalla(
    db: Configuracion, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Un contrato: si desaparece una clave, la pestaña se queda en blanco."""

    async def sondeo_falso(*_: object, **__: object) -> estado.Pieza:
        return estado.Pieza("x", "X", estado.OK, "de mentira")

    monkeypatch.setattr(estado, "_ollama", sondeo_falso)
    monkeypatch.setattr(estado, "_vault", sondeo_falso)
    monkeypatch.setattr(estado, "_google", sondeo_falso)

    panel = asyncio.run(estado.reunir(db, _RouterFalso()))
    for clave in (
        "generado",
        "encendido_segundos",
        "piezas",
        "trabajos",
        "agentes",
        "disparadores",
        "cuota",
    ):
        assert clave in panel, clave
    assert panel["piezas"], "el panel no puede venir sin piezas"
    assert all({"id", "nombre", "estado", "detalle", "arreglo"} <= set(p) for p in panel["piezas"])


def test_el_panel_dice_que_disparadores_estan_apagados(
    db: Configuracion, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Los registrados y los encendidos no son lo mismo, y la diferencia importa:
    un disparador apagado explica por qué no llega ningún aviso."""

    async def sondeo_falso(*_: object, **__: object) -> estado.Pieza:
        return estado.Pieza("x", "X", estado.OK, "de mentira")

    monkeypatch.setattr(estado, "_ollama", sondeo_falso)
    monkeypatch.setattr(estado, "_vault", sondeo_falso)
    monkeypatch.setattr(estado, "_google", sondeo_falso)

    panel = asyncio.run(estado.reunir(db, _RouterFalso()))
    nombres = {d["nombre"] for d in panel["disparadores"]}
    assert {"correo", "agenda"} <= nombres
    assert all(isinstance(d["activo"], bool) for d in panel["disparadores"])
