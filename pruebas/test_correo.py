"""El correo: el triaje escala, el titular no suelta contenido."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from perseo_core import correo, triaje


def test_el_titular_cuenta_y_no_suelta_asuntos() -> None:
    """Lo que sale por Telegram es un recuento, nunca el asunto."""
    recuento = triaje.recontar(
        [
            triaje.Clasificacion(triaje.REQUIERE_ACCION, "hay que contestar"),
            triaje.Clasificacion(triaje.IGNORAR, "publicidad"),
        ]
    )
    texto = correo.titular(recuento)
    assert texto.startswith("2 correos")
    assert "1 requiere acción" in texto
    assert "hay que contestar" not in texto


def test_sin_nada_relevante_no_hay_titular() -> None:
    """Un canal que avisa de nada acaba silenciado."""
    recuento = triaje.recontar([triaje.Clasificacion(triaje.IGNORAR, "publicidad")])
    assert correo.titular(recuento) is None


def test_lo_dudoso_tambien_avisa() -> None:
    recuento = triaje.recontar([triaje.Clasificacion(triaje.NO_SEGURO, "ni idea")])
    assert correo.titular(recuento) is not None


def test_el_titular_concuerda_en_singular() -> None:
    recuento = triaje.recontar([triaje.Clasificacion(triaje.REQUIERE_ACCION, "x")])
    assert correo.titular(recuento) == "1 correo: 1 requiere acción"


def test_recontar_siempre_trae_todas_las_claves() -> None:
    """Quien lo lee no debe distinguir entre «cero» y «no vino ese campo»."""
    recuento = triaje.recontar([])
    for clase in triaje.CLASES:
        assert recuento[clase] == 0
    assert recuento["total"] == 0


def test_el_buzon_falso_lee_del_disco(tmp_path: Path) -> None:
    fichero = tmp_path / "buzon.json"
    fichero.write_text(
        json.dumps([{"id": "1", "remitente": "a@b.c", "asunto": "Hola"}]), encoding="utf-8"
    )
    mensajes = asyncio.run(correo.BuzonFalso(fichero).nuevos())
    assert len(mensajes) == 1
    assert mensajes[0].asunto == "Hola"


def test_un_buzon_que_no_existe_no_rompe(tmp_path: Path) -> None:
    assert asyncio.run(correo.BuzonFalso(tmp_path / "no_esta.json").nuevos()) == []


def test_un_buzon_corrupto_no_rompe(tmp_path: Path) -> None:
    fichero = tmp_path / "buzon.json"
    fichero.write_text("{esto no es json", encoding="utf-8")
    assert asyncio.run(correo.BuzonFalso(fichero).nuevos()) == []


def test_un_buzon_que_no_es_una_lista_no_rompe(tmp_path: Path) -> None:
    fichero = tmp_path / "buzon.json"
    fichero.write_text('{"mensajes": []}', encoding="utf-8")
    assert asyncio.run(correo.BuzonFalso(fichero).nuevos()) == []


def test_abrir_buzon_sin_configurar_devuelve_nada(cfg) -> None:
    """No configurado no es lo mismo que roto."""
    assert correo.abrir_buzon(cfg) is None


def test_el_agente_tria_un_lote(monkeypatch) -> None:
    class TriajeFalso:
        async def clasificar(self, mensaje):
            return triaje.Clasificacion(triaje.REQUIERE_ACCION, "porque si")

    monkeypatch.setattr(correo, "_triaje", TriajeFalso())
    resultado = asyncio.run(
        correo._correo(
            {
                "peticion": {
                    "accion": "triar",
                    "mensajes": [
                        {"id": "1", "remitente": "a@b.c", "asunto": "Presupuesto", "extracto": "x"}
                    ],
                }
            }
        )
    )

    assert resultado["recuento"]["total"] == 1
    assert resultado["clasificados"][0]["clase"] == triaje.REQUIERE_ACCION
    # El detalle se queda en la cola; el titular es lo que sale por Telegram.
    assert resultado["clasificados"][0]["asunto"] == "Presupuesto"
    assert "Presupuesto" not in resultado["titular"]


def test_un_lote_vacio_no_avisa(monkeypatch) -> None:
    monkeypatch.setattr(correo, "_triaje", object())
    resultado = asyncio.run(correo._correo({"peticion": {"accion": "triar", "mensajes": []}}))
    assert resultado["recuento"]["total"] == 0
    assert resultado["titular"] is None


def test_sin_triaje_iniciado_el_agente_lo_dice(monkeypatch) -> None:
    monkeypatch.setattr(correo, "_triaje", None)
    with pytest.raises(RuntimeError, match="iniciar"):
        asyncio.run(
            correo._correo({"peticion": {"accion": "triar", "mensajes": [{"id": "1"}]}})
        )
