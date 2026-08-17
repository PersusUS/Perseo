"""Los otros proyectos: qué se puede abrir y, sobre todo, qué no.

El panel se alcanza desde el tailnet, así que lo que decide qué se ejecuta es un
fichero del disco y nunca la petición. Estas pruebas comprueban el borde.
"""

from __future__ import annotations

import json
from pathlib import Path

from perseo_core import proyectos


def escribir(directorio: Path, entradas: list[dict]) -> None:
    (directorio / proyectos.NOMBRE_FICHERO).write_text(
        json.dumps(entradas), encoding="utf-8"
    )


def test_sin_fichero_no_hay_proyectos(tmp_path: Path) -> None:
    """Es un estado válido: la pantalla enseña cómo crearlo."""
    assert proyectos.listar(tmp_path) == []


def test_un_fichero_roto_no_revienta(tmp_path: Path) -> None:
    (tmp_path / proyectos.NOMBRE_FICHERO).write_text("{esto no es json", encoding="utf-8")
    assert proyectos.listar(tmp_path) == []


def test_una_carpeta_se_lee(tmp_path: Path) -> None:
    escribir(tmp_path, [{"id": "armario", "nombre": "Armario", "modo": "carpeta",
                         "destino": str(tmp_path)}])
    lista = proyectos.listar(tmp_path)
    assert [p.id for p in lista] == ["armario"]
    assert lista[0].nombre == "Armario"


def test_una_entrada_rota_no_se_lleva_las_buenas(tmp_path: Path) -> None:
    """El fichero lo escribe una persona: perder cinco proyectos por una coma
    es peor que perder el que está mal."""
    escribir(tmp_path, [
        {"id": "roto", "modo": "telepatia", "destino": "x"},
        {"id": "bueno", "modo": "carpeta", "destino": str(tmp_path)},
    ])
    assert [p.id for p in proyectos.listar(tmp_path)] == ["bueno"]


def test_un_esquema_que_no_es_http_se_descarta(tmp_path: Path) -> None:
    escribir(tmp_path, [{"id": "malo", "modo": "url", "destino": "file:///C:/Windows"}])
    assert proyectos.listar(tmp_path) == []


def test_un_programa_fuera_de_la_lista_blanca_se_descarta(tmp_path: Path) -> None:
    """La lista blanca es la misma del agente `pc`: dos acabarían discrepando."""
    escribir(tmp_path, [{"id": "malo", "modo": "programa", "destino": "powershell",
                         "carpeta": str(tmp_path)}])
    assert proyectos.listar(tmp_path) == []


def test_un_programa_de_la_lista_blanca_si_vale(tmp_path: Path) -> None:
    escribir(tmp_path, [{"id": "notas", "modo": "programa", "destino": "notepad",
                         "carpeta": str(tmp_path)}])
    assert [p.destino for p in proyectos.listar(tmp_path)] == ["notepad"]


def test_abrir_algo_que_no_esta_en_la_lista_es_un_error(tmp_path: Path) -> None:
    """Por HTTP llega cuál de los proyectos, no qué ejecutar."""
    escribir(tmp_path, [{"id": "armario", "modo": "carpeta", "destino": str(tmp_path)}])
    assert proyectos.abrir(tmp_path, "cmd").startswith("Error:")
    assert proyectos.abrir(tmp_path, "../../otra-cosa").startswith("Error:")


def test_abrir_una_carpeta_que_ya_no_existe_lo_dice(tmp_path: Path) -> None:
    escribir(tmp_path, [{"id": "fantasma", "modo": "carpeta",
                         "destino": str(tmp_path / "no-existe")}])
    assert proyectos.abrir(tmp_path, "fantasma").startswith("Error:")
