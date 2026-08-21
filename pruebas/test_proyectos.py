"""Los otros proyectos: qué se puede abrir y, sobre todo, qué no.

El panel se alcanza desde el tailnet, así que lo que decide qué se ejecuta es un
fichero del disco y nunca la petición. Estas pruebas comprueban el borde.
"""

from __future__ import annotations

import json
import sys
import time
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


# --------------------------------------------------------------------------- #
# Arrancar el proyecto, no abrir su carpeta (T-6, 2026-08-21)
# --------------------------------------------------------------------------- #


def _con_lista(tmp_path: Path, entradas: list[dict]) -> Path:
    escribir(tmp_path, entradas)
    return tmp_path


def test_un_arranque_bien_escrito_entra_en_la_lista(tmp_path) -> None:
    datos = _con_lista(
        tmp_path,
        [{"id": "web", "nombre": "Web", "modo": "arranque",
          "arranque": [sys.executable, "-c", "pass"], "carpeta": str(tmp_path)}],
    )
    lista = proyectos.listar(datos)
    assert len(lista) == 1
    assert lista[0].arranque == (sys.executable, "-c", "pass")


def test_un_arranque_sin_destino_vale(tmp_path) -> None:
    """Es el único modo que no lo necesita: lo que se abre es la orden."""
    datos = _con_lista(
        tmp_path,
        [{"id": "web", "modo": "arranque", "arranque": [sys.executable, "-V"]}],
    )
    assert len(proyectos.listar(datos)) == 1


def test_una_orden_en_una_sola_cadena_se_rechaza(tmp_path) -> None:
    """`"npm run dev"` de una pieza solo se puede ejecutar dándoselo a un shell,
    y ahí es donde viven las comillas y los `&&`. Se exige lista."""
    datos = _con_lista(
        tmp_path, [{"id": "web", "modo": "arranque", "arranque": "npm run dev"}]
    )
    assert proyectos.listar(datos) == []


def test_un_arranque_vacio_o_con_huecos_se_rechaza(tmp_path) -> None:
    datos = _con_lista(
        tmp_path,
        [
            {"id": "a", "modo": "arranque", "arranque": []},
            {"id": "b", "modo": "arranque", "arranque": [sys.executable, "  "]},
            {"id": "c", "modo": "arranque"},
        ],
    )
    assert proyectos.listar(datos) == []


def test_un_programa_que_no_existe_se_descarta_al_leer(tmp_path) -> None:
    """Como una carpeta que ya no está: se dice al leer la lista, en vez de
    dejar el botón puesto para que falle al pulsarlo."""
    datos = _con_lista(
        tmp_path,
        [{"id": "web", "modo": "arranque", "arranque": ["no-existe-este-programa-jamas"]}],
    )
    assert proyectos.listar(datos) == []


def test_una_entrada_mala_no_se_lleva_por_delante_a_las_buenas(tmp_path) -> None:
    datos = _con_lista(
        tmp_path,
        [
            {"id": "malo", "modo": "arranque", "arranque": "npm run dev"},
            {"id": "bueno", "modo": "arranque", "arranque": [sys.executable, "-V"]},
        ],
    )
    assert [p.id for p in proyectos.listar(datos)] == ["bueno"]


def test_arrancar_lanza_el_programa_y_lo_dice(tmp_path) -> None:
    """De punta a punta y con un proceso de verdad: se lanza y se contesta que
    se lanzó, que es lo único que se puede saber en ese momento."""
    testigo = tmp_path / "arranco.txt"
    guion = f"open(r'{testigo}', 'w').write('si')"
    datos = _con_lista(
        tmp_path,
        [{"id": "web", "nombre": "Web", "modo": "arranque",
          "arranque": [sys.executable, "-c", guion], "carpeta": str(tmp_path)}],
    )

    respuesta = proyectos.abrir(datos, "web")
    assert respuesta.startswith("Éxito")

    for _ in range(50):
        if testigo.exists():
            break
        time.sleep(0.1)
    assert testigo.read_text(encoding="utf-8") == "si"


def test_arrancar_desde_una_carpeta_que_ya_no_existe_lo_dice(tmp_path) -> None:
    datos = _con_lista(
        tmp_path,
        [{"id": "web", "nombre": "Web", "modo": "arranque",
          "arranque": [sys.executable, "-V"], "carpeta": str(tmp_path / "fantasma")}],
    )
    respuesta = proyectos.abrir(datos, "web")
    assert respuesta.startswith("Error") and "ya no existe" in respuesta


def test_por_http_sigue_viajando_solo_el_id() -> None:
    """La regla que sostiene todo: `abrir` recibe un identificador, y lo que se
    ejecuta sale del fichero del disco. Si algún día acepta la orden por
    parámetro, esto tiene que ponerse rojo."""
    import inspect

    firma = inspect.signature(proyectos.abrir)
    assert list(firma.parameters) == ["directorio_datos", "id_proyecto"]


def test_nunca_se_invoca_un_shell() -> None:
    """La otra regla de la casa. Un `shell=True` aquí convertiría la lista del
    disco en una línea que el sistema vuelve a parsear.

    Se mira el árbol y no el texto: la cabecera del módulo habla de `shell=True`
    justo para explicar por qué no lo hay, y buscar la cadena encontraría eso.
    """
    import ast
    import inspect

    arbol = ast.parse(inspect.getsource(proyectos))
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.Call):
            continue
        for argumento in nodo.keywords:
            assert argumento.arg != "shell" or not getattr(
                argumento.value, "value", False
            ), "hay un shell=True en proyectos.py"
