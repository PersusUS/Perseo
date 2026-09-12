"""El espejo del tablero de tareas.

Lo que se prueba aquí es lo poco que el núcleo hace con las tareas, que es a
propósito muy poco: guardar la copia que le manda la ventana, devolverla, y
avisar cuando la copia está pasada. Contar notas no se prueba aquí porque no se
hace aquí — se hace una sola vez, en `RealTime/src/lib/tareas.ts`, y de eso
responden `RealTime/pruebas/tareas.test.ts`.

Los dos casos que de verdad importan son el del almacén vacío y el de la copia
vieja, y el segundo pesa más que en los hábitos: un Perseo que recuerda una
tarea que el señor Persus cerró ayer no solo se equivoca, molesta.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from perseo_core.servicios import tareas


def test_lo_guardado_vuelve_tal_cual(tmp_path: Path) -> None:
    copia = tareas.guardar(tmp_path, "Tablero: 3 sin hacer, 1 en proceso.")

    assert copia["sellado"]
    assert tareas.cargar(tmp_path)["texto"] == "Tablero: 3 sin hacer, 1 en proceso."
    # El texto sale entero y sin adornos: el aviso solo aparece si viene viejo.
    assert tareas.resumen(tmp_path) == "Tablero: 3 sin hacer, 1 en proceso."


def test_la_foto_viaja_al_lado_del_texto(tmp_path: Path) -> None:
    """El texto es para hablar; la foto está por si algún día alguien la mira."""
    tareas.guardar(tmp_path, "resumen", {"sinHacer": 3, "enProceso": 1})
    assert tareas.cargar(tmp_path)["foto"] == {"sinHacer": 3, "enProceso": 1}

    # Y una foto que no sea un objeto no se guarda a medias.
    tareas.guardar(tmp_path, "resumen", None)
    assert tareas.cargar(tmp_path)["foto"] is None


def test_sin_copia_lo_dice_en_vez_de_suponer(tmp_path: Path) -> None:
    assert tareas.cargar(tmp_path) is None
    texto = tareas.resumen(tmp_path)
    assert "No hay copia" in texto
    # Y le dice al modelo qué hacer con eso, que es no inventarse la lista.
    assert "en vez de suponer" in texto


def test_una_copia_rota_vale_lo_mismo_que_ninguna(tmp_path: Path) -> None:
    """Media lista de pendientes parece una lista entera, y ahí está el daño."""
    tareas.ruta(tmp_path).write_text('{"texto": "a mitad', encoding="utf-8")
    assert tareas.cargar(tmp_path) is None
    assert "No hay copia" in tareas.resumen(tmp_path)


def test_una_copia_vieja_avisa_antes_de_enumerar(tmp_path: Path) -> None:
    """El aviso va DELANTE. Una tarea cerrada ayer y recordada hoy como
    pendiente es lo peor que puede salir de este fichero."""
    tareas.guardar(tmp_path, "Pendientes: comprar pan.")
    viejo = json.loads(tareas.ruta(tmp_path).read_text(encoding="utf-8"))
    viejo["sellado"] = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat(timespec="seconds")
    tareas.ruta(tmp_path).write_text(json.dumps(viejo), encoding="utf-8")

    texto = tareas.resumen(tmp_path)
    assert texto.startswith("Atención:")
    assert "de hace 3 días" in texto
    # Pero la copia se sigue dando: vieja es mejor que nada.
    assert "Pendientes: comprar pan." in texto


def test_una_copia_de_hoy_no_molesta_con_avisos(tmp_path: Path) -> None:
    tareas.guardar(tmp_path, "Pendientes: comprar pan.")
    reciente = json.loads(tareas.ruta(tmp_path).read_text(encoding="utf-8"))
    reciente["sellado"] = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat(timespec="seconds")
    tareas.ruta(tmp_path).write_text(json.dumps(reciente), encoding="utf-8")

    assert tareas.resumen(tmp_path) == "Pendientes: comprar pan."


def test_la_escritura_no_deja_medio_fichero(tmp_path: Path) -> None:
    """La ventana manda esto en cada cambio del tablero; un corte a mitad tiene
    que dejar la copia anterior entera, no un JSON partido."""
    tareas.guardar(tmp_path, "primera")
    tareas.guardar(tmp_path, "segunda")

    assert tareas.cargar(tmp_path)["texto"] == "segunda"
    # Y el temporal no se queda por ahí.
    assert not list(tmp_path.glob("*.tmp"))


def test_el_espejo_de_tareas_no_pisa_al_de_habitos(tmp_path: Path) -> None:
    """Los dos buzones comparten directorio. Si compartiesen fichero, abrir la
    pantalla de tareas borraría los hábitos y nadie lo notaría hasta preguntar."""
    from perseo_core.servicios import habitos

    habitos.guardar(tmp_path, "hábitos")
    tareas.guardar(tmp_path, "tareas")

    assert habitos.cargar(tmp_path)["texto"] == "hábitos"
    assert tareas.cargar(tmp_path)["texto"] == "tareas"


# --------------------------------------------------------------------------- #
# Lo que se le pide a la ventana
#
# El núcleo no escribe el tablero: encola. Lo que se prueba aquí es el portero
# —qué órdenes pasan y cuáles no— y que recoger vacía, porque una orden que se
# aplicara dos veces clavaría la nota dos veces.
# --------------------------------------------------------------------------- #


def test_una_orden_bien_puesta_espera_a_la_ventana(tmp_path: Path) -> None:
    orden = tareas.encolar(tmp_path, "crear", "  Llamar al fontanero  ", detalle="el del bajo")

    assert orden["accion"] == "crear"
    assert orden["titulo"] == "Llamar al fontanero"
    assert orden["detalle"] == "el del bajo"
    assert orden["pedida"]
    assert tareas.ordenes(tmp_path) == [orden]


def test_recoger_entrega_y_vacia(tmp_path: Path) -> None:
    """Si no vaciara, la ventana clavaría la misma nota en cada vuelta."""
    tareas.encolar(tmp_path, "crear", "Comprar pan")
    tareas.encolar(tmp_path, "mover", "Comprar pan", columna="completadas")

    recogidas = tareas.recoger(tmp_path)
    assert [o["titulo"] for o in recogidas] == ["Comprar pan", "Comprar pan"]
    assert tareas.recoger(tmp_path) == []


def test_el_orden_se_respeta(tmp_path: Path) -> None:
    """Crear y mover la misma nota en la misma vuelta solo funciona en orden."""
    tareas.encolar(tmp_path, "crear", "Comprar pan")
    tareas.encolar(tmp_path, "mover", "Comprar pan", columna="en_proceso")

    assert [o["accion"] for o in tareas.recoger(tmp_path)] == ["crear", "mover"]


def test_lo_que_no_se_entiende_se_rechaza_aqui(tmp_path: Path) -> None:
    """Rechazar en el núcleo se le puede contar al modelo en el acto; una orden
    que viaja y muere al otro lado se pierde en silencio."""
    for accion, titulo, extra in [
        ("borrar", "Comprar pan", {}),          # acción que no existe
        ("crear", "   ", {}),                    # nota sin título
        ("mover", "Comprar pan", {}),            # mover sin destino
        ("mover", "Comprar pan", {"columna": "inventada"}),
    ]:
        with pytest.raises(ValueError):
            tareas.encolar(tmp_path, accion, titulo, **extra)

    assert tareas.ordenes(tmp_path) == []


def test_la_cola_no_crece_sin_tope(tmp_path: Path) -> None:
    """Con la app cerrada una semana, lo que sobra son las viejas."""
    for i in range(tareas.TOPE_ORDENES + 10):
        tareas.encolar(tmp_path, "crear", f"nota {i}")

    cola = tareas.ordenes(tmp_path)
    assert len(cola) == tareas.TOPE_ORDENES
    # Se queda la última, no la primera: lo que acaba de pedir importa más.
    assert cola[-1]["titulo"] == f"nota {tareas.TOPE_ORDENES + 9}"


def test_una_cola_rota_no_bloquea_lo_siguiente(tmp_path: Path) -> None:
    tareas.ruta_ordenes(tmp_path).write_text("[esto no es json", encoding="utf-8")
    assert tareas.ordenes(tmp_path) == []

    tareas.encolar(tmp_path, "crear", "Comprar pan")
    assert [o["titulo"] for o in tareas.ordenes(tmp_path)] == ["Comprar pan"]


def test_las_ordenes_no_pisan_el_espejo(tmp_path: Path) -> None:
    """Uno es lo que hay y el otro lo que se ha pedido: ficheros distintos."""
    tareas.guardar(tmp_path, "Tablero: 1 sin hacer.")
    tareas.encolar(tmp_path, "crear", "Comprar pan")

    assert tareas.cargar(tmp_path)["texto"] == "Tablero: 1 sin hacer."
    tareas.recoger(tmp_path)
    assert tareas.cargar(tmp_path)["texto"] == "Tablero: 1 sin hacer."
