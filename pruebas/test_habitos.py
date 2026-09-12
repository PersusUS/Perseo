"""El espejo del seguimiento de hábitos.

Lo que se prueba aquí es lo poco que el núcleo hace con los hábitos, que es a
propósito muy poco: guardar la copia que le manda la ventana, devolverla, y
avisar cuando la copia está pasada. Contar casillas no se prueba aquí porque no
se hace aquí — se hace una sola vez, en `RealTime/src/lib/datos/habitos.ts`, y de eso
responden `RealTime/pruebas/habitos.test.ts`.

El caso que de verdad importa es el tercero: **un almacén vacío tiene que
decirlo**. Un Perseo que, sin copia, contesta «vas muy bien este mes» es peor
que uno que no sabe nada de hábitos, porque el señor Persus se lo creería.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from perseo_core.servicios import habitos


def test_lo_guardado_vuelve_tal_cual(tmp_path: Path) -> None:
    copia = habitos.guardar(tmp_path, "Hábitos a 25/08/2026: 171 de 300 casillas.")

    assert copia["sellado"]
    assert habitos.cargar(tmp_path)["texto"] == "Hábitos a 25/08/2026: 171 de 300 casillas."
    # El texto sale entero y sin adornos: el aviso solo aparece si viene viejo.
    assert habitos.resumen(tmp_path) == "Hábitos a 25/08/2026: 171 de 300 casillas."


def test_la_foto_viaja_al_lado_del_texto(tmp_path: Path) -> None:
    """El texto es para hablar; la foto está por si algún día alguien la mira."""
    habitos.guardar(tmp_path, "resumen", {"hechos": 171, "objetivo": 300})
    assert habitos.cargar(tmp_path)["foto"] == {"hechos": 171, "objetivo": 300}

    # Y una foto que no sea un objeto no se guarda a medias.
    habitos.guardar(tmp_path, "resumen", None)
    assert habitos.cargar(tmp_path)["foto"] is None


def test_sin_copia_lo_dice_en_vez_de_suponer(tmp_path: Path) -> None:
    assert habitos.cargar(tmp_path) is None
    texto = habitos.resumen(tmp_path)
    assert "No hay copia" in texto
    # Y le dice al modelo qué hacer con eso, que es no inventarse cómo va.
    assert "en vez de suponer" in texto


def test_una_copia_rota_vale_lo_mismo_que_ninguna(tmp_path: Path) -> None:
    """Medio JSON daría medio recuento, que es peor que ninguno."""
    habitos.ruta(tmp_path).write_text('{"texto": "a mitad', encoding="utf-8")
    assert habitos.cargar(tmp_path) is None
    assert "No hay copia" in habitos.resumen(tmp_path)


def test_una_copia_vieja_avisa_antes_de_contar(tmp_path: Path) -> None:
    """El aviso va DELANTE: un modelo que lee doce líneas de cifras y encuentra
    la salvedad al final ya ha decidido cómo contestar."""
    habitos.guardar(tmp_path, "Van 171 de 300.")
    viejo = json.loads(habitos.ruta(tmp_path).read_text(encoding="utf-8"))
    viejo["sellado"] = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat(timespec="seconds")
    habitos.ruta(tmp_path).write_text(json.dumps(viejo), encoding="utf-8")

    texto = habitos.resumen(tmp_path)
    assert texto.startswith("Atención:")
    assert "de hace 3 días" in texto
    # Pero la copia se sigue dando: vieja es mejor que nada.
    assert "Van 171 de 300." in texto


def test_una_copia_de_hoy_no_molesta_con_avisos(tmp_path: Path) -> None:
    habitos.guardar(tmp_path, "Van 171 de 300.")
    reciente = json.loads(habitos.ruta(tmp_path).read_text(encoding="utf-8"))
    reciente["sellado"] = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat(timespec="seconds")
    habitos.ruta(tmp_path).write_text(json.dumps(reciente), encoding="utf-8")

    assert habitos.resumen(tmp_path) == "Van 171 de 300."


def test_la_escritura_no_deja_medio_fichero(tmp_path: Path) -> None:
    """La ventana manda esto en cada cambio; un corte a mitad tiene que dejar la
    copia anterior entera, no un JSON partido."""
    habitos.guardar(tmp_path, "primera")
    habitos.guardar(tmp_path, "segunda")

    assert habitos.cargar(tmp_path)["texto"] == "segunda"
    # Y el temporal no se queda por ahí.
    assert not list(tmp_path.glob("*.tmp"))
