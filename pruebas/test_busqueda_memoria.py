"""Buscar en el vault preguntando como se habla.

La búsqueda del vault es **literal**: con el plugin de Obsidian la hace él, y no
dobla tildes ni entiende sinónimos. Un modelo de voz pregunta como habla —"algo
de segundo de carrera sobre Ada"— y esa cadena no aparece en ninguna nota, así
que la respuesta era "no hay nada" con la nota delante. Pasó en una llamada real
el 2026-08-16.

Se arregla en el núcleo y no en el prompt porque las tres caras —la voz, el panel
del PC y el móvil— preguntan igual de mal.
"""

from __future__ import annotations

import asyncio

from perseo_core.agentes.memoria import Nota, buscar_con_reintentos, _terminos


class VaultFalso:
    """Un vault que solo encuentra lo que contiene el término, literal."""

    def __init__(self, notas: dict[str, list[str]]) -> None:
        #: ruta -> palabras que la encuentran
        self._notas = notas
        self.consultas: list[str] = []

    async def buscar(self, consulta: str, limite: int = 10) -> list[Nota]:
        self.consultas.append(consulta)
        halladas = [
            Nota(titulo=ruta.split("/")[-1], ruta=ruta, extracto="…")
            for ruta, palabras in self._notas.items()
            if consulta in palabras
        ]
        return halladas[:limite]


# --------------------------------------------------------------------------- #
# Trocear la consulta
# --------------------------------------------------------------------------- #


def test_se_caen_las_palabras_que_no_distinguen() -> None:
    assert _terminos("algo de segundo de carrera sobre Ada") == ["segundo", "carrera", "Ada"]


def test_no_se_tocan_las_mayusculas() -> None:
    """La búsqueda la hace el vault; si distingue mayúsculas no es asunto nuestro."""
    assert "MAGI" in _terminos("¿Qué pone sobre el proyecto MAGI?")


def test_se_quitan_los_signos() -> None:
    assert "MAGI" in _terminos("MAGI,")
    assert "cita" in _terminos("¿la cita?")


def test_no_se_repiten() -> None:
    assert _terminos("Ada y Ada otra vez Ada").count("Ada") == 1


# --------------------------------------------------------------------------- #
# El reintento
# --------------------------------------------------------------------------- #


def test_las_palabras_se_buscan_aunque_la_frase_acierte() -> None:
    """No es un despilfarro: es lo que permite ordenar.

    Que la frase devuelva algo no quiere decir que devuelva lo bueno — el plugin
    encuentra resultados para casi cualquier cosa. Sin buscar también las
    palabras sueltas no hay forma de saber cuál distingue, y era justo lo que
    hacía que "el proyecto MAGI" contestara sobre una plantilla.
    """
    vault = VaultFalso({"n.md": ["proyecto MAGI", "MAGI"]})
    notas, _ = asyncio.run(buscar_con_reintentos(vault, "proyecto MAGI", 10))
    assert len(notas) == 1
    assert "proyecto MAGI" in vault.consultas and "MAGI" in vault.consultas


def test_la_frase_que_no_existe_se_trocea() -> None:
    """El caso de la llamada real: la nota existe y la frase no."""
    vault = VaultFalso({"02_PROYECTOS/Ada.md": ["Ada"]})
    notas, buscado = asyncio.run(
        buscar_con_reintentos(vault, "algo de segundo de carrera sobre Ada", 10)
    )
    assert [n.ruta for n in notas] == ["02_PROYECTOS/Ada.md"]
    assert buscado == "Ada"


def test_manda_la_palabra_que_menos_devuelve() -> None:
    """La que menos devuelve es la que más distingue.

    "proyecto" sale en media biblioteca y "MAGI" en una nota: por longitud se
    habría probado "proyecto" primero y el hueco se habría llenado de ruido.
    """
    vault = VaultFalso({
        "a.md": ["proyecto"],
        "b.md": ["proyecto"],
        "c.md": ["proyecto"],
        "MAGI.md": ["MAGI", "proyecto"],
    })
    notas, buscado = asyncio.run(buscar_con_reintentos(vault, "el proyecto MAGI ese", 10))
    assert notas[0].ruta == "MAGI.md"
    assert buscado.startswith("MAGI")


def test_una_sola_palabra_no_se_reintenta() -> None:
    """No hay nada que trocear, y una segunda petición idéntica es tiempo tirado."""
    vault = VaultFalso({})
    notas, buscado = asyncio.run(buscar_con_reintentos(vault, "Ada", 10))
    assert notas == [] and buscado == "Ada"
    assert vault.consultas == ["Ada"]


def test_no_se_repiten_notas_encontradas_por_dos_palabras() -> None:
    vault = VaultFalso({"n.md": ["Ada", "carrera"]})
    notas, _ = asyncio.run(buscar_con_reintentos(vault, "carrera de Ada", 10))
    assert len(notas) == 1


def test_se_respeta_el_limite() -> None:
    vault = VaultFalso({f"n{i}.md": ["Ada", "carrera"] for i in range(10)})
    notas, _ = asyncio.run(buscar_con_reintentos(vault, "carrera de Ada", 3))
    assert len(notas) == 3


def test_no_se_prueban_diez_palabras() -> None:
    """Cada palabra es una petición al vault; una frase larga no mejora por
    buscar su décima."""
    vault = VaultFalso({})
    asyncio.run(
        buscar_con_reintentos(
            vault, "una frase larguisima llena palabras distintas todas ellas", 10
        )
    )
    assert len(vault.consultas) <= 5  # la frase entera y cuatro términos
