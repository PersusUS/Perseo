"""La forma del repositorio, comprobada.

Aquí no se prueba qué hace el código sino **dónde vive**. Es el sitio donde
paran los fallos que no son de programación: dos copias que se desincronizan, un
ciclo de importaciones esquivado con un truco, un fichero que crece hasta que
nadie lo lee entero.

La razón de que esto exista y no sea una nota en `AGENTS.md`: una carpeta bien
puesta se deshace en tres meses; una prueba que falla en el CI, no.

Las reglas y sus listas de excepciones viven en `commands/arquitectura.py`. Las
listas **solo pueden encoger**: hay una prueba para eso también, porque una
excepción que sobra es lo que convierte una regla en decoración.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "commands"))

import arquitectura  # noqa: E402


def test_el_nucleo_no_tiene_ciclos_nuevos() -> None:
    """Un ciclo se paga con importes dentro de funciones, que son deuda oculta.

    El que había cuando se escribió esto está apuntado en `CICLOS_CONOCIDOS` con
    su porqué. Cualquier otro es rojo.
    """
    nuevos = arquitectura.ciclos_nuevos(arquitectura.grafo_del_nucleo())
    assert not nuevos, "ciclos de importación nuevos: " + "; ".join(
        " -> ".join(c) for c in nuevos
    )


def test_la_lista_de_ciclos_conocidos_no_tiene_basura() -> None:
    """Un ciclo ya deshecho que sigue en la lista deja de proteger de nada."""
    sobran = arquitectura.ciclos_ya_deshechos(arquitectura.grafo_del_nucleo())
    assert not sobran, "ciclos apuntados que ya no existen, quítalos: " + "; ".join(
        " -> ".join(c) for c in sobran
    )


def test_ningun_fichero_pasa_del_techo() -> None:
    """Un fichero de mil quinientas líneas es donde se esconde lo duplicado.

    Cubre los dos casos: un fichero nuevo que se pasa, y uno de la lista de
    excepciones que **crece**. La excepción era para el tamaño de entonces.
    """
    rotos = arquitectura.techo_roto()
    detalle = ", ".join(f"{n} ({c} líneas)" for n, c in sorted(rotos.items()))
    assert not rotos, f"pasan del techo de {arquitectura.TECHO_DURO} líneas: {detalle}"


def test_la_lista_de_excepciones_de_tamano_no_tiene_basura() -> None:
    """Igual que con los ciclos: lo que ya se partió sale de la lista."""
    sobran = arquitectura.excepciones_muertas()
    assert not sobran, "ya están por debajo del techo, quítalos: " + ", ".join(sobran)


def test_nadie_importa_hacia_arriba() -> None:
    """La regla que hace que las capas sigan existiendo dentro de tres meses.

    `dominio` no sabe de nadie, `infra` solo del dominio, y así hasta `caras`.
    Los módulos que aún viven en la raíz del paquete no se juzgan; según se
    mudan, esta prueba los va cubriendo sola.
    """
    saltos = arquitectura.saltos_de_capa(arquitectura.grafo_del_nucleo())
    detalle = ", ".join(f"{quien} -> {que}" for quien, que in saltos)
    assert not saltos, f"importes hacia arriba: {detalle}"


# ==========================================================================
# La frontera de la cara
# ==========================================================================
def test_ningun_componente_nuevo_llama_al_nucleo() -> None:
    """«Las caras no piensan», que hasta hoy solo estaba escrito.

    Un componente que abre él mismo la puerta a Rust hace dos trabajos —pintar y
    decidir qué pedir— y no se puede probar sin Rust delante. Lo que debe hacer
    es pedirle los datos a un gancho de `lib/datos/`.

    Los que ya lo hacen están perdonados por su nombre, y esa lista solo puede
    encoger: lo que esta prueba impide es que aparezca **uno más**.
    """
    nuevos = sorted(
        arquitectura.componentes_con_puerta_propia()
        - arquitectura.COMPONENTES_QUE_LLAMAN_AL_NUCLEO
    )
    assert not nuevos, (
        "abren ellos mismos la puerta al núcleo; pídele los datos a un gancho de "
        "lib/datos/: " + ", ".join(nuevos)
    )


def test_la_lista_de_componentes_perdonados_no_tiene_basura() -> None:
    """Uno que ya se arregló y sigue en la lista deja de proteger de nada."""
    sobran = sorted(
        arquitectura.COMPONENTES_QUE_LLAMAN_AL_NUCLEO
        - arquitectura.componentes_con_puerta_propia()
    )
    assert not sobran, "ya no llaman al núcleo, quítalos de la lista: " + ", ".join(sobran)
