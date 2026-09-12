"""El catálogo de herramientas: una sola fuente, y las copias vigiladas.

Hasta el 2026-09-12 las herramientas estaban declaradas dos veces —enteras en
TypeScript para la llamada y enteras en Python para el chat escrito— y las dos
copias se habían separado en algo que no era cosmético: el chat anunciaba una
acción `navegar_url` que el agente `pc` no tiene. El modelo podía pedirla, `pc`
la rechazaba como desconocida, la política trata lo desconocido como
irreversible, y el trabajo se quedaba esperando un sí que nadie llegaba a ver.

Ahora la fuente es `servicios/catalogo.py`. Queda **una** copia, la que la cara
de la voz lleva incrustada para poder abrir la llamada cuando el núcleo todavía
no está levantado, y estas pruebas son lo que impide que envejezca.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "commands"))

import comprobar  # noqa: E402

from perseo_core.servicios import catalogo  # noqa: E402

CHAT_PY = RAIZ / "perseo_core" / "agentes" / "chat.py"
PC_PY = RAIZ / "perseo_core" / "agentes" / "pc.py"


def despacha(fichero: Path, variable: str) -> set[str]:
    """Los valores que un despachador compara con `==`, leídos del código.

    Se lee el árbol en vez de mantener una lista al lado, porque una lista al
    lado es otra copia que desincronizar — que es justo lo que este fichero
    existe para impedir.
    """
    import ast

    arbol = ast.parse(fichero.read_text(encoding="utf-8"))
    valores: set[str] = set()
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.Compare) or len(nodo.ops) != 1:
            continue
        if not isinstance(nodo.ops[0], ast.Eq):
            continue
        izquierda, derecha = nodo.left, nodo.comparators[0]
        if isinstance(izquierda, ast.Name) and izquierda.id == variable:
            if isinstance(derecha, ast.Constant) and isinstance(derecha.value, str):
                valores.add(derecha.value)
    # Y los `if nombre in ("crear_tarea", "mover_tarea")`, que son la misma
    # decisión escrita para dos casos que comparten cuerpo.
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.Compare) or len(nodo.ops) != 1:
            continue
        if not isinstance(nodo.ops[0], ast.In):
            continue
        izquierda, derecha = nodo.left, nodo.comparators[0]
        if isinstance(izquierda, ast.Name) and izquierda.id == variable:
            if isinstance(derecha, (ast.Tuple, ast.List, ast.Set)):
                for elemento in derecha.elts:
                    if isinstance(elemento, ast.Constant) and isinstance(elemento.value, str):
                        valores.add(elemento.value)
    return valores


def test_la_copia_de_la_cara_dice_lo_mismo_que_el_nucleo() -> None:
    """La guardia de la fase: si las dos se separan, el CI se pone rojo.

    Si esto falla, el arreglo no es tocar la prueba:

        python commands/perseo.py catalogo --incrustar
    """
    copia = comprobar.COPIA_DEL_CATALOGO.read_text(encoding="utf-8")
    assert copia == comprobar.texto_de_la_copia(), (
        "la copia incrustada de RealTime no dice lo mismo que el núcleo; "
        "regenérala con: python commands/perseo.py catalogo --incrustar"
    )


def test_la_copia_es_json_de_verdad() -> None:
    """Red de seguridad: comparar dos ficheros iguales por casualidad no vale.

    Si la copia dejara de ser legible, la prueba de arriba seguiría pasando
    mientras las dos estuvieran igual de rotas.
    """
    copia = comprobar.COPIA_DEL_CATALOGO.read_text(encoding="utf-8")
    # Desde el `= [` de la asignación: antes hay un `[]` en el tipo.
    cuerpo = copia[copia.index("= [") + 2 : copia.rindex("]") + 1]
    herramientas = json.loads(cuerpo)
    assert [h["name"] for h in herramientas] == catalogo.nombres("voz")


def test_las_acciones_de_pc_son_las_que_pc_sabe_hacer() -> None:
    """El fallo concreto que abrió todo esto, convertido en regla.

    El `enum` que se le enseña al modelo tiene que ser exactamente lo que el
    agente implementa. Una de más deja un trabajo colgado esperando un sí que
    nadie ve; una de menos es una función que existe y nadie puede pedir.
    """
    herramienta = catalogo.por_nombre("controlar_pc")
    assert herramienta is not None
    (accion,) = [p for p in herramienta.parametros if p.nombre == "accion"]
    hace = despacha(PC_PY, "accion")
    assert set(accion.opciones) == hace, (
        f"el catálogo ofrece {sorted(accion.opciones)} y `pc` hace {sorted(hace)}"
    )


def test_cada_herramienta_esta_en_alguna_cara() -> None:
    """Una herramienta que no ve nadie es código muerto con buena presencia."""
    huerfanas = [h.nombre for h in catalogo.CATALOGO if not h.voz and not h.chat]
    assert not huerfanas, "no las ve ninguna cara: " + ", ".join(huerfanas)


def test_el_chat_escrito_sabe_ejecutar_lo_que_declara() -> None:
    """Declarar una herramienta que el chat no sabe despachar es prometer y no dar.

    Es el mismo fallo que `navegar_url` un escalón más arriba: el modelo la
    pide, nadie la atiende, y lo que llega es un error raro en vez de una
    respuesta.
    """
    atiende = despacha(CHAT_PY, "nombre")
    sin_atender = sorted(set(catalogo.nombres("chat")) - atiende)
    assert not sin_atender, "el chat las declara y no las sabe ejecutar: " + ", ".join(sin_atender)


def test_las_dos_caras_comparten_la_forma_de_lo_que_comparten() -> None:
    """Lo que se cuenta puede cambiar por cara; lo que se manda, no.

    El texto va por cara a propósito —por voz se pide el sí hablando, por
    escrito se pulsa un botón—. Pero el nombre de cada parámetro, su tipo, sus
    opciones y si es obligatorio son lo que el modelo tiene que acertar para que
    la llamada funcione, y de eso hay una sola versión por construcción: los
    `Parametro` no llevan variante por cara.
    """
    for herramienta in catalogo.CATALOGO:
        if not (herramienta.voz and herramienta.chat):
            continue
        for cara in ("voz", "chat"):
            esquema = catalogo.esquema(herramienta, cara)
            otra = catalogo.esquema(herramienta, "chat" if cara == "voz" else "voz")
            assert set(esquema["properties"]) == set(otra["properties"])
            assert esquema.get("required") == otra.get("required")
            for nombre, detalle in esquema["properties"].items():
                assert detalle["type"] == otra["properties"][nombre]["type"]
                assert detalle.get("enum") == otra["properties"][nombre].get("enum")
