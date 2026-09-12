"""La forma del repositorio, medida en vez de recordada.

Este módulo no hace nada en tiempo de ejecución: lo leen
`pruebas/test_arquitectura.py` —que convierte cada medida en una prueba que se
pone roja— y `perseo.py comprobar --arquitectura`, que imprime el cuadro de
mandos.

Existe por una razón concreta. El 2026-09-12 se arreglaron a mano una docena de
fallos que no eran de programación sino de estructura: dos sitios que decían
cosas distintas sobre lo mismo, código que no llamaba nadie, documentación que
describía una API inexistente, un ciclo de importaciones esquivado con dos
importes metidos dentro de una función. Una carpeta bien puesta se deshace en
tres meses; una prueba que falla en el CI, no.

Las reglas que sostiene:

  · **Capas.** `perseo_core/` va de abajo arriba —dominio, infra, servicios,
    agentes, caras— y nadie importa hacia arriba. Así el ciclo no vuelve.
  · **Techo.** Un fichero de mil quinientas líneas es donde se esconde lo
    duplicado. Blando a 600 (aviso), duro a 900 (rojo), con una lista de
    excepciones que **solo puede encoger**.
"""

from __future__ import annotations

import ast
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
NUCLEO = RAIZ / "perseo_core"

# ==========================================================================
# Capas
# ==========================================================================
# De abajo arriba: cada una puede importar de las anteriores y de ninguna
# posterior. El orden ES la regla; cambiarlo cambia lo que se permite.
CAPAS: tuple[str, ...] = ("dominio", "infra", "servicios", "agentes", "caras")

# Lo que aún vive en la raíz del paquete no tiene capa asignada, y mientras la
# tenga no se le puede exigir nada. Cuando esta situación se vacíe, la prueba de
# capas pasará a cubrir el paquete entero sin tocar una línea.
SIN_CAPA = ""

# Vacía, y así se queda. El único ciclo que hubo —`google_api` importaba `Evento`
# de `agenda` y `Mensaje` de `correo`, y esos dos importaban `google_api` dentro
# de una función para que Python no se quejara al arrancar— murió al bajar los
# dos tipos a `dominio/`. Si algo vuelve a aparecer aquí, es que se ha vuelto a
# pagar un ciclo con un truco; igual que la lista de tamaños, **solo puede
# encoger**.
CICLOS_CONOCIDOS: tuple[tuple[str, ...], ...] = ()


# ==========================================================================
# Techo de tamaño
# ==========================================================================
TECHO_BLANDO = 600
TECHO_DURO = 900

# Los que hoy pasan del techo duro, con el tamaño que tenían cuando se escribió
# la regla. La prueba comprueba dos cosas: que no aparece ninguno nuevo, y que
# ninguno de estos **crece**. La lista solo puede encoger.
EXCEPCIONES_DE_TAMANO: dict[str, int] = {
    "perseo_core/agentes/dev.py": 1543,
    "RealTime/src/components/Panel.tsx": 1479,
    "RealTime/src/lib/gemini-live.ts": 1443,
    "perseo_core/agentes/chat.py": 1308,
    "perseo_core/infra/almacen.py": 1192,
    "RealTime/src/App.tsx": 1162,
    "RealTime/src/components/Habitos.tsx": 1056,
    "perseo_core/servicios/mcp.py": 1053,
    "perseo_core/caras/api.py": 988,
}

# Dónde se mide. La bitácora, el vault y lo que no escribimos se quedan fuera.
CARPETAS_MEDIDAS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("perseo_core", (".py",)),
    ("commands", (".py",)),
    ("pruebas", (".py",)),
    ("verificadores", (".py",)),
    ("RealTime/src", (".ts", ".tsx")),
)

_IGNORADAS = ("__pycache__", "node_modules", "target", "modelos")


def ficheros_medidos() -> list[Path]:
    """Todo el código propio, en rutas absolutas y en orden estable."""
    encontrados: list[Path] = []
    for carpeta, extensiones in CARPETAS_MEDIDAS:
        base = RAIZ / carpeta
        if not base.exists():
            continue
        for ruta in sorted(base.rglob("*")):
            if ruta.suffix not in extensiones or not ruta.is_file():
                continue
            if any(parte in _IGNORADAS for parte in ruta.parts):
                continue
            encontrados.append(ruta)
    return encontrados


def relativo(ruta: Path) -> str:
    """La ruta como se escribe en las excepciones: con barras, desde la raíz."""
    return ruta.relative_to(RAIZ).as_posix()


def lineas(ruta: Path) -> int:
    return len(ruta.read_text(encoding="utf-8-sig", errors="replace").splitlines())


def tamanos() -> dict[str, int]:
    return {relativo(r): lineas(r) for r in ficheros_medidos()}


def pasan_del_techo(duro: bool = True) -> dict[str, int]:
    """Los que pasan del techo, sea el duro o el blando."""
    limite = TECHO_DURO if duro else TECHO_BLANDO
    return {n: c for n, c in sorted(tamanos().items()) if c > limite}


def techo_roto() -> dict[str, int]:
    """Los que pasan del techo duro **y no tienen permiso** para hacerlo.

    Un fichero de la lista de excepciones que ha crecido cuenta como roto: la
    excepción era para el tamaño de entonces, no un cheque en blanco.
    """
    rotos: dict[str, int] = {}
    for nombre, cuenta in pasan_del_techo().items():
        permitido = EXCEPCIONES_DE_TAMANO.get(nombre)
        if permitido is None or cuenta > permitido:
            rotos[nombre] = cuenta
    return rotos


def excepciones_muertas() -> list[str]:
    """Excepciones de ficheros que ya se partieron, o que ya no existen.

    Sobran, y una lista con basura dentro deja de leerse.
    """
    actuales = tamanos()
    return sorted(
        nombre for nombre in EXCEPCIONES_DE_TAMANO if actuales.get(nombre, 0) <= TECHO_DURO
    )


# ==========================================================================
# Grafo de importaciones del núcleo
# ==========================================================================
def _modulo_de(ruta: Path) -> str:
    """`perseo_core/agentes/correo.py` → `agentes.correo`; `x/__init__.py` → `x`."""
    partes = list(ruta.relative_to(NUCLEO).parts)
    if partes[-1] == "__init__.py":
        partes.pop()
    else:
        partes[-1] = partes[-1][: -len(".py")]
    return ".".join(partes)


def _resolver(modulo: str, nivel: int, destino: str | None) -> list[str] | None:
    """A qué apunta un `from . import x` visto desde `modulo`, en partes.

    `nivel` es el número de puntos y `destino` lo que va detrás, que puede no
    haber (`from . import almacen`, donde el nombre viaja en los alias). Devolver
    la lista vacía **no** es lo mismo que devolver `None`: la lista vacía es la
    raíz del paquete —el caso de casi todo el núcleo hoy— y `None` es un importe
    que se sale de él.
    """
    partes = modulo.split(".")
    # Un `from .` desde `agentes.correo` sube a `agentes`; desde `correo`, a la
    # raíz del paquete. Cada punto de más sube un escalón.
    if nivel > len(partes):
        return None
    base = partes[: len(partes) - nivel]
    if destino:
        base = base + destino.split(".")
    return base


def _modulos_del_nucleo() -> dict[str, Path]:
    modulos: dict[str, Path] = {}
    for ruta in sorted(NUCLEO.rglob("*.py")):
        if any(parte in _IGNORADAS for parte in ruta.parts):
            continue
        modulos[_modulo_de(ruta)] = ruta
    return modulos


def grafo_del_nucleo() -> dict[str, set[str]]:
    """Quién importa a quién dentro de `perseo_core/`, leído sin ejecutar nada.

    Los importes **dentro de una función** cuentan igual que los de arriba: son
    el truco con el que se esquiva un ciclo, y esconderlos del grafo sería
    esconder justo lo que interesa medir.
    """
    modulos = _modulos_del_nucleo()
    grafo: dict[str, set[str]] = {}
    for modulo, ruta in modulos.items():
        arbol = ast.parse(ruta.read_text(encoding="utf-8-sig", errors="replace"), str(ruta))
        vecinos: set[str] = set()
        for nodo in ast.walk(arbol):
            candidatos: list[str] = []
            if isinstance(nodo, ast.ImportFrom) and nodo.level:
                base = _resolver(modulo, nodo.level, nodo.module)
                if base is not None:
                    if base:
                        candidatos.append(".".join(base))
                    # `from . import a, b` no pone nada en `nodo.module`.
                    candidatos += [".".join(base + [a.name]) for a in nodo.names]
            elif isinstance(nodo, ast.ImportFrom) and (nodo.module or "").startswith(
                "perseo_core"
            ):
                resto = (nodo.module or "").removeprefix("perseo_core").lstrip(".")
                if resto:
                    candidatos.append(resto)
                candidatos += [".".join(p for p in (resto, a.name) if p) for a in nodo.names]
            for candidato in candidatos:
                # Solo cuenta lo que es un módulo de verdad: `from .politica import
                # LIBRE` apunta a `politica`, no a `politica.LIBRE`.
                if candidato in modulos and candidato != modulo:
                    vecinos.add(candidato)
        grafo[modulo] = vecinos
    return grafo


def ciclos(grafo: dict[str, set[str]]) -> list[list[str]]:
    """Los grupos de módulos que se importan en círculo (Tarjan, iterativo).

    Iterativo y no recursivo a propósito: con cincuenta módulos la recursión
    sobra, pero el día que sean quinientos la pila no perdona.
    """
    indice: dict[str, int] = {}
    bajo: dict[str, int] = {}
    pila: list[str] = []
    en_pila: set[str] = set()
    contador = 0
    grupos: list[list[str]] = []

    for raiz in sorted(grafo):
        if raiz in indice:
            continue
        trabajo: list[tuple[str, list[str]]] = [(raiz, sorted(grafo.get(raiz, ())))]
        indice[raiz] = bajo[raiz] = contador
        contador += 1
        pila.append(raiz)
        en_pila.add(raiz)
        while trabajo:
            nodo, pendientes = trabajo[-1]
            if pendientes:
                vecino = pendientes.pop(0)
                if vecino not in indice:
                    indice[vecino] = bajo[vecino] = contador
                    contador += 1
                    pila.append(vecino)
                    en_pila.add(vecino)
                    trabajo.append((vecino, sorted(grafo.get(vecino, ()))))
                elif vecino in en_pila:
                    bajo[nodo] = min(bajo[nodo], indice[vecino])
                continue
            trabajo.pop()
            if trabajo:
                padre = trabajo[-1][0]
                bajo[padre] = min(bajo[padre], bajo[nodo])
            if bajo[nodo] == indice[nodo]:
                grupo: list[str] = []
                while True:
                    otro = pila.pop()
                    en_pila.discard(otro)
                    grupo.append(otro)
                    if otro == nodo:
                        break
                # Un módulo solo es un ciclo únicamente si se importa a sí mismo.
                if len(grupo) > 1 or nodo in grafo.get(nodo, ()):
                    grupos.append(sorted(grupo))
    return sorted(grupos)


def ciclos_nuevos(grafo: dict[str, set[str]]) -> list[list[str]]:
    """Los ciclos que no estaban el día que se escribió la regla."""
    conocidos = {tuple(sorted(c)) for c in CICLOS_CONOCIDOS}
    return [c for c in ciclos(grafo) if tuple(c) not in conocidos]


def ciclos_ya_deshechos(grafo: dict[str, set[str]]) -> list[tuple[str, ...]]:
    """Ciclos apuntados como conocidos que ya no existen: sobran de la lista."""
    vivos = {tuple(c) for c in ciclos(grafo)}
    return sorted(c for c in CICLOS_CONOCIDOS if tuple(sorted(c)) not in vivos)


def capa(modulo: str) -> str:
    """La capa de un módulo, o `SIN_CAPA` si todavía vive en la raíz."""
    primera = modulo.split(".")[0]
    return primera if primera in CAPAS else SIN_CAPA


def saltos_de_capa(grafo: dict[str, set[str]]) -> list[tuple[str, str]]:
    """Importes que van hacia arriba: `(quien_importa, lo_importado)`.

    Los módulos sin capa no se juzgan —están de paso— y la capa de más arriba
    puede con todo lo de abajo, que es justo lo que significa estar arriba.
    """
    salto: list[tuple[str, str]] = []
    for modulo, vecinos in sorted(grafo.items()):
        origen = capa(modulo)
        if origen == SIN_CAPA:
            continue
        altura = CAPAS.index(origen)
        for vecino in sorted(vecinos):
            destino = capa(vecino)
            if destino == SIN_CAPA:
                continue
            if CAPAS.index(destino) > altura:
                salto.append((modulo, vecino))
    return salto
