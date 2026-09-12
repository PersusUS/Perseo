"""`perseo comprobar` — todo lo que tiene que estar verde, en un solo sitio.

Antes de esto, «lo que hay que pasar antes de dar algo por bueno» estaba escrito
tres veces: en `AGENTS.md`, en el README y en el fichero del CI. Tres copias que
ya habían empezado a divergir —los números de pruebas no coincidían en ninguna—
y esa es exactamente la clase de fallo que este módulo existe para que no vuelva:
**una fuente, y las demás que la citen**.

    perseo comprobar                  las cinco comprobaciones, en orden de coste
    perseo comprobar --rapido         sin `cargo check`, que es la que tarda
    perseo comprobar --arquitectura   el cuadro de mandos de la estructura
    perseo cuentas                    los números que salen en la documentación

El orden no es alfabético: primero lo que tarda segundos, al final lo que tarda
minutos. Así un fallo tonto se ve enseguida y no después de esperar a Rust.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent

sys.path.insert(0, str(AQUI))

import arquitectura  # noqa: E402


def _programa(nombre: str) -> str | None:
    """Dónde está un ejecutable, o `None`. En Windows esto encuentra `npm.cmd`."""
    return shutil.which(nombre)


def _pasos(rapido: bool) -> list[tuple[str, list[str], Path]]:
    """Qué se ejecuta, en orden de coste creciente.

    Un paso cuyo programa no esté instalado se salta diciéndolo, en vez de
    fallar: quien toca solo el núcleo no tiene por qué tener Rust puesto.
    """
    npm = _programa("npm")
    npx = _programa("npx")
    cargo = _programa("cargo")

    pasos: list[tuple[str, list[str], Path]] = [
        ("pruebas del núcleo", [sys.executable, "-m", "pytest"], RAIZ),
        ("estilo", [sys.executable, "-m", "ruff", "check", "."], RAIZ),
    ]
    if npx:
        pasos.append(("tipos del frontend", [npx, "tsc", "--noEmit"], RAIZ / "RealTime"))
    if npm:
        pasos.append(("pruebas del frontend", [npm, "test"], RAIZ / "RealTime"))
    if cargo and not rapido:
        pasos.append(
            ("rust", [cargo, "check", "--locked"], RAIZ / "RealTime" / "src-tauri")
        )
    return pasos


def comprobar(argumentos: list[str] | None = None) -> int:
    """Las comprobaciones de siempre. Devuelve el código de salida."""
    argumentos = argumentos or []
    if "--arquitectura" in argumentos:
        cuadro_de_mandos()
        return 0

    rapido = "--rapido" in argumentos
    fallos: list[str] = []
    for nombre, orden, donde in _pasos(rapido):
        print(f"\n=== {nombre} ===", flush=True)
        codigo = subprocess.run(orden, cwd=donde).returncode
        if codigo != 0:
            fallos.append(nombre)

    print()
    if fallos:
        print("FALLA: " + ", ".join(fallos))
        return 1
    print("Todo verde." + (" (sin Rust: --rapido)" if rapido else ""))
    return 0


# ==========================================================================
# Los números que la documentación cita
# ==========================================================================
def _pruebas_de_python() -> int:
    """Cuántas recoge pytest, con las parametrizadas ya expandidas.

    Se lo pregunta a pytest en vez de contar `def test_` a mano porque una
    parametrizada son varias pruebas, y ese es el número que sale en el README.
    """
    salida = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=RAIZ,
        capture_output=True,
        text=True,
    )
    for linea in reversed(salida.stdout.splitlines()):
        # La última línea útil es «742 tests collected in 0.55s».
        trozos = linea.split()
        if len(trozos) >= 2 and trozos[1].startswith("test") and trozos[0].isdigit():
            return int(trozos[0])
    return 0


def _pruebas_del_frontend() -> int:
    """Cuántas declara vitest, contadas leyendo los ficheros.

    Contarlas en vez de ejecutarlas tiene un motivo: esto lo llama una prueba de
    pytest, y arrancar Node desde dentro de pytest ataría el CI de Python a que
    hubiera `npm` instalado. La cuenta coincide porque en este repositorio no hay
    `it.each` ni pruebas generadas en un bucle; si algún día las hay, esto se
    queda corto y la prueba que lo compara lo dirá en voz alta.
    """
    import re

    patron = re.compile(r"^\s*(it|test)(\.\w+)?\(", re.MULTILINE)
    total = 0
    for ruta in sorted((RAIZ / "RealTime" / "pruebas").glob("*.test.ts")):
        total += len(patron.findall(ruta.read_text(encoding="utf-8-sig")))
    return total


def _verificadores() -> int:
    carpeta = RAIZ / "verificadores"
    if carpeta.exists():
        return len(list(carpeta.glob("verificar_*.py")))
    return len(list((RAIZ / "perseo_core").glob("verificar_*.py")))


def cuentas() -> dict[str, int]:
    """Los números que la documentación cita, medidos aquí y no a mano.

    Cuatro sitios con el recuento escrito a mano son cuatro sitios que
    envejecen: el 2026-09-12 el README decía 880, `AGENTS.md` decía 724 y la
    bitácora 617, y las tres cifras habían sido verdad alguna vez.
    """
    python = _pruebas_de_python()
    frontend = _pruebas_del_frontend()
    return {
        "pruebas_python": python,
        "pruebas_frontend": frontend,
        "pruebas_total": python + frontend,
        "verificadores": _verificadores(),
    }


def imprimir_cuentas(argumentos: list[str] | None = None) -> int:
    numeros = cuentas()
    if argumentos and "--json" in argumentos:
        print(json.dumps(numeros, indent=2))
        return 0
    for clave, valor in numeros.items():
        print(f"{clave:20} {valor}")
    return 0


# ==========================================================================
# El cuadro de mandos de la estructura
# ==========================================================================
def cuadro_de_mandos() -> None:
    """Lo que hay que mirar en el repaso trimestral, de un vistazo.

    Ciclos, techos, excepciones vivas. No falla nunca: lo que falla es
    `pruebas/test_arquitectura.py`. Esto es para leer, no para vigilar.
    """
    grafo = arquitectura.grafo_del_nucleo()
    vivos = arquitectura.ciclos(grafo)
    saltos = arquitectura.saltos_de_capa(grafo)
    pasados = arquitectura.pasan_del_techo()
    blandos = arquitectura.pasan_del_techo(duro=False)

    print(f"Módulos del núcleo: {len(grafo)}  ·  importes: {sum(len(v) for v in grafo.values())}")

    print("\nCapas")
    por_capa: dict[str, int] = {}
    for modulo in grafo:
        por_capa[arquitectura.capa(modulo) or "(sin capa)"] = (
            por_capa.get(arquitectura.capa(modulo) or "(sin capa)", 0) + 1
        )
    for nombre in (*arquitectura.CAPAS, "(sin capa)"):
        if nombre in por_capa:
            print(f"  {nombre:12} {por_capa[nombre]:3} módulos")
    print(f"  saltos hacia arriba: {len(saltos)}")
    for quien, que in saltos:
        print(f"    {quien} -> {que}")

    print("\nCiclos")
    if not vivos:
        print("  ninguno")
    for ciclo in vivos:
        conocido = tuple(ciclo) in {tuple(sorted(c)) for c in arquitectura.CICLOS_CONOCIDOS}
        print(f"  {' -> '.join(ciclo)}{'  (conocido)' if conocido else '  <- NUEVO'}")

    print(
        f"\nTamaño  ·  techo blando {arquitectura.TECHO_BLANDO}"
        f", duro {arquitectura.TECHO_DURO}"
    )
    print(f"  pasan del blando: {len(blandos)}   pasan del duro: {len(pasados)}")
    for nombre, cuenta in sorted(pasados.items(), key=lambda p: -p[1]):
        permitido = arquitectura.EXCEPCIONES_DE_TAMANO.get(nombre)
        marca = "excepción" if permitido is not None else "<- SIN PERMISO"
        holgura = f" (apuntado {permitido})" if permitido not in (None, cuenta) else ""
        print(f"  {cuenta:5}  {nombre}  {marca}{holgura}")

    muertas = arquitectura.excepciones_muertas()
    if muertas:
        print("\nExcepciones que ya sobran: " + ", ".join(muertas))


if __name__ == "__main__":
    raise SystemExit(comprobar(sys.argv[1:]))
