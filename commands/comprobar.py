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
        # Lo que no llama nadie. La lista de excepciones, con el porqué de cada
        # una, está en `verificadores/vulture_permitidos.py`.
        (
            "código muerto del núcleo",
            [
                sys.executable,
                "-m",
                "vulture",
                "perseo_core",
                "commands",
                "verificadores",
                "pruebas",
                "verificadores/vulture_permitidos.py",
                "--min-confidence",
                "60",
            ],
            RAIZ,
        ),
    ]
    if npx:
        pasos.append(("tipos del frontend", [npx, "tsc", "--noEmit"], RAIZ / "RealTime"))
    if npm:
        pasos.append(("pruebas del frontend", [npm, "test"], RAIZ / "RealTime"))
        pasos.append(("código muerto de la interfaz", [npm, "run", "muertos"], RAIZ / "RealTime"))
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
    # Sin `-q` aquí: `pytest.ini` ya lo pone, y un segundo `-q` cambia el formato
    # del resumen a una lista por fichero sin total. Costó un «0 pruebas».
    salida = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-p", "no:cacheprovider"],
        cwd=RAIZ,
        capture_output=True,
        text=True,
    )
    # Un fichero que no importa se recoge como error y pytest **sigue contando**
    # el resto: el número saldría bajo y parecería que faltan pruebas. Pasó al
    # escribir esto mismo, y el README se quedó con una cifra de menos.
    if salida.returncode != 0:
        raise RuntimeError(
            "pytest no pudo recoger las pruebas; arregla eso antes de contar:\n"
            + salida.stdout[-1500:]
        )
    for linea in reversed(salida.stdout.splitlines()):
        # La última línea útil es «742 tests collected in 0.55s».
        trozos = linea.split()
        if len(trozos) >= 2 and trozos[1].startswith("test") and trozos[0].isdigit():
            return int(trozos[0])
    raise RuntimeError("pytest no dijo cuántas pruebas recoge:\n" + salida.stdout[-800:])


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
    """Cuántos hay, los dieciséis de `verificadores/` y el que vive en `commands/`.

    El de la palabra clave está allí y no aquí porque necesita la voz de Windows
    y un modelo de audio: no lo puede correr el CI, y arrastrarlo a la carpeta de
    los que sí corren haría pensar que se pasa con los demás.
    """
    return len(list((RAIZ / "verificadores").glob("verificar_*.py"))) + len(
        list((RAIZ / "commands").glob("verificar_*.py"))
    )


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


#: Dónde está escrito a mano cada recuento, y con qué medida tiene que cuadrar.
#: El patrón lleva **un solo grupo**: el número. Con eso se hacen las dos cosas
#: —comprobar que coincide y reescribirlo— sin tener la frase escrita dos veces.
#:
#: La forma exacta de cada frase va aquí a propósito. Si alguien la reescribe, el
#: patrón deja de encontrarla y `test_documentacion.py` falla diciéndolo, en vez
#: de dejar de vigilar en silencio, que es como envejecen estas cosas.
CITAS: tuple[tuple[str, str, str], ...] = (
    ("README.md", r"!\[(\d+) pruebas\]", "pruebas_total"),
    ("README.md", r"badge/pruebas-(\d+)-black", "pruebas_total"),
    ("README.md", r"Dicen \*qué\* se ha roto: \*\*(\d+)\*\* en total", "pruebas_total"),
    ("README.md", r"python -m pytest\s+# (\d+), el núcleo", "pruebas_python"),
    ("README.md", r"npm test\s+# (\d+), la interfaz", "pruebas_frontend"),
    ("README.md", r"líneas, (\d+) pruebas", "pruebas_total"),
    ("README.md", r"pruebas y (\d+) verificadores", "verificadores"),
    ("README.en.md", r"!\[(\d+) tests\]", "pruebas_total"),
    ("README.en.md", r"badge/tests-(\d+)-black", "pruebas_total"),
    ("README.en.md", r"you \*what\* broke: \*\*(\d+)\*\* in total", "pruebas_total"),
    ("README.en.md", r"python -m pytest\s+# (\d+), core", "pruebas_python"),
    ("README.en.md", r"npm test\s+# (\d+), the interface", "pruebas_frontend"),
    ("README.en.md", r"lines, (\d+) tests", "pruebas_total"),
    ("README.en.md", r"tests and (\d+) verifiers", "verificadores"),
)


def revisar_citas(medidas: dict[str, int] | None = None) -> list[str]:
    """Qué cifras escritas a mano no cuadran con la realidad. Vacío es bueno."""
    import re

    medidas = medidas or cuentas()
    errores: list[str] = []
    for fichero, patron, clave in CITAS:
        texto = (RAIZ / fichero).read_text(encoding="utf-8")
        encontrado = re.search(patron, texto)
        if encontrado is None:
            errores.append(f"{fichero}: ya no está la frase que dice «{clave}» ({patron})")
        elif int(encontrado.group(1)) != medidas[clave]:
            errores.append(
                f"{fichero}: dice {encontrado.group(1)} y son {medidas[clave]} ({clave})"
            )
    return errores


def arreglar_citas() -> list[str]:
    """Reescribe las cifras con las medidas. Es el arreglo de `revisar_citas`."""
    import re

    medidas = cuentas()
    cambios: list[str] = []
    for fichero, patron, clave in CITAS:
        ruta = RAIZ / fichero
        texto = ruta.read_text(encoding="utf-8")
        valor = str(medidas[clave])

        def poner(encontrado: "re.Match[str]") -> str:
            return encontrado.group(0).replace(encontrado.group(1), valor, 1)

        nuevo, veces = re.subn(patron, poner, texto)
        if not veces:
            cambios.append(f"{fichero}: no encuentro la frase de «{clave}»; míralo a mano")
        elif nuevo != texto:
            ruta.write_text(nuevo, encoding="utf-8")
            cambios.append(f"{fichero}: {clave} -> {valor}")
    return cambios


def imprimir_cuentas(argumentos: list[str] | None = None) -> int:
    argumentos = argumentos or []
    if "--arreglar" in argumentos:
        cambios = arreglar_citas()
        print("\n".join(cambios) if cambios else "La documentación ya dice lo que hay.")
        return 0

    numeros = cuentas()
    if "--json" in argumentos:
        print(json.dumps(numeros, indent=2))
        return 0
    for clave, valor in numeros.items():
        print(f"{clave:20} {valor}")
    pendientes = revisar_citas(numeros)
    if pendientes:
        print("\nLa documentación no dice esto:")
        for linea in pendientes:
            print("  " + linea)
        print("\n  python commands/perseo.py cuentas --arreglar")
    return 0


# ==========================================================================
# La copia del catálogo que lleva la cara de la voz
# ==========================================================================
#: Dónde vive la copia incrustada. La cara pide el catálogo al núcleo al
#: conectar, pero arranca sin él más veces de las que parece —abrir la app antes
#: de que el núcleo termine de levantarse es lo normal— y una llamada sin
#: herramientas sería peor que una llamada con las de ayer.
COPIA_DEL_CATALOGO = RAIZ / "RealTime" / "src" / "lib" / "llamada" / "catalogo-incrustado.ts"

_CABECERA_COPIA = """\
/**
 * GENERADO. No se edita a mano.
 *
 *     python commands/perseo.py catalogo --incrustar
 *
 * La copia de respaldo del catálogo de herramientas. La fuente es
 * `perseo_core/servicios/catalogo.py`; esto es lo que la llamada usa cuando el
 * núcleo no contesta a tiempo, que pasa cada vez que se abre la app antes de que
 * el núcleo termine de levantarse.
 *
 * Que exista una copia es el precio de no bloquear el socket esperando. Lo que
 * impide que envejezca es `pruebas/test_catalogo.py`, que la compara con el
 * núcleo y pone el CI en rojo si difieren.
 */

import type { HerramientaNeutra } from './catalogo';

export const CATALOGO_INCRUSTADO: HerramientaNeutra[] = """


def _catalogo_del_nucleo(cara: str = "voz") -> list[dict]:
    sys.path.insert(0, str(RAIZ))
    from perseo_core.servicios import catalogo

    return catalogo.para(cara)


def texto_de_la_copia() -> str:
    """El contenido exacto que debe tener el fichero incrustado."""
    cuerpo = json.dumps(_catalogo_del_nucleo(), ensure_ascii=False, indent=2)
    return _CABECERA_COPIA + cuerpo + ";\n"


def incrustar_catalogo() -> bool:
    """Reescribe la copia. Devuelve si hizo falta cambiarla."""
    nuevo = texto_de_la_copia()
    viejo = COPIA_DEL_CATALOGO.read_text(encoding="utf-8") if COPIA_DEL_CATALOGO.exists() else ""
    if nuevo == viejo:
        return False
    COPIA_DEL_CATALOGO.write_text(nuevo, encoding="utf-8")
    return True


def catalogo_cli(argumentos: list[str] | None = None) -> int:
    argumentos = argumentos or []
    if "--incrustar" in argumentos:
        cambio = incrustar_catalogo()
        print("Copia regenerada." if cambio else "La copia ya decía lo mismo que el núcleo.")
        return 0
    for cara in ("voz", "chat"):
        herramientas = _catalogo_del_nucleo(cara)
        print(f"\n{cara} ({len(herramientas)})")
        for h in herramientas:
            obligatorios = h["parameters"].get("required") or []
            argumentos_ = ", ".join(
                n + ("*" if n in obligatorios else "")
                for n in (h["parameters"].get("properties") or {})
            )
            print(f"  {h['name']}({argumentos_})")
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

    print("\nLa frontera de la cara")
    con_puerta = arquitectura.componentes_con_puerta_propia()
    perdonados = arquitectura.COMPONENTES_QUE_LLAMAN_AL_NUCLEO
    nuevos = sorted(con_puerta - perdonados)
    print(
        f"  componentes que llaman al núcleo: {len(con_puerta)}"
        f"   (perdonados: {len(perdonados)})"
    )
    for nombre in sorted(con_puerta):
        print(f"    {nombre}" + ("  <- NUEVO" if nombre in nuevos else ""))

    print("\nEl repaso, en una línea: nada NUEVO arriba, y las listas más cortas")
    print("que la última vez. Lo que no se pueda encoger, que tenga su página en")
    print("docs/adr/ diciendo por qué.")


if __name__ == "__main__":
    raise SystemExit(comprobar(sys.argv[1:]))
