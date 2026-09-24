"""`perseo boveda` y `perseo navegador`: lo que los recados necesitan de él.

Las dos cosas que un recado no puede hacer solo, y que por eso se hacen aquí,
en una terminal y no por el chat ni por la voz —lo que se dice o se escribe ahí
pasa por un modelo en la nube, y es justo lo que la bóveda existe para evitar—:

    perseo boveda                         lo guardado, sin un solo valor
    perseo boveda guardar resy --sitio resy.com
    perseo boveda guardar visa --tarjeta --sitio resy.com --tope 60
    perseo boveda borrar resy
    perseo navegador                      abre el navegador de los recados

Los valores se piden con `getpass`, sin eco: no quedan en el historial de la
terminal ni en la pantalla.

**`perseo navegador`** abre Chrome con el perfil de los recados
(`<datos>/navegador`) para que entres a mano en tus sitios —con su código de
verificación, si lo piden— una sola vez. Las sesiones se quedan en el perfil y
los recados las usan después. Es lo que Instinct llama «un ordenador con tus
sesiones», con la diferencia de que el ordenador es el tuyo.
"""

from __future__ import annotations

import argparse
import getpass
import os
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))


def _directorio_datos() -> Path:
    return Path(os.environ.get("PERSEO_CORE_DATOS", RAIZ / "perseo_core" / "datos"))


def _pedir(etiqueta: str, secreto: bool) -> str:
    valor = getpass.getpass(f"  {etiqueta}: ") if secreto else input(f"  {etiqueta}: ")
    return valor.strip()


def _guardar(args: argparse.Namespace) -> int:
    from perseo_core.servicios import boveda

    caja = boveda.Boveda(_directorio_datos() / boveda.NOMBRE_FICHERO)
    if args.tarjeta:
        print(f"Tarjeta «{args.nombre}». Deja en blanco lo que no quieras guardar.")
        campos = {
            "titular": _pedir("titular", False),
            "numero": _pedir("número (no se ve al escribir)", True).replace(" ", ""),
            "caducidad": _pedir("caducidad (MM/AA)", False),
            "cvc": _pedir("CVC (no se ve al escribir)", True),
        }
    else:
        print(f"Cuenta «{args.nombre}».")
        campos = {
            "usuario": _pedir("usuario o correo", False),
            "clave": _pedir("contraseña (no se ve al escribir)", True),
        }
    campos = {campo: valor for campo, valor in campos.items() if valor}
    try:
        caja.guardar(
            args.nombre, "tarjeta" if args.tarjeta else "cuenta", args.sitio or [], campos, args.tope
        )
    except boveda.ErrorBoveda as e:
        print(f"No se guardó: {e}")
        return 1
    print(f"Guardada. El modelo la verá como {{{{boveda:{args.nombre.lower()}.<campo>}}}} y solo en: {', '.join(args.sitio)}.")
    return 0


def _listar() -> int:
    from perseo_core.servicios import boveda

    try:
        entradas = boveda.Boveda(_directorio_datos() / boveda.NOMBRE_FICHERO).listar()
    except boveda.ErrorBoveda as e:
        print(e)
        return 1
    if not entradas:
        print("La bóveda está vacía. `perseo boveda guardar <nombre> --sitio <dominio>` para empezar.")
        return 0
    for e in entradas:
        tope = f", tope {e['limite_euros']:g} €" if e.get("limite_euros") is not None else ""
        print(f"  {e['nombre']:<16} {e['tipo']:<8} en {', '.join(e['sitios'])}{tope}  ·  campos: {', '.join(e['campos'])}")
    return 0


def _borrar(args: argparse.Namespace) -> int:
    from perseo_core.servicios import boveda

    hecho = boveda.Boveda(_directorio_datos() / boveda.NOMBRE_FICHERO).borrar(args.nombre)
    print("Borrada." if hecho else f"No había ninguna entrada «{args.nombre}».")
    return 0 if hecho else 1


def boveda_cli(argumentos: list[str]) -> int:
    lector = argparse.ArgumentParser(prog="perseo boveda", description="Contraseñas y tarjetas de los recados.")
    ordenes = lector.add_subparsers(dest="orden")
    guardar = ordenes.add_parser("guardar", help="guarda o reemplaza una entrada")
    guardar.add_argument("nombre")
    guardar.add_argument("--sitio", action="append", required=True, help="dominio donde vale; se puede repetir")
    guardar.add_argument("--tarjeta", action="store_true", help="es una tarjeta, no una cuenta")
    guardar.add_argument("--tope", type=float, default=None, help="importe máximo que se deja pulsar con ella")
    borrar = ordenes.add_parser("borrar", help="quita una entrada")
    borrar.add_argument("nombre")
    args = lector.parse_args(argumentos)
    if args.orden == "guardar":
        return _guardar(args)
    if args.orden == "borrar":
        return _borrar(args)
    return _listar()


# --------------------------------------------------------------------------- #
# El navegador de los recados, a mano
# --------------------------------------------------------------------------- #

_CHROME = (
    Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Google/Chrome/Application/chrome.exe",
    Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / "Google/Chrome/Application/chrome.exe",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
)


def _perfil_en_uso(perfil: Path) -> bool:
    """Si algún proceso tiene ya abierto el perfil: un recado en marcha, o esperando un sí."""
    try:
        import psutil
    except ImportError:
        return False
    marca = str(perfil).lower()
    for proceso in psutil.process_iter(["cmdline"]):
        try:
            linea = " ".join(proceso.info.get("cmdline") or []).lower()
        except (psutil.Error, TypeError):
            continue
        if marca in linea:
            return True
    return False


def navegador_cli(argumentos: list[str]) -> int:
    perfil = _directorio_datos() / "navegador"
    chrome = next((ruta for ruta in _CHROME if ruta.is_file()), None)
    if chrome is None:
        print("No encuentro Chrome. Los recados usan Google Chrome: instálalo y vuelve a probar.")
        return 1
    if _perfil_en_uso(perfil):
        # Chrome no abre un perfil dos veces: la ventana nueva iría a parar al
        # proceso del recado, que no tiene pantalla, y no se vería nada.
        print(
            "El navegador de los recados está en uso: hay un recado en marcha o esperando un sí. "
            "Espera a que acabe (o recházalo) y vuelve a probar."
        )
        return 1
    perfil.mkdir(parents=True, exist_ok=True)
    url = argumentos[0] if argumentos else "about:blank"
    subprocess.Popen([str(chrome), f"--user-data-dir={perfil}", "--no-first-run", url])
    print(
        "Abierto el navegador de los recados. Entra en tus sitios y ciérralo al acabar: "
        "las sesiones se quedan para los recados."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(boveda_cli(sys.argv[1:]))
