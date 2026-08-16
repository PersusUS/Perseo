"""`perseo` — encender Perseo entero desde una terminal.

Perseo son tres procesos y hasta ahora había que saberse los tres:

    pythonw commands/vigilante.py        el núcleo, con quien lo revive
    pythonw commands/clap_detector.py    el detector de aplausos
    RealTime\\...\\perseo.exe              la app de voz

Arrancan solos al iniciar sesión en Windows (`manage_startup.py install`), así
que este comando es para el resto de los casos: después de matar algo, después de
un `git pull`, o cuando quieres mirar si está todo en pie sin acordarte de las
tres rutas.

    perseo            enciende lo que falte y abre la app
    perseo estado     dice qué hay vivo, sin tocar nada
    perseo nucleo     solo el núcleo
    perseo parar      cierra la app y el núcleo (el vigilante incluido)

**Nada de esto arranca lo que ya está corriendo.** Se comprueba antes: dos
núcleos peleándose por el puerto 8787 es un `OSError 10048` que no se parece en
nada al problema real, y dos detectores escuchando el mismo micrófono responden
a la vez.

Dos dependencias que este comando **no** levanta, y que conviene tener abiertas:
Ollama —sin él no hay triaje— y Obsidian, sin el cual la memoria falla si el
vault va por el plugin. `perseo estado` las mira y lo dice.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent

sys.path.insert(0, str(AQUI))

import manage_startup  # noqa: E402
import presencia  # noqa: E402

#: Dónde puede estar la app construida, de la más buena a la menos.
#: `release` es la de verdad; `debug` la deja `npm run tauri dev` y sirve igual
#: para abrirla, solo que arranca más despacio.
APPS = (
    RAIZ / "RealTime" / "src-tauri" / "target" / "release" / "perseo.exe",
    RAIZ / "RealTime" / "src-tauri" / "target" / "debug" / "perseo.exe",
)


def _pythonw() -> str:
    """El intérprete sin consola. Con `python.exe` a secas, cada proceso abriría
    una ventana negra que además lo mata al cerrarla — que es exactamente cómo se
    murió el vigilante el 2026-08-16."""
    ejecutable = sys.executable
    candidato = ejecutable.replace("python.exe", "pythonw.exe")
    return candidato if candidato != ejecutable and os.path.isfile(candidato) else ejecutable


def _sin_consola(argumentos: list[str]) -> None:
    """Lanza algo y se desentiende: ni consola, ni esperar, ni morir con esta."""
    banderas = 0
    if os.name == "nt":
        banderas = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
    subprocess.Popen(argumentos, cwd=str(RAIZ), creationflags=banderas, close_fds=True)


def _corriendo(fragmento: str) -> bool:
    """Si hay algún proceso de Python cuya línea de comandos lleve eso dentro.

    Se pregunta a Windows por WMI en vez de mirar un fichero de PID porque el
    detector no deja ninguno. Si la consulta falla, se contesta que **no** está:
    el peor caso es arrancar algo dos veces, y eso se nota; dar por vivo algo
    muerto deja a Perseo apagado sin que nadie lo sepa.
    """
    if os.name != "nt":
        return False
    try:
        salida = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" "
                "| Select-Object -ExpandProperty CommandLine",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return any(fragmento in linea for linea in salida.splitlines())


def _ollama() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _obsidian() -> bool:
    """Si el plugin de Obsidian contesta. Sin clave da 401, que también vale:
    lo que se quiere saber es si Obsidian está abierto."""
    import ssl

    contexto = ssl.create_default_context()
    contexto.check_hostname = False
    contexto.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(
            "https://127.0.0.1:27124/", timeout=3, context=contexto
        ) as r:
            return r.status in (200, 401)
    except (urllib.error.URLError, OSError):
        return False


def _app() -> Path | None:
    for ruta in APPS:
        if ruta.is_file():
            return ruta
    return None


# --------------------------------------------------------------------------- #
# Lo que hace cada orden
# --------------------------------------------------------------------------- #


def arrancar_nucleo() -> bool:
    """Levanta el vigilante, que a su vez levanta el núcleo. Devuelve si hizo algo."""
    if manage_startup.nucleo_responde():
        print("  [ya estaba]  El núcleo responde en", manage_startup.url_salud())
        return False

    print("  [arrancando] El núcleo, con su vigilante")
    _sin_consola([_pythonw(), str(AQUI / "vigilante.py")])

    # El vigilante espera un momento antes del primer intento, así que no se
    # pregunta de inmediato: se pregunta hasta que conteste o se acabe la
    # paciencia. Diez segundos son de sobra en frío.
    for _ in range(20):
        time.sleep(0.5)
        if manage_startup.nucleo_responde(espera=1.0):
            print("  [listo]      El núcleo responde")
            return True
    print("  [ojo]        El núcleo no ha contestado todavía; mira <datos>/vigilante.log")
    return True


def arrancar_detector() -> bool:
    if _corriendo("clap_detector.py"):
        print("  [ya estaba]  El detector de aplausos")
        return False
    print("  [arrancando] El detector de aplausos")
    _sin_consola([_pythonw(), str(AQUI / "clap_detector.py")])
    return True


def arrancar_app() -> bool:
    if presencia.app_viva():
        print("  [ya estaba]  La app de voz")
        return False

    binario = _app()
    if binario is None:
        print("  [falta]      La app no está construida. Constrúyela una vez:")
        print("               cd RealTime && npm install && npm run tauri build")
        return False

    print(f"  [arrancando] La app de voz ({binario.parent.name})")
    _sin_consola([str(binario)])
    return True


def estado() -> None:
    manage_startup.estado()
    print()
    print("Lo que no arranca solo:")
    for nombre, vivo, sin_el in (
        ("Ollama", _ollama(), "sin él no hay triaje de correo"),
        ("Obsidian", _obsidian(), "sin él la memoria falla si el vault va por el plugin"),
    ):
        marca = "[activo]  " if vivo else "[PARADO]  "
        print(f"  {marca}   {nombre} — {sin_el}")

    print()
    detector = "[activo]  " if _corriendo("clap_detector.py") else "[PARADO]  "
    app = "[activo]  " if presencia.app_viva() else "[PARADO]  "
    print(f"  {detector}   Detector de aplausos")
    print(f"  {app}   App de voz")


def parar() -> None:
    """Cierra la app y el núcleo. **Al vigilante primero**, o resucita el núcleo.

    No toca el detector de aplausos: es lo que despierta a Perseo, y pararlo sin
    querer deja el sistema mudo de una forma que no se nota hasta que aplaudes.
    """
    if os.name != "nt":
        print("Esto solo sabe parar procesos en Windows.")
        return

    for descripcion, patron in (
        ("el vigilante", "vigilante.py"),
        ("el núcleo", "-m perseo_core"),
    ):
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" "
                f"| Where-Object {{ $_.CommandLine -like '*{patron}*' }} "
                "| ForEach-Object { Stop-Process -Id $_.ProcessId -Force }",
            ],
            capture_output=True,
        )
        print(f"  [parado]     {descripcion}")

    subprocess.run(
        ["powershell", "-NoProfile", "-Command", "Stop-Process -Name perseo -Force -ErrorAction SilentlyContinue"],
        capture_output=True,
    )
    print("  [parado]     La app de voz")
    print("\n  El detector de aplausos sigue en pie: es lo que despierta a Perseo.")


def todo() -> None:
    print("Encendiendo Perseo:\n")
    arrancar_nucleo()
    arrancar_detector()
    arrancar_app()

    faltan = [n for n, vivo in (("Ollama", _ollama()), ("Obsidian", _obsidian())) if not vivo]
    if faltan:
        print(f"\n  Ojo: {' y '.join(faltan)} sin arrancar. `perseo estado` cuenta por qué importa.")
    print("\n  Panel y cola: botón de cuadrícula en la app, o http://127.0.0.1:8787")


ORDENES = {
    "": todo,
    "estado": estado,
    "nucleo": lambda: arrancar_nucleo(),
    "núcleo": lambda: arrancar_nucleo(),
    "parar": parar,
}


def main() -> None:
    orden = (sys.argv[1] if len(sys.argv) > 1 else "").strip().lower()
    if orden in ("-h", "--help", "ayuda"):
        print(__doc__)
        return
    hacer = ORDENES.get(orden)
    if hacer is None:
        print(f"No sé qué es «{orden}». Órdenes: {', '.join(o for o in ORDENES if o)}")
        sys.exit(2)
    hacer()


if __name__ == "__main__":
    main()
