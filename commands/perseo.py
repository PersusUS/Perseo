"""`perseo` — encender y apagar Perseo entero desde una terminal.

Perseo son cinco cosas y hasta ahora había que saberse las cinco:

    pythonw commands/vigilante.py        el núcleo, con quien lo revive
    pythonw commands/clap_detector.py    el detector de aplausos
    RealTime\\...\\perseo.exe              la app de voz
    ollama app.exe                       sin él no hay triaje
    Obsidian.exe                         sin él la memoria falla con el plugin

Arrancan solos al iniciar sesión en Windows (`manage_startup.py install`), así
que este comando es para el resto de los casos: después de matar algo, después de
un `git pull`, o cuando quieres mirar si está todo en pie sin acordarte de las
cinco rutas.

    perseo on         enciende lo que falte y abre la app
    perseo off        apaga Perseo entero, el detector incluido
    perseo estado     dice qué hay vivo, sin tocar nada
    perseo nucleo     solo el núcleo
    perseo parar      apaga, pero **deja el detector**: se despierta aplaudiendo

`perseo` a secas sigue siendo `perseo on`, que es como se ha escrito siempre en
esta bitácora.

**Nada de esto arranca lo que ya está corriendo.** Se comprueba antes: dos
núcleos peleándose por el puerto 8787 es un `OSError 10048` que no se parece en
nada al problema real, y dos detectores escuchando el mismo micrófono responden
a la vez.

Ollama y Obsidian **se encienden pero no se apagan**. Son programas del señor
Persus, no piezas de Perseo: cerrarle Obsidian con lo que estuviera escribiendo
dentro, porque ha apagado el asistente, sería un mal negocio. `off` lo dice.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
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

#: Dónde se instala Ollama. `ollama app.exe` es la bandeja, que a su vez levanta
#: el servidor; `ollama.exe serve` es el servidor a pelo, y es el respaldo para
#: una instalación donde no esté la ventana. Lo que hace falta es que conteste el
#: 11434: la bandeja es un medio, no el fin.
OLLAMA_BANDEJA = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama app.exe"
OLLAMA_SERVIDOR = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"

#: Dónde se instala Obsidian. Si no está en ninguna, queda el URI `obsidian:`,
#: que es lo que ya usa la lista blanca del agente `pc`.
OBSIDIANES = (
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Obsidian" / "Obsidian.exe",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Obsidian" / "Obsidian.exe",
    Path(os.environ.get("PROGRAMFILES", "")) / "Obsidian" / "Obsidian.exe",
)


def _pythonw() -> str:
    """El intérprete sin consola. Con `python.exe` a secas, cada proceso abriría
    una ventana negra que además lo mata al cerrarla — que es exactamente cómo se
    murió el vigilante el 2026-08-16."""
    ejecutable = sys.executable
    candidato = ejecutable.replace("python.exe", "pythonw.exe")
    return candidato if candidato != ejecutable and os.path.isfile(candidato) else ejecutable


#: Escapar del *job* de quien nos lanzó. Python no lo expone con nombre.
CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def _sin_consola(argumentos: list[str]) -> None:
    """Lanza algo y se desentiende: ni consola, ni esperar, ni morir con esta.

    **`DETACHED_PROCESS` no basta en Windows**, y esto costó tres días de Perseo
    apagado (H-53). Las terminales modernas y los agentes meten lo que ejecutan
    en un *job object* con `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`: cuando esa
    sesión termina, **Windows mata a todos los descendientes**, estén detached o
    no, sin avisar y sin código que lo explique. Medido en esta máquina el
    2026-08-21: la terminal corría dentro de un job con esa bandera puesta.

    Se pide `CREATE_BREAKAWAY_FROM_JOB`, que es lo que existe para salirse. **No
    siempre sirve**, y aquí se comprobó que no basta: los hijos acaban en un job
    igualmente, así que un Perseo lanzado desde una terminal puede seguir
    muriéndose con ella. Se pide de todas formas porque es gratis y en una
    terminal normal sí funciona; lo que de verdad garantiza que Perseo vuelva es
    la tarea `PerseoRevivir` (`manage_startup.py`), que mira cada diez minutos
    desde fuera de cualquier sesión.

    Si el job no admite el intento, `Popen` falla con un `OSError` y se lanza
    como antes: arrancar y quizá morir con la terminal es mejor que no arrancar.
    """
    if os.name != "nt":
        subprocess.Popen(argumentos, cwd=str(RAIZ), close_fds=True)
        return

    banderas = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
    try:
        subprocess.Popen(
            argumentos,
            cwd=str(RAIZ),
            creationflags=banderas | CREATE_BREAKAWAY_FROM_JOB,
            close_fds=True,
        )
    except OSError:
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


def _exe_vivo(nombre: str) -> bool:
    """Si hay algún proceso con ese nombre de ejecutable.

    Es el hermano de `_corriendo`, que solo sabe de procesos de Python: Obsidian
    no deja PID en ningún sitio y su plugin puede estar apagado, así que la
    pregunta «¿está abierto?» no se puede hacer por HTTP.
    """
    if os.name != "nt":
        return False
    try:
        salida = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {nombre}", "/NH"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        # Mismo criterio que `_corriendo`: si no se sabe, se contesta que no.
        # Abrir dos veces algo que ya está abierto se nota; darlo por vivo
        # cuando está muerto deja a Perseo capado sin que nadie lo sepa.
        return False
    return nombre.lower() in salida.lower()


def _primera_que_exista(rutas: tuple[Path, ...]) -> Path | None:
    for ruta in rutas:
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


def arrancar_ollama() -> bool:
    """Levanta Ollama y espera a que conteste. Devuelve si hizo algo.

    Se espera al 11434 y no a que aparezca la ventana: el triaje llama por HTTP,
    y una bandeja abierta con el servidor todavía cargando falla igual que si no
    estuviera. Son unos segundos en frío.
    """
    if _ollama():
        print("  [ya estaba]  Ollama responde en el 11434")
        return False

    binario = OLLAMA_BANDEJA if OLLAMA_BANDEJA.is_file() else None
    argumentos: list[str] | None = None
    if binario is not None:
        argumentos = [str(binario)]
    elif OLLAMA_SERVIDOR.is_file():
        argumentos = [str(OLLAMA_SERVIDOR), "serve"]

    if argumentos is None:
        print("  [falta]      Ollama no está instalado donde se le busca:")
        print(f"               {OLLAMA_BANDEJA.parent}")
        return False

    print("  [arrancando] Ollama")
    _sin_consola(argumentos)

    for _ in range(30):
        time.sleep(0.5)
        if _ollama():
            print("  [listo]      Ollama responde")
            return True
    print("  [ojo]        Ollama no contesta todavía en el 11434; sin él no hay triaje")
    return True


def arrancar_obsidian() -> bool:
    """Abre Obsidian, con el vault que se quedara abierto. Devuelve si hizo algo.

    Se mira **el proceso**, no el plugin: `_obsidian()` pregunta al Local REST
    API, que puede estar apagado en un Obsidian perfectamente abierto. Preguntar
    por ahí abriría una segunda ventana cada vez que el plugin estuviera caído.
    """
    if _exe_vivo("Obsidian.exe"):
        print("  [ya estaba]  Obsidian")
        return False

    binario = _primera_que_exista(OBSIDIANES)
    if binario is not None:
        print("  [arrancando] Obsidian")
        _sin_consola([str(binario)])
        return True

    # Sin ruta conocida queda el URI, que es lo que ya usa la lista blanca del
    # agente `pc`. Abre el mismo programa por el registro de Windows.
    print("  [arrancando] Obsidian (por el enlace obsidian:)")
    webbrowser.open("obsidian://")
    return True


#: Fichero que Rust vigila para sacar la ventana del escondite. Es distinto del
#: de la palabra clave (`.perseo-autollamada`), que además entra en llamada.
MARCADOR_MOSTRAR = RAIZ / ".perseo-mostrar"


def arrancar_app() -> bool:
    if presencia.app_viva():
        # Cerrar la ventana no cierra Perseo: la esconde en la bandeja. Así que
        # "ya estaba" era verdad y a la vez inútil — el usuario escribía `perseo`
        # y no pasaba nada en pantalla, que desde fuera se ve igual que una app
        # rota. Se pide que vuelva.
        try:
            MARCADOR_MOSTRAR.write_text("", encoding="utf-8")
            print("  [al frente]  La app de voz ya estaba; se saca de la bandeja")
        except OSError:
            print("  [ya estaba]  La app de voz (escondida en la bandeja)")
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
    print("Las dependencias, que `perseo on` enciende:")
    # De Obsidian se dicen las dos cosas: si el programa está abierto y si su
    # plugin contesta. Un Obsidian abierto con el plugin apagado deja la memoria
    # igual de rota que un Obsidian cerrado, y son dos arreglos distintos.
    obsidian_abierto = _exe_vivo("Obsidian.exe")
    obsidian_plugin = _obsidian()
    for nombre, vivo, sin_el in (
        ("Ollama", _ollama(), "sin él no hay triaje de correo"),
        (
            "Obsidian",
            obsidian_abierto,
            "sin él la memoria falla si el vault va por el plugin"
            if obsidian_plugin or not obsidian_abierto
            else "abierto, pero su Local REST API no contesta: mira el plugin",
        ),
    ):
        marca = "[activo]  " if vivo else "[PARADO]  "
        print(f"  {marca}   {nombre} — {sin_el}")

    print()
    detector = "[activo]  " if _corriendo("clap_detector.py") else "[PARADO]  "
    app = "[activo]  " if presencia.app_viva() else "[PARADO]  "
    print(f"  {detector}   Detector de aplausos")
    print(f"  {app}   App de voz")


def parar(avisar_del_detector: bool = True) -> None:
    """Cierra la app y el núcleo. **Al vigilante primero**, o resucita el núcleo.

    No toca el detector de aplausos: es lo que despierta a Perseo, y pararlo sin
    querer deja el sistema mudo de una forma que no se nota hasta que aplaudes.
    Quien sí lo mata es `apagar`, y por eso puede callar ese aviso.
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
    if avisar_del_detector:
        print("\n  El detector de aplausos sigue en pie: es lo que despierta a Perseo.")


def parar_detector() -> None:
    """Mata el detector de aplausos. Lo que separa `off` de `parar`."""
    if os.name != "nt":
        print("Esto solo sabe parar procesos en Windows.")
        return
    subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" "
            "| Where-Object { $_.CommandLine -like '*clap_detector.py*' } "
            "| ForEach-Object { Stop-Process -Id $_.ProcessId -Force }",
        ],
        capture_output=True,
    )
    print("  [parado]     El detector de aplausos")


def apagar() -> None:
    """`perseo off`: Perseo entero, el detector incluido.

    Es lo que `parar` no hace, y la diferencia importa: con el detector vivo,
    Perseo sigue escuchando el micrófono y dos palmadas lo encienden otra vez.
    Eso está bien para reiniciar el núcleo y mal para apagarlo de verdad.

    Ollama y Obsidian se quedan abiertos: son programas del señor Persus, no
    piezas de Perseo.
    """
    print("Apagando Perseo:\n")
    parar(avisar_del_detector=False)
    parar_detector()
    print("\n  Ollama y Obsidian siguen abiertos: no son de Perseo.")
    print("  Para volver: perseo on")


def todo() -> None:
    """`perseo on`: las cinco piezas, y las dependencias antes que nada.

    Ollama y Obsidian van primero **a propósito**: el núcleo arranca igual sin
    ellos, pero el primer triaje de correo y la primera nota que se guarde caen
    en el hueco. Ollama, además, tarda en contestar, así que lo que se gana es
    ese arranque mientras suben el núcleo y la app.
    """
    print("Encendiendo Perseo:\n")
    arrancar_ollama()
    arrancar_obsidian()
    arrancar_nucleo()
    arrancar_detector()
    arrancar_app()
    print("\n  Panel y cola: botón de cuadrícula en la app, o http://127.0.0.1:8787")


#: Las órdenes, con sus sinónimos. `on` y `off` son las que pidió el señor
#: Persus el 2026-08-21; `perseo` a secas se queda como `on` porque es lo que
#: dice la bitácora entera, y `parar` porque apagar dejando el detector vivo
#: sigue siendo útil para reiniciar el núcleo sin quedarse sordo.
ORDENES = {
    "": todo,
    "on": todo,
    "encender": todo,
    "off": apagar,
    "apagar": apagar,
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
