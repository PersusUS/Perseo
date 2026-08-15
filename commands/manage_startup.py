"""Gestiona qué procesos de Perseo arrancan con Windows.

Escribe en HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run, que afecta
solo al usuario actual y es reversible desde este mismo script.

Son dos servicios independientes:
  - El detector de aplausos, que despierta la aplicación.
  - El núcleo (`perseo-core`), que es lo que tiene que estar siempre encendido:
    sin él no hay cola, ni memoria, ni triaje de correo, y la app se queda sin
    herramientas.

Antes el segundo servicio era el indexador del vault (`RAG/automator.py`), que
se jubiló con el paso a v2: la memoria la lleva ahora el agente `memoria` del
núcleo, que escribe en el mismo vault sin base vectorial de por medio.
"""

import os
import sys
import winreg

RUTA_CLAVE = r"Software\Microsoft\Windows\CurrentVersion\Run"

#: Nombre en el registro -> qué se arranca. El detector es un script suelto; el
#: núcleo es un paquete y hay que arrancarlo como módulo, que es distinto.
SERVICIOS = ("PerseoClapDetector", "PerseoNucleo")


def _raiz_proyecto() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _pythonw() -> str:
    """Intérprete sin consola, para que los servicios corran en silencio."""
    ejecutable = sys.executable
    if ejecutable.endswith("python.exe"):
        candidato = ejecutable.replace("python.exe", "pythonw.exe")
        if os.path.isfile(candidato):
            return candidato
    return ejecutable


def _comando(nombre: str) -> str | None:
    """La línea que se escribe en el registro, o `None` si falta algo.

    El núcleo no se puede arrancar como fichero suelto: `perseo_core` es un
    paquete y sus módulos se importan entre sí con rutas relativas, así que
    `pythonw perseo_core\__main__.py` falla con un ImportError que no dice nada
    del problema real. Se arranca como módulo, y como una entrada del registro no
    tiene directorio de trabajo, la raíz del proyecto se mete en `sys.path` a
    mano. Sin shell, que es la regla de toda la casa.
    """
    raiz = _raiz_proyecto()

    if nombre == "PerseoClapDetector":
        ruta = os.path.join(raiz, "commands", "clap_detector.py")
        if not os.path.isfile(ruta):
            print(f"[-] No se encuentra el script: {ruta}")
            return None
        return f'"{_pythonw()}" "{ruta}"'

    if not os.path.isdir(os.path.join(raiz, "perseo_core")):
        print(f"[-] No se encuentra el paquete perseo_core en {raiz}")
        return None
    arranque = (
        f"import sys; sys.path.insert(0, r'{raiz}'); "
        "import runpy; runpy.run_module('perseo_core', run_name='__main__')"
    )
    return f'"{_pythonw()}" -c "{arranque}"'


def añadir(nombre: str) -> None:
    comando = _comando(nombre)
    if comando is None:
        return
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUTA_CLAVE, 0, winreg.KEY_SET_VALUE) as clave:
            winreg.SetValueEx(clave, nombre, 0, winreg.REG_SZ, comando)
        print(f"[+] '{nombre}' arrancará automáticamente con Windows.")
    except Exception as e:
        print(f"[-] Error al registrar '{nombre}': {e}")


def quitar(nombre: str) -> None:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUTA_CLAVE, 0, winreg.KEY_SET_VALUE) as clave:
            winreg.DeleteValue(clave, nombre)
        print(f"[+] '{nombre}' ya no arrancará con Windows.")
    except FileNotFoundError:
        print(f"[i] '{nombre}' no estaba configurado para arrancar con Windows.")
    except Exception as e:
        print(f"[-] Error al eliminar '{nombre}': {e}")


def estado() -> None:
    print("Estado del arranque automático:\n")
    for nombre in SERVICIOS:
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUTA_CLAVE) as clave:
                valor, _ = winreg.QueryValueEx(clave, nombre)
            print(f"  [activo]   {nombre}\n             {valor}")
        except FileNotFoundError:
            print(f"  [inactivo] {nombre}")


def _uso() -> None:
    print(__doc__)
    print("Uso:")
    print("  python manage_startup.py install [servicio]   activa el arranque")
    print("  python manage_startup.py remove  [servicio]   lo desactiva")
    print("  python manage_startup.py status               muestra el estado")
    print()
    print(f"  servicios: {', '.join(SERVICIOS)} (por defecto, todos)")


if __name__ == "__main__":
    accion = sys.argv[1].lower() if len(sys.argv) > 1 else "install"
    objetivo = sys.argv[2] if len(sys.argv) > 2 else None

    if accion in ("-h", "--help", "help"):
        _uso()
        raise SystemExit(0)

    if objetivo and objetivo not in SERVICIOS:
        print(f"[-] Servicio desconocido: '{objetivo}'. Opciones: {', '.join(SERVICIOS)}")
        raise SystemExit(1)

    objetivos = [objetivo] if objetivo else list(SERVICIOS)

    if accion == "status":
        estado()
    elif accion == "remove":
        for nombre in objetivos:
            quitar(nombre)
    elif accion == "install":
        print("--- Configuración de arranque automático ---")
        for nombre in objetivos:
            añadir(nombre)
        print("\nPara comprobarlo:  python manage_startup.py status")
        print("Para desactivarlo: python manage_startup.py remove")
    else:
        print(f"[-] Acción desconocida: '{accion}'")
        _uso()
        raise SystemExit(1)
