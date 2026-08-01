"""Gestiona qué procesos de Perseo arrancan con Windows.

Escribe en HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run, que afecta
solo al usuario actual y es reversible desde este mismo script.

Son dos servicios independientes:
  - El detector de aplausos, que despierta la aplicación.
  - El indexador del vault, que mantiene al día la memoria a largo plazo.

El indexador no arrancaba en ningún sitio, y era una de las razones por las que
la base vectorial llevaba meses vacía. Ver H-02.
"""

import os
import sys
import winreg

RUTA_CLAVE = r"Software\Microsoft\Windows\CurrentVersion\Run"

# nombre en el registro -> (script, carpeta relativa a la raíz del proyecto)
SERVICIOS = {
    "PerseoClapDetector": ("clap_detector.py", "commands"),
    "PerseoRagIndexer": ("automator.py", "RAG"),
}


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


def añadir(nombre: str) -> None:
    script, carpeta = SERVICIOS[nombre]
    ruta = os.path.join(_raiz_proyecto(), carpeta, script)

    if not os.path.isfile(ruta):
        print(f"[-] No se encuentra el script: {ruta}")
        return

    comando = f'"{_pythonw()}" "{ruta}"'
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
