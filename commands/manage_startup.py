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
    r"""La línea que se escribe en el registro, o `None` si falta algo.

    El núcleo no se arranca desde aquí, sino a través de `vigilante.py`, que sí
    es un fichero suelto. De paso se evita el otro problema: `perseo_core` es un
    paquete y sus módulos se importan entre sí, así que `pythonw
    perseo_core\__main__.py` falla con un ImportError que no dice nada del
    problema real. El vigilante lo lanza como módulo y con el directorio de
    trabajo puesto. Sin shell, que es la regla de toda la casa.
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

    # En el registro va el vigilante, no el núcleo. `Run` lanza una vez: si el
    # núcleo se cae a media tarde, sin un padre que lo levante Perseo se apaga
    # hasta el siguiente reinicio y nadie se entera.
    vigilante = os.path.join(raiz, "commands", "vigilante.py")
    if not os.path.isfile(vigilante):
        print(f"[-] No se encuentra el vigilante: {vigilante}")
        return None
    return f'"{_pythonw()}" "{vigilante}"'


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


def rutas_del_comando(comando: str) -> list[str]:
    """Las rutas entrecomilladas de una línea del registro: intérprete y script."""
    return [trozo for trozo in comando.split('"') if os.path.sep in trozo or ":" in trozo]


def revisar(comando: str) -> list[str]:
    """Qué hay roto en una entrada del registro, si es que hay algo.

    Una entrada de `Run` guarda rutas absolutas y **no avisa cuando dejan de
    existir**: Windows intenta arrancarla, falla en silencio, y lo que ve el
    usuario es que Perseo ya no está. Pasa al actualizar Python —la ruta lleva la
    versión dentro— y al mover la carpeta del proyecto.
    """
    problemas = []
    for ruta in rutas_del_comando(comando):
        if not os.path.exists(ruta):
            problemas.append(f"no existe: {ruta}")
    return problemas


def url_salud() -> str:
    """Dónde preguntar si el núcleo está vivo. Misma variable que usa él."""
    puerto = os.environ.get("PERSEO_CORE_PUERTO", "8787").strip() or "8787"
    return f"http://127.0.0.1:{puerto}/salud"


def nucleo_responde(url: str | None = None, espera: float = 3.0) -> bool:
    """Si hay un núcleo contestando ahí ahora mismo.

    `/salud` es pública a propósito, así que esto no necesita el token. Se
    pregunta por HTTP y no por la lista de procesos porque lo que importa no es
    que exista un `python.exe`, sino que la API atienda: un núcleo colgado sigue
    siendo un proceso vivo.
    """
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url or url_salud(), timeout=espera) as respuesta:
            return respuesta.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def estado() -> None:
    print("Estado del arranque automático:\n")
    for nombre in SERVICIOS:
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUTA_CLAVE) as clave:
                valor, _ = winreg.QueryValueEx(clave, nombre)
        except FileNotFoundError:
            print(f"  [inactivo] {nombre}")
            continue

        problemas = revisar(valor)
        etiqueta = "[roto]  " if problemas else "[activo]"
        print(f"  {etiqueta}   {nombre}\n             {valor}")
        for problema in problemas:
            print(f"             ^ {problema}")
        if problemas:
            print("             Vuelve a instalarlo: python manage_startup.py install " + nombre)

    # Lo que arranca con Windows no trae las variables de tu terminal. Si no hay
    # `entorno.json`, el núcleo se levanta capado —sin correo, sin agenda, sin
    # vault por Obsidian— y parece que funciona.
    ajustes = os.path.join(_raiz_proyecto(), "perseo_core", "datos", "entorno.json")
    print()
    if os.path.isfile(ajustes):
        print(f"  [activo]   Configuración de arranque\n             {ajustes}")
    else:
        print("  [aviso]    No hay perseo_core/datos/entorno.json.")
        print("             El núcleo arrancará sin correo, sin agenda y sin vault REST.")
        print("             Escríbelo con: python commands/configurar_arranque.py")

    # Y lo que todo lo de arriba **no** dice: si ahora mismo hay algo encendido.
    # Una entrada del registro solo cuenta lo que Windows intentará arrancar en
    # la próxima sesión. El 2026-08-16 esto decía "activo" con el vigilante
    # muerto —se había lanzado desde una consola y murió con ella— y Perseo
    # llevaba horas apagado sin que nada lo dijera.
    print()
    if nucleo_responde():
        print(f"  [activo]   El núcleo responde en {url_salud()}")
    else:
        print(f"  [PARADO]   Nadie contesta en {url_salud()}")
        print("             El registro dice lo que arrancará en el próximo inicio de sesión,")
        print("             no si algo está encendido ahora. Para levantarlo sin reiniciar:")
        print("             pythonw commands/vigilante.py")


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
