"""Gestiona qué arranca con Windows, y qué lo revive si se cae.

**Desde el 2026-08-22 el registro no arranca nada de Perseo.** Lo decidió el
señor Persus: Perseo se abre con `perseo on` o despertado por el detector de
aplausos, y por más nada — una entrada en `Run` que abriera ventanas al
encender el PC es justo lo que no quiere. Este script queda para dos cosas:

* **Limpiar entradas viejas** (`LEGADO`) si sobreviven en una máquina antigua:
  convivirían con nada y aun así arrancarían un segundo detector que se pelea
  por el micrófono.
* **Poner y quitar la tarea programada** `PerseoRevivir`, que cada diez minutos
  llama a `commands/arranque.py --revivir` para levantar el núcleo o el detector
  si se han caído. Es lo único que mira desde fuera de cualquier sesión, y es lo
  que de verdad evita otro H-53: el 2026-08-18 el núcleo murió a las 16:42, el
  vigilante se fue detrás y Perseo estuvo tres días apagado sin que nada lo
  dijera. Un vigilante no puede vigilar su propia muerte.

Antes de v2 aquí había un tercer servicio, el indexador del vault
(`RAG/automator.py`), que se jubiló: la memoria la lleva ahora el agente
`memoria` del núcleo, que escribe en el mismo vault sin base vectorial de por
medio.
"""

import os
import subprocess
import sys
import winreg

RUTA_CLAVE = r"Software\Microsoft\Windows\CurrentVersion\Run"

#: Nada. Lo que había aquí era una entrada `Run` → `arranque.py`, y antes aún
#: dos más (vigilante y detector por separado). El 2026-08-22 se decidió que el
#: PC encendido no abre Perseo: la lista queda vacía para que `install` no
#: vuelva a escribirla por error. Si algún día cambia la decisión, aquí va el
#: nombre de la entrada y `_comando()` sigue abajo como referencia.
SERVICIOS = ()

#: Lo que hubo algún día y hay que quitar si se encuentra. Si se quedan,
#: arrancan solos al iniciar sesión y se pelean por el mismo micrófono.
LEGADO = ("PerseoClapDetector", "PerseoNucleo")

#: La tarea programada que revive lo que se caiga, y cada cuánto mira.
TAREA = "PerseoRevivir"
MINUTOS_ENTRE_REVISIONES = 10


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
    r"""La línea que se escribiría en el registro, si algún día se vuelve ahí.

    **Hoy no la llama nadie**: `SERVICIOS` está vacío porque el registro no
    arranca Perseo (2026-08-22). Se conserva escrita para que volver a tenerla
    no sea reescribirla: el núcleo no se lanza directo sino a través de
    `vigilante.py` — `pythonw perseo_core\__main__.py` falla con un ImportError
    que no dice nada del problema real, y el vigilante lo lanza como módulo y
    con el directorio de trabajo puesto. Sin shell, que es la regla de la casa.
    """
    raiz = _raiz_proyecto()

    if not os.path.isdir(os.path.join(raiz, "perseo_core")):
        print(f"[-] No se encuentra el paquete perseo_core en {raiz}")
        return None

    guion = os.path.join(raiz, "commands", "arranque.py")
    if not os.path.isfile(guion):
        print(f"[-] No se encuentra el guion de arranque: {guion}")
        return None
    return f'"{_pythonw()}" "{guion}"'


# --------------------------------------------------------------------------- #
# La tarea que revive lo que se caiga
# --------------------------------------------------------------------------- #


def _orden_de_la_tarea() -> str | None:
    raiz = _raiz_proyecto()
    guion = os.path.join(raiz, "commands", "arranque.py")
    if not os.path.isfile(guion):
        print(f"[-] No se encuentra el guion de arranque: {guion}")
        return None
    return f'"{_pythonw()}" "{guion}" --revivir'


def añadir_tarea() -> None:
    """Crea la tarea programada que mira cada diez minutos si falta algo.

    `/F` la reemplaza si ya existía: reinstalar tras mover la carpeta tiene que
    actualizar la ruta, no fallar diciendo que ya está.
    """
    orden = _orden_de_la_tarea()
    if orden is None:
        return
    try:
        resultado = subprocess.run(
            [
                "schtasks", "/Create", "/F",
                "/TN", TAREA,
                "/TR", orden,
                "/SC", "MINUTE",
                "/MO", str(MINUTOS_ENTRE_REVISIONES),
            ],
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as e:
        print(f"[-] No se pudo crear la tarea '{TAREA}': {e}")
        return
    if resultado.returncode == 0:
        print(f"[+] '{TAREA}' revisará cada {MINUTOS_ENTRE_REVISIONES} minutos que Perseo siga en pie.")
    else:
        print(f"[-] No se pudo crear la tarea '{TAREA}': {resultado.stderr.strip() or resultado.stdout.strip()}")


def quitar_tarea() -> None:
    try:
        resultado = subprocess.run(
            ["schtasks", "/Delete", "/F", "/TN", TAREA], capture_output=True, text=True
        )
    except (OSError, subprocess.SubprocessError) as e:
        print(f"[-] No se pudo borrar la tarea '{TAREA}': {e}")
        return
    if resultado.returncode == 0:
        print(f"[+] '{TAREA}' ya no revisará nada.")
    else:
        print(f"[i] '{TAREA}' no estaba puesta.")


def tarea_puesta() -> bool:
    try:
        resultado = subprocess.run(
            ["schtasks", "/Query", "/TN", TAREA], capture_output=True, text=True
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return resultado.returncode == 0


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
    if SERVICIOS:
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
    else:
        print("  [decisión] El registro no abre Perseo al encender el PC (2026-08-22):")
        print("             se enciende con `perseo on` o con dos palmadas.")

    for nombre in LEGADO:
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUTA_CLAVE) as clave:
                valor, _ = winreg.QueryValueEx(clave, nombre)
        except FileNotFoundError:
            continue
        # Si sobrevive una entrada vieja, arranca a la vez que `Perseo` y las dos
        # se pelean: dos núcleos por el puerto 8787, dos detectores por el mismo
        # micrófono.
        print(f"  [sobra]    {nombre}\n             {valor}")
        print("             Es del arranque de antes. Quítalo: python manage_startup.py install")

    print()
    if tarea_puesta():
        print(f"  [activo]   {TAREA}")
        print(f"             Cada {MINUTOS_ENTRE_REVISIONES} min levanta el núcleo o el detector si se han caído")
    else:
        print(f"  [inactivo] {TAREA}")
        print("             Nadie revive a Perseo si se cae con el PC encendido.")
        print("             Ponla con: python manage_startup.py install")

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
    print("  python manage_startup.py install   limpia entradas viejas y pone la tarea que revive")
    print("  python manage_startup.py remove    quita la tarea")
    print("  python manage_startup.py status    muestra el estado")


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
        quitar_tarea()
    elif accion == "install":
        print("--- Configuración de arranque automático ---")
        for nombre in objetivos:
            añadir(nombre)
        # Las entradas de antes se van al instalar, no cuando alguien se acuerde:
        # conviviendo con la nueva, arrancan dos núcleos y dos detectores.
        for nombre in LEGADO:
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUTA_CLAVE) as clave:
                    winreg.QueryValueEx(clave, nombre)
            except FileNotFoundError:
                continue
            print(f"[i] '{nombre}' es del arranque de antes y sobra ahora.")
            quitar(nombre)
        añadir_tarea()
        print("\nPara comprobarlo:  python manage_startup.py status")
        print("Para desactivarlo: python manage_startup.py remove")
    else:
        print(f"[-] Acción desconocida: '{accion}'")
        _uso()
        raise SystemExit(1)
