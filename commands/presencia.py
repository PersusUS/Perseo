"""¿Está Perseo abierto?

El detector necesita saberlo: si la aplicación ya está viva, un doble aplauso
solo tiene que despertarla; si no, hay que lanzarla, que tarda. Hasta ahora se
averiguaba buscando `temp-app.exe` en el `tasklist`, y eso ataba dos cosas que no
deberían estarlo — **el nombre del binario y el funcionamiento del detector**.
Renombrar el paquete Rust rompía la detección sin que nada avisara: la app
seguía arrancando, y de pronto cada aplauso abría una segunda instancia.

Ahora la aplicación deja su PID en un fichero al arrancar (`presencia.rs`) y aquí
se pregunta al sistema si ese proceso sigue vivo. Da igual cómo se llame el
ejecutable.

**Lo que se comprueba es el proceso, no el fichero.** Si la app muere de mala
manera, el fichero se queda; comprobar solo su existencia haría creer para
siempre que Perseo está abierto, y entonces el aplauso dejaría de lanzarlo.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: El fichero que deja la aplicación. Mismo nombre que en `presencia.rs`.
FICHERO = ".perseo-app.pid"

_RAIZ_PROYECTO = Path(__file__).resolve().parent.parent


def rutas() -> list[Path]:
    """Dónde puede estar la marca: en el árbol de fuentes y en la configuración.

    Los dos sitios, y en este orden, porque son los dos que escribe la
    aplicación: el árbol cuando se trabaja con `tauri dev` y el directorio de
    configuración cuando está instalada.
    """
    candidatas = [_RAIZ_PROYECTO / FICHERO]

    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            # Tauri usa el identificador de `tauri.conf.json` como carpeta.
            candidatas.append(Path(appdata) / "com.perseo.app" / FICHERO)
    else:
        candidatas.append(Path.home() / ".config" / "com.perseo.app" / FICHERO)

    return candidatas


def _proceso_vivo(pid: int) -> bool:
    """Si existe un proceso con ese identificador.

    En Windows no vale `os.kill(pid, 0)`: Python lo traduce a `TerminateProcess`,
    así que preguntar mataría justo lo que se quería comprobar. Se abre un
    manejador con el permiso mínimo y se mira si el proceso sigue sin señalar.
    """
    # Un identificador de proceso cabe en 32 bits en los dos sistemas. Lo que se
    # salga de ahí no es un proceso, es basura en el fichero, y conviene
    # descartarlo **antes** de preguntar: `os.kill` en Linux y `OpenProcess` en
    # Windows no lo rechazan igual, y uno de los dos lanza. Lo encontró el CI.
    if pid <= 0 or pid > 0xFFFF_FFFF:
        return False

    if sys.platform == "win32":
        import ctypes

        SYNCHRONIZE = 0x00100000
        WAIT_TIMEOUT = 0x00000102
        kernel32 = ctypes.windll.kernel32

        manejador = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if not manejador:
            return False
        try:
            # Un proceso vivo no está señalado, así que la espera de cero
            # milisegundos se agota. Uno que ya terminó devuelve WAIT_OBJECT_0.
            return kernel32.WaitForSingleObject(manejador, 0) == WAIT_TIMEOUT
        finally:
            kernel32.CloseHandle(manejador)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Existe, pero es de otro usuario. Para lo que se pregunta aquí, existe.
        return True
    except (OverflowError, OSError):
        # Un número que no cabe en el `pid_t` del sistema no es un proceso: es
        # basura en el fichero. `os.kill` lanza `OverflowError` en Linux con
        # cualquier PID por encima de `INT_MAX`, y sin esto una marca corrupta
        # tumbaba al que preguntara.
        return False
    return True


def pid_anunciado() -> int | None:
    """El PID que dejó la aplicación, o `None` si no hay ninguno legible."""
    for ruta in rutas():
        try:
            if not ruta.is_file():
                continue
            crudo = ruta.read_text(encoding="utf-8").strip()
            if crudo.isdigit():
                return int(crudo)
        except OSError:
            continue
    return None


def app_viva() -> bool:
    """Si Perseo está abierto ahora mismo."""
    pid = pid_anunciado()
    return _proceso_vivo(pid) if pid is not None else False


def limpiar_marca_rancia() -> None:
    """Borra la marca si el proceso ya no existe.

    No hace falta para decidir nada —`app_viva` ya mira el proceso— pero evita
    que un fichero de hace tres arranques confunda a quien lo lea a mano.
    """
    pid = pid_anunciado()
    if pid is None or _proceso_vivo(pid):
        return
    for ruta in rutas():
        try:
            if ruta.is_file():
                ruta.unlink()
        except OSError:
            pass


if __name__ == "__main__":  # pragma: no cover
    pid = pid_anunciado()
    if pid is None:
        print("No hay marca: Perseo no ha dejado ningun PID.")
    else:
        print(f"PID anunciado: {pid} -> {'vivo' if _proceso_vivo(pid) else 'muerto'}")
