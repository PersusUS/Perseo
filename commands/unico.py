"""Que de una pieza solo haya una corriendo, aunque falle quien lo comprueba.

`perseo.py` ya pregunta «¿está vivo el detector?» antes de arrancarlo, y esa
pregunta es una consulta a Windows por WMI que **puede fallar**: tarda, se pasa
del tope de veinte segundos, salta la excepción y `_corriendo` contesta que no
—que es lo conservador para el núcleo, porque dar por vivo algo muerto deja a
Perseo apagado sin que nadie lo sepa—. Con el detector de aplausos ese criterio
sale al revés: contestar «no está» de más arranca otro, y la tarea
`PerseoRevivir` lo intenta **cada diez minutos**, para siempre.

Se vio el 2026-08-26 con tres detectores a la vez sobre el mismo micrófono
(H-76). Tres procesos oyendo el mismo aplauso disparan tres veces, y desde
fuera el síntoma es que Perseo «se abre solo» o que la palabra clave hace cosas
raras.

La cerradura es un **mutex con nombre del sistema operativo** y no un fichero de
PID a propósito: lo libera Windows cuando el proceso muere, se cuelgue como se
cuelgue. Un fichero hay que borrarlo, y el día que no se borra la pieza no
vuelve a arrancar nunca — que es un fallo peor que el que se está arreglando.

    cerrojo = tomar("PerseoDetectorAplausos")
    if cerrojo is None:
        print("Ya hay otro; me voy.")
        raise SystemExit(0)
    # ... y `cerrojo` se guarda vivo mientras dure el proceso.
"""

from __future__ import annotations

import os
from typing import Any

#: Lo que devuelve `CreateMutexW` por `GetLastError` cuando ya existía.
_ERROR_ALREADY_EXISTS = 183


class Cerrojo:
    """La cerradura tomada. Mantenerla viva es lo que la sostiene."""

    def __init__(self, asa: Any, ruta: str = "") -> None:
        self._asa = asa
        self._ruta = ruta

    def soltar(self) -> None:
        """Devuelve la cerradura. No hace falta llamarlo: morir también vale."""
        if self._asa is None:
            return
        try:
            if os.name == "nt":
                import ctypes

                ctypes.windll.kernel32.CloseHandle(self._asa)
            else:
                self._asa.close()
                if self._ruta:
                    with _silencio():
                        os.unlink(self._ruta)
        finally:
            self._asa = None


class _silencio:
    """`contextlib.suppress(OSError)` sin importar contextlib por una línea."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, tipo, valor, traza) -> bool:
        return isinstance(valor, OSError)


def tomar(nombre: str) -> Cerrojo | None:
    """La cerradura de esa pieza, o `None` si ya la tiene otro proceso.

    Si el sistema no deja tomarla —permisos, una plataforma rara— se devuelve
    una cerradura de mentira y se sigue adelante: esto evita duplicados, no es
    un requisito para funcionar. Quedarse sin detector de aplausos por no poder
    crear un mutex sería cambiar un problema por otro peor.
    """
    if os.name == "nt":
        return _tomar_en_windows(nombre)
    return _tomar_en_posix(nombre)


def _tomar_en_windows(nombre: str) -> Cerrojo | None:
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        # `Local\` y no `Global\`: la pieza es de esta sesión de escritorio, y
        # `Global\` necesita privilegios que un proceso normal no tiene.
        asa = kernel32.CreateMutexW(None, False, f"Local\\{nombre}")
        if not asa:
            return Cerrojo(None)
        if kernel32.GetLastError() == _ERROR_ALREADY_EXISTS:
            kernel32.CloseHandle(asa)
            return None
        return Cerrojo(asa)
    except Exception:  # noqa: BLE001 — sin cerradura se sigue, ver el docstring
        return Cerrojo(None)


def _tomar_en_posix(nombre: str) -> Cerrojo | None:
    try:
        import fcntl
        import tempfile

        ruta = os.path.join(tempfile.gettempdir(), f"{nombre}.lock")
        fichero = open(ruta, "w", encoding="utf-8")  # noqa: SIM115 — vive con el proceso
        try:
            fcntl.flock(fichero.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fichero.close()
            return None
        fichero.write(str(os.getpid()))
        fichero.flush()
        return Cerrojo(fichero, ruta)
    except Exception:  # noqa: BLE001 — ver el docstring
        return Cerrojo(None)
