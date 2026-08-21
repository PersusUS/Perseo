"""Lo que Windows lanza: al iniciar sesión, todo; cada diez minutos, lo que falte.

Antes en el registro había dos entradas —el vigilante y el detector— y la app de
voz no estaba en ninguna. O sea que al encender el PC, Perseo arrancaba a medias
y sin ventana. Ahora en el registro va **una sola línea**, esta, que llama a lo
mismo que llamaría una persona:

    perseo on

Con eso el arranque de Windows y el de la terminal dejan de ser dos caminos que
se pueden desincronizar. Lo que se arregle en `perseo.py` vale para los dos.

**Y una tarea programada llama aquí cada diez minutos con `--revivir`.** No es
un lujo: el 2026-08-18 el núcleo murió a las 16:42, el vigilante se fue detrás, y
Perseo estuvo **tres días apagado** con el detector escuchando para nada. Un
vigilante no puede vigilar su propia muerte; alguien de fuera tiene que mirar.

Dos diferencias entre los dos modos, y las dos son a propósito:

* **`--revivir` solo levanta el núcleo y el detector.** Ni la app, ni Obsidian,
  ni Ollama: son cosas que el señor Persus puede haber cerrado porque quería
  cerrarlas, y una ventana que reaparece sola cada diez minutos es un programa
  con el que no se puede convivir. El núcleo y el detector no tienen ventana: si
  están apagados es que algo falló.
* **Al iniciar sesión sí se abre todo**, porque encender el PC es justamente
  pedir que esté todo.

Este proceso **no tiene consola**: lo lanza `pythonw` desde el registro, así que
`print` escribiría contra un `sys.stdout` que vale `None` y reventaría en la
primera línea. Todo lo que diga va a `<datos>/arranque.log`, que además es el
sitio donde mirar el día que el PC arranque y Perseo no. Es H-34 y H-41 otra vez:
un proceso sin consola que no escribe en un fichero no está diciendo nada.

    python commands/arranque.py              # a mano, para verlo trabajar
    python commands/arranque.py --revivir    # solo lo que se haya caído
"""

from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent

sys.path.insert(0, str(AQUI))

import vigilante  # noqa: E402

#: Cuánto puede ocupar `arranque.log` antes de apartarlo. Mismo criterio que el
#: registro del núcleo: esto arranca con Windows y nadie lo mira hasta que hace
#: falta, así que sin tope crece para siempre.
TOPE_REGISTRO = 2 * 1024 * 1024


def _directorio_datos() -> Path:
    ruta = Path(os.environ.get("PERSEO_CORE_DATOS", RAIZ / "perseo_core" / "datos"))
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta


def registro() -> Path:
    return _directorio_datos() / "arranque.log"


def _cabecera(modo: str) -> str:
    marca = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"\n===== {modo} {marca} =====\n"


def revivir() -> None:
    """El núcleo y el detector, si se han caído. Nada más.

    Los dos comprueban antes si ya están vivos, así que esto es inofensivo
    cuando todo va bien: la vuelta normal no arranca nada y solo deja una línea.
    """
    import perseo

    perseo.arrancar_nucleo()
    perseo.arrancar_detector()


def encender() -> None:
    import perseo

    perseo.todo()


def main(argumentos: list[str] | None = None) -> int:
    argumentos = sys.argv[1:] if argumentos is None else argumentos
    modo_revivir = "--revivir" in argumentos

    fichero = registro()
    vigilante.apartar_si_crece(fichero, TOPE_REGISTRO)

    try:
        salida = fichero.open("a", encoding="utf-8")
    except OSError:
        # Sin registro se sigue adelante: quedarse sin arrancar Perseo por no
        # poder escribir en un fichero de texto sería el peor de los cambios.
        salida = None

    anterior = (sys.stdout, sys.stderr)
    if salida is not None:
        sys.stdout = sys.stderr = salida
        salida.write(_cabecera("Revivir" if modo_revivir else "Arranque de sesión"))

    try:
        if modo_revivir:
            revivir()
        else:
            encender()
        return 0
    except Exception:
        # Un fallo aquí es un Perseo que no arranca con el PC, que es el síntoma
        # más difícil de investigar de todos: no hay consola, no hay error, y lo
        # que se ve es que "Perseo ya no está".
        traceback.print_exc()
        return 1
    finally:
        if salida is not None:
            sys.stdout, sys.stderr = anterior
            try:
                salida.close()
            except OSError:
                pass


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
