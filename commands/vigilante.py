"""Mantiene el núcleo encendido, que es lo único que se le pide.

Una entrada en `HKCU\\...\\Run` lanza un proceso **una vez**. Si el núcleo se
cae a las tres horas —Ollama que se lleva la memoria por delante, un disco
lleno, una excepción no prevista— Windows no lo vuelve a levantar y nadie se
entera: no hay cola, no hay triaje, no hay avisos. Y el síntoma es que Perseo
"dejó de hacer cosas", que no lleva a ninguna causa.

Así que en el registro va esto, no el núcleo: un padre que arranca al hijo, se
queda esperando y lo vuelve a arrancar si se muere.

Tres reglas:

1. **La espera crece.** Si el núcleo revienta al arrancar —configuración rota,
   puerto ocupado— reintentar cada segundo llena el disco de registro y no
   arregla nada. Se empieza en 5 s y se dobla hasta un minuto.
2. **Una salida limpia es una orden.** Si el núcleo termina con código 0, es que
   alguien lo paró a propósito; el vigilante se va detrás. Solo se reinicia lo
   que se muere mal.
3. **Se deja escrito.** Todo va a `<datos>/vigilante.log`, con fecha, porque
   este proceso no tiene consola: lo lanza `pythonw` desde el registro. Es H-34
   aplicado antes de que muerda.

    python commands/vigilante.py           # a mano, para verlo trabajar
    python commands/manage_startup.py install PerseoNucleo
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

#: Cuánto se espera antes del primer reintento, y hasta dónde puede crecer.
ESPERA_INICIAL = 5.0
ESPERA_MAXIMA = 60.0

#: Si el núcleo aguanta esto encendido, se considera que arrancó bien y la
#: espera vuelve al principio. Sin esto, un núcleo que funciona una semana y
#: luego se cae heredaría la espera larga de un fallo de hace días.
ARRANQUE_BUENO = 120.0


def siguiente_espera(espera: float, vivio: float) -> float:
    """Cuánto esperar antes de volver a arrancar.

    Se dobla mientras el núcleo se caiga rápido, y se reinicia en cuanto uno
    aguanta en pie: lo que se quiere frenar es el bucle de arranques fallidos,
    no el arranque de mañana.
    """
    if vivio >= ARRANQUE_BUENO:
        return ESPERA_INICIAL
    return min(espera * 2, ESPERA_MAXIMA)


def _apuntar(registro: Path, mensaje: str) -> None:
    marca = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linea = f"{marca}  {mensaje}\n"
    try:
        with registro.open("a", encoding="utf-8") as fichero:
            fichero.write(linea)
    except OSError:
        pass
    # También a la salida, por si alguien lo ejecuta a mano desde un terminal.
    print(linea, end="", flush=True)


def _directorio_datos() -> Path:
    ruta = Path(os.environ.get("PERSEO_CORE_DATOS", RAIZ / "perseo_core" / "datos"))
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta


def vigilar() -> int:
    registro = _directorio_datos() / "vigilante.log"
    espera = ESPERA_INICIAL
    _apuntar(registro, f"Vigilante en marcha sobre {RAIZ}.")

    while True:
        comienzo = time.monotonic()
        try:
            proceso = subprocess.Popen(
                [sys.executable, "-m", "perseo_core"],
                cwd=str(RAIZ),
                # El núcleo hereda el entorno, pero lo que de verdad lo
                # configura cuando arranca con Windows es `datos/entorno.json`:
                # una entrada del registro no trae variables de nadie.
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
        except OSError as e:
            _apuntar(registro, f"No se pudo arrancar el núcleo: {e}. Se reintenta en {espera:.0f}s.")
            time.sleep(espera)
            espera = min(espera * 2, ESPERA_MAXIMA)
            continue

        try:
            codigo = proceso.wait()
        except KeyboardInterrupt:
            _apuntar(registro, "Vigilante interrumpido; se para el núcleo.")
            proceso.terminate()
            return 0

        vivio = time.monotonic() - comienzo
        if codigo == 0:
            _apuntar(registro, f"El núcleo terminó bien tras {vivio:.0f}s. El vigilante se retira.")
            return 0

        espera = siguiente_espera(espera, vivio)
        _apuntar(
            registro,
            f"El núcleo murió con código {codigo} tras {vivio:.0f}s. "
            f"Se vuelve a arrancar en {espera:.0f}s.",
        )
        time.sleep(espera)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(vigilar())
