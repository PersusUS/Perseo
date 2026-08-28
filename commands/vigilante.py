"""Mantiene el núcleo encendido, que es lo único que se le pide.

Una entrada en `HKCU\\...\\Run` lanza un proceso **una vez**. Si el núcleo se
cae a las tres horas —Ollama que se lleva la memoria por delante, un disco
lleno, una excepción no prevista— Windows no lo vuelve a levantar y nadie se
entera: no hay cola, no hay triaje, no hay avisos. Y el síntoma es que Perseo
"dejó de hacer cosas", que no lleva a ninguna causa.

Así que en el registro va esto, no el núcleo: un padre que arranca al hijo, se
queda esperando y lo vuelve a arrancar si se muere.

Cuatro reglas:

1. **La espera crece.** Si el núcleo revienta al arrancar —configuración rota,
   puerto ocupado— reintentar cada segundo llena el disco de registro y no
   arregla nada. Se empieza en 5 s y se dobla hasta un minuto.
2. **Una salida limpia es una orden.** Si el núcleo termina con código 0, es que
   alguien lo paró a propósito; el vigilante se va detrás. Solo se reinicia lo
   que se muere mal.
3. **Se deja escrito.** Todo va a `<datos>/vigilante.log`, con fecha, porque
   este proceso no tiene consola: lo lanza `pythonw` desde el registro. Es H-34
   aplicado antes de que muerda.
4. **Lo que diga el núcleo también se guarda**, en `<datos>/nucleo.log`. El
   vigilante arranca sin consola, así que el núcleo hereda un `sys.stderr` que
   vale `None`: su registro se lo traga `logging` sin quejarse y sus trazas no
   aparecen en ninguna parte. Sin esto, `vigilante.log` apunta *que* se murió y
   nadie apunta *por qué* (H-41).

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

sys.path.insert(0, str(Path(__file__).resolve().parent))

import unico  # noqa: E402

#: Cómo se llama la cerradura de esta pieza. Ver `commands/unico.py`.
CERROJO = "PerseoVigilante"

#: Cuánto se espera antes del primer reintento, y hasta dónde puede crecer.
ESPERA_INICIAL = 5.0
ESPERA_MAXIMA = 60.0

#: Si el núcleo aguanta esto encendido, se considera que arrancó bien y la
#: espera vuelve al principio. Sin esto, un núcleo que funciona una semana y
#: luego se cae heredaría la espera larga de un fallo de hace días.
ARRANQUE_BUENO = 120.0

#: Cuánto puede ocupar `nucleo.log` antes de apartarlo. El núcleo escribe una
#: línea por trabajo y esto arranca con Windows: sin tope, un registro que nadie
#: mira se come el disco en unos meses.
TOPE_REGISTRO = 5 * 1024 * 1024


def siguiente_espera(espera: float, vivio: float) -> float:
    """Cuánto esperar antes de volver a arrancar.

    Se dobla mientras el núcleo se caiga rápido, y se reinicia en cuanto uno
    aguanta en pie: lo que se quiere frenar es el bucle de arranques fallidos,
    no el arranque de mañana.
    """
    if vivio >= ARRANQUE_BUENO:
        return ESPERA_INICIAL
    return min(espera * 2, ESPERA_MAXIMA)


def apartar_si_crece(registro: Path, tope: int = TOPE_REGISTRO) -> bool:
    """Aparta el registro a `.viejo` si pasó del tope. Devuelve si lo apartó.

    Se mira antes de cada arranque y no mientras el núcleo escribe: en Windows
    no se puede renombrar un fichero que otro proceso tiene abierto, y el único
    momento en que seguro no lo tiene es justo antes de arrancarlo.
    """
    try:
        if registro.stat().st_size < tope:
            return False
        registro.replace(registro.with_suffix(registro.suffix + ".viejo"))
    except OSError:
        # Un registro que no se puede apartar no es motivo para dejar el núcleo
        # apagado. Se sigue escribiendo en el de siempre.
        return False
    return True


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
    datos = _directorio_datos()
    registro = datos / "vigilante.log"
    registro_nucleo = datos / "nucleo.log"

    # Un vigilante y no dos. Dos vigilantes son dos núcleos peleándose por el
    # 8787: el segundo muere con `OSError 10048`, el segundo vigilante lo toma
    # por una muerte anormal y lo vuelve a arrancar, y así para siempre. La
    # cerradura la suelta el sistema cuando el proceso muere, así que un
    # vigilante colgado no deja a Perseo sin poder arrancar nunca más (H-76).
    cerrojo = unico.tomar(CERROJO)
    if cerrojo is None:
        _apuntar(registro, "Ya hay otro vigilante en marcha; este se retira.")
        return 0

    espera = ESPERA_INICIAL
    _apuntar(registro, f"Vigilante en marcha sobre {RAIZ}.")

    while True:
        comienzo = time.monotonic()
        apartar_si_crece(registro_nucleo)
        try:
            # Sin esto el núcleo escribe en el vacío: lanzado desde aquí hereda
            # un `sys.stderr` que vale `None`, y `logging` se traga el fallo de
            # escritura porque su propio aviso de error va al mismo sitio.
            salida = registro_nucleo.open("a", encoding="utf-8")
        except OSError as e:
            # Preferible un núcleo mudo a un núcleo apagado.
            _apuntar(registro, f"No se pudo abrir {registro_nucleo.name}: {e}. El núcleo arranca sin registro.")
            salida = None

        try:
            if salida is not None:
                marca = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                salida.write(f"\n===== Arranque del núcleo {marca} =====\n")
                salida.flush()
            try:
                proceso = subprocess.Popen(
                    [sys.executable, "-m", "perseo_core"],
                    cwd=str(RAIZ),
                    # El núcleo hereda el entorno, pero lo que de verdad lo
                    # configura cuando arranca con Windows es `datos/entorno.json`:
                    # una entrada del registro no trae variables de nadie.
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                    stdout=salida,
                    # Junto y en orden: una traza se entiende con las líneas de
                    # registro que la rodean, y en dos ficheros no se cruzan.
                    stderr=subprocess.STDOUT if salida is not None else None,
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
        finally:
            # El hijo ya tiene su propio descriptor; este sobra, y dejarlo
            # abierto en cada vuelta va sumando hasta quedarse sin ninguno.
            if salida is not None:
                salida.close()

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


def vigilar_diciendo_como_muere() -> int:
    """`vigilar()`, pero dejando escrito si se muere por una excepción.

    Este proceso no tiene consola: una excepción no prevista lo mataba en el más
    absoluto silencio, y lo que quedaba en `vigilante.log` era su última línea
    normal. Fue exactamente lo que se vio el 2026-08-18: «El núcleo murió…, se
    vuelve a arrancar en 5s», y después nada durante tres días.

    No se reintenta nada aquí a propósito: si el vigilante no puede vigilar, lo
    que toca es que se note. De volver a levantarlo se encarga la tarea
    programada `PerseoRevivir` (`manage_startup.py`), que mira desde fuera.
    """
    registro = _directorio_datos() / "vigilante.log"
    try:
        return vigilar()
    except BaseException as e:  # noqa: BLE001 — se re-lanza abajo
        import traceback

        _apuntar(registro, f"El vigilante se muere por {type(e).__name__}: {e}")
        _apuntar(registro, traceback.format_exc())
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(vigilar_diciendo_como_muere())
