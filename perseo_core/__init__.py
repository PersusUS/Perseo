"""perseo-core: el núcleo de Perseo v2.

Un solo proceso que decide. Las caras —la app Tauri, la web del móvil, la sesión
de voz— solo transportan entrada y salida; ninguna piensa. Esa separación es lo
que permite mudar el núcleo a la Raspberry Pi más adelante sin reescribir nada.


"""

__all__ = ["almacen", "bus", "agentes", "api"]
