"""perseo-core: el núcleo de Perseo v2.

Un solo proceso que decide. Las caras —la app Tauri, la web del móvil, la sesión
de voz— solo transportan entrada y salida; ninguna piensa. Esa separación es lo
que permite mudar el núcleo a la Raspberry Pi más adelante sin reescribir nada.

Ver bitacora/05_PLAN_PERSEO_V2.md.
"""

__all__ = ["almacen", "bus", "agentes", "api"]
