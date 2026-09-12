"""Los que atienden un trabajo de la cola.

Uno por fichero y el fichero se llama como el agente, que es la convención de
todo el repositorio. Cada uno se apunta en el registro con `registrar`, y el
router de `infra` decide a cuál va cada cosa.

No confundir con `infra/router.py`, que es **el registro y el despachador**: eso
se llamaba `agentes.py` y por eso este nombre estaba ocupado.

Puede importar `dominio`, `infra` y `servicios`.
"""
