"""Lo que sostiene al resto y no sabe de agentes.

La cola y su base de datos (`almacen`), el bus de eventos, el reloj que dispara
los trabajos periódicos, el prompt compartido, la política de confirmaciones y
el router que reparte cada trabajo al agente que le toca.

Puede importar `dominio`, y nada más del paquete. La regla la comprueba
`pruebas/test_arquitectura.py`.
"""
