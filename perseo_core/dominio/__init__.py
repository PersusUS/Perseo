"""La capa de abajo: los tipos y el vocabulario, sin depender de nada.

`dominio` no importa nada del resto del paquete. Esa es toda la regla, y
`pruebas/test_arquitectura.py` la comprueba.

Está aquí porque sin ella había un ciclo: `servicios/google_api` construía
`Evento` y `Mensaje`, que vivían dentro de los agentes `agenda` y `correo`, que
a su vez necesitaban a Google. Python no admite eso al arrancar, así que los dos
agentes importaban el módulo de Google **dentro de una función** — un truco que
funcionaba y que escondía el problema en vez de arreglarlo. Con los tipos aquí
abajo, cada uno importa hacia donde debe y no hay nada que esquivar.
"""
