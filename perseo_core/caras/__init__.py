"""Por donde entra y sale el mundo.

La API HTTP con su cola y su flujo de eventos, el canal de Telegram, la pantalla
de estado y la web del móvil (`interfaz/`, un solo fichero sin build).

**Las caras no piensan**: encolan un trabajo y sondean el resultado. Toda
decisión ocurre por debajo. Esta capa puede importar de todas las demás, que es
justo lo que significa estar arriba del todo.
"""
