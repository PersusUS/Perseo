"""Lo que `vulture` marca como muerto y no lo está, con el porqué de cada uno.

`vulture` lee el código sin ejecutarlo y dice qué no llama nadie. Acierta casi
siempre, y cuando se equivoca es por lo mismo: algo a lo que llaman **desde
fuera de Python** —un decorador que lo apunta en un registro, una biblioteca que
busca un método por su nombre, pytest resolviendo un fixture—.

Esta lista es la respuesta a esas equivocaciones. Cada entrada dice quién llama
de verdad a esa cosa. **Una entrada sin porqué no vale**: si nadie sabe explicar
por qué algo sigue aquí, la respuesta correcta es borrarlo, que es justo lo que
esta herramienta viene a provocar.

    python -m vulture perseo_core commands verificadores pruebas \
        verificadores/vulture_permitidos.py --min-confidence 60

Este fichero no se importa ni se ejecuta: solo lo lee `vulture`.
"""

# --------------------------------------------------------------------------- #
# Agentes: los llama el registro, no una llamada escrita en ninguna parte
# --------------------------------------------------------------------------- #
# `@registrar("pc")` los mete en `REGISTRO` y el router los busca por su nombre.
# Que no haya ninguna llamada literal es precisamente el diseño.
_pc
_eco
_simulacro

# --------------------------------------------------------------------------- #
# Callbacks de bibliotecas: el nombre es el contrato
# --------------------------------------------------------------------------- #
# `BaseHTTPRequestHandler` despacha por el nombre del método, y `log_message` se
# sobreescribe vacío para que los servidores de mentira no ensucien la salida de
# los verificadores.
_.do_GET
_.do_POST
_.do_PUT
_.log_message
_.handle_error

# `sqlite3` y `socketserver` leen estos atributos de la instancia.
_.row_factory
_.daemon_threads

# --------------------------------------------------------------------------- #
# Fixtures y funciones que pytest resuelve por su nombre
# --------------------------------------------------------------------------- #
politica_limpia
sin_memoria
_limpio
todos_instalados
# `armado` enciende las confirmaciones mientras dura una prueba del
# trabajador (ver ADR 0005). Se pide por el nombre del parámetro y no se
# usa dentro del cuerpo: el efecto es el monkeypatch, no un valor.
armado

# Funciones que se declaran dentro de una prueba solo para ver qué pasa al
# declararlas: el `@registrar` duplicado que tiene que fallar, el disparador que
# choca. Nunca se llaman, y ese es el caso de prueba.
_otro

# --------------------------------------------------------------------------- #
# Campos de dataclass que solo viajan serializados
# --------------------------------------------------------------------------- #
# Los rellena quien construye el objeto y salen por `asdict`; nadie los lee con
# un punto delante, pero se guardan en la base y se enseñan en la pantalla.
fin
fecha
color

# --------------------------------------------------------------------------- #
# Parámetros que impone una firma ajena
# --------------------------------------------------------------------------- #
# `sounddevice` llama al callback con cuatro argumentos, `aiohttp` con los suyos
# y `socket.getaddrinfo` devuelve tuplas de cinco. No usarlos no es olvido: es
# que la firma no la elegimos nosotros.
frames
time_info
family
params
duration

# El detalle de una excepción que se captura para decidir, no para imprimir.
traza
