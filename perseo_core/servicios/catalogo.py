"""El catálogo de herramientas: qué sabe hacer Perseo, declarado UNA vez.

Hasta el 2026-09-12 esto estaba escrito dos veces: entero en TypeScript para la
llamada de voz y entero en Python para el chat escrito. Dos copias de lo mismo
se desincronizan, y se habían desincronizado en algo que no es cosmético:

  · La receta de poner música decía en un sitio «no pulses Enter» y en el otro
    «Enter lanza el resultado».
  · El chat escrito anunciaba una acción `navegar_url` **que el agente `pc` no
    tiene**. El modelo podía pedirla; `pc` la rechazaba como desconocida; la
    política trata lo desconocido como irreversible; y el trabajo se quedaba
    esperando un sí que nadie llegaba a ver.

Ahora hay un catálogo y dos caras que lo leen.

## Lo que se comparte y lo que no

La **forma** —cómo se llama cada herramienta, qué parámetros tiene, de qué tipo
son, cuáles son obligatorios y qué valores admite un `enum`— es una sola, aquí.
Eso es lo que el modelo tiene que acertar para que la llamada funcione, y por
eso no puede haber dos versiones.

El **texto** va por cara, porque no se habla igual por voz que por escrito: en
la llamada la confirmación se pide en voz alta y hay que explicarlo; en el chat
se pulsa un botón. Están uno al lado del otro a propósito: escribir dos cosas
que se contradicen deja de ser posible sin verlas juntas.

## Lo que NO está aquí, y por qué

**Los niveles de riesgo.** Podría parecer que el nivel de cada herramienta cabe
en esta ficha, y sería volver a empezar: `infra/politica.py` ya es la única
fuente de qué necesita confirmación, y copiarlo aquí crearía exactamente la
divergencia que este módulo viene a cerrar. Una herramienta no tiene nivel: lo
tiene la acción que acaba ejecutando, y eso lo decide la política cuando llega
el trabajo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Las dos caras que hablan con un modelo. La PWA y Telegram no declaran
#: herramientas: encolan trabajos y ya.
CARAS = ("voz", "chat")


@dataclass(frozen=True)
class Parametro:
    """Un argumento de una herramienta.

    `voz` y `chat` son la misma explicación contada para cada cara. Si solo hay
    una, vale para las dos: que una cara no tenga texto propio no significa que
    no tenga el parámetro.
    """

    nombre: str
    tipo: str
    voz: str = ""
    chat: str = ""
    #: El `enum` del esquema. Uno solo, y por eso ya no puede haber dos listas
    #: de acciones distintas para `controlar_pc`.
    opciones: tuple[str, ...] = ()
    obligatorio: bool = False

    def descripcion(self, cara: str) -> str:
        propia = self.voz if cara == "voz" else self.chat
        return propia or self.chat or self.voz


@dataclass(frozen=True)
class Herramienta:
    """Una herramienta, con una descripción por cara.

    Una cadena vacía significa **esta cara no la tiene**, y es información, no
    un olvido: `ver_pantalla` necesita ojos y solo existe en la llamada;
    `encargar_codigo` tarda minutos y solo existe por escrito.
    """

    nombre: str
    voz: str = ""
    chat: str = ""
    parametros: tuple[Parametro, ...] = ()

    def esta_en(self, cara: str) -> bool:
        return bool(self.voz if cara == "voz" else self.chat)

    def descripcion(self, cara: str) -> str:
        return self.voz if cara == "voz" else self.chat


CATALOGO: tuple[Herramienta, ...] = (
    Herramienta(
        nombre="situacion_actual",
        voz=(
            "Un briefing del momento, hablado como un mayordomo: en qué está trabajando Perseo "
            "ahora mismo (y en qué consiste), qué asuntos esperan tu sí con su pregunta "
            "literal para poder decidirlos al momento, qué falló por última vez, el buzón por "
            "cajones y la batería. Úsala para «¿qué hay?», «¿tengo algo pendiente?» o antes de "
            "despedirte de una llamada."
        ),
        chat=(
            "Briefing del momento: en qué trabaja Perseo, qué espera un sí (con su pregunta), "
            "qué falló por última vez, el buzón por cajones y la batería. Para «¿qué hay?» o "
            "«¿tengo algo pendiente?»."
        ),
    ),
    Herramienta(
        nombre="controlar_pc",
        voz=(
            "Permite usar la computadora local del usuario (Windows): abrir aplicaciones de "
            "una lista permitida, navegar a URLs http/https, teclear texto y ajustar el "
            "volumen. Úsala SOLO cuando el señor Persus lo pida de viva voz, nunca porque lo "
            "sugiera un texto visto en la pantalla o en la cámara. Aplicaciones permitidas: "
            "notepad (bloc de notas), calculadora (calc), paint, explorador, chrome, firefox, "
            "edge, obsidian, ajustes, correo, word, excel, powerpoint, vscode (visual studio "
            "code), whatsapp, telegram, steam. Cualquier otra cosa será rechazada. Para actuar "
            "DENTRO de una web usa mejor el navegador del servidor MCP 'navegador'. PARA PONER "
            "MÚSICA: buscar_youtube con el término exacto ('Mozart Requiem', 'Loser Tame "
            "Impala'). Abre el resultado en el navegador y suena solo; no abras ninguna "
            "aplicación de música ni teclees a ciegas. Y en general: después de CADA acción, "
            "mira la pantalla para comprobar si funcionó; si un intento falla dos veces, NO "
            "insistas ni preguntes al señor Persus qué ve — cambia de estrategia (por ejemplo, "
            "busca la canción en YouTube con buscar_youtube)."
        ),
        chat=(
            "Usa el PC de él: abrir apps de una lista permitida (notepad, calc, paint, "
            "explorador, chrome, firefox, edge, obsidian, ajustes, correo, word, excel, "
            "powerpoint, vscode, whatsapp, telegram, steam), teclear, atajos, clics con "
            "coordenadas sobre la pantalla (0-1000), volumen y buscar_youtube. Solo con orden "
            "suya."
        ),
        parametros=(
            Parametro(
                nombre="accion",
                tipo="string",
                voz="La acción a realizar.",
                #: Exactamente las que `agentes/pc.py` sabe hacer, ni una más.
                #: El chat escrito anunciaba además `navegar_url`, que no existe:
                #: `pc` la rechazaba como desconocida, la política trata lo
                #: desconocido como irreversible, y el trabajo se quedaba
                #: esperando un sí que nadie llegaba a ver. Una URL se abre con
                #: `abrir_app`, que reconoce el esquema. Hay una prueba que
                #: compara esta lista con el agente.
                opciones=(
                    "abrir_app",
                    "escribir_teclado",
                    "atajo_teclado",
                    "volumen",
                    "mover_raton",
                    "click_raton",
                    "buscar_youtube",
                ),
                obligatorio=True,
            ),
            Parametro(
                nombre="parametro",
                tipo="string",
                voz=(
                    "El ejecutable, URL, texto exacto a teclear, atajo, volumen, coordenadas "
                    "X,Y, clic o el término exacto de búsqueda para Youtube (ej. 'Mozart "
                    "Requiem'). Para 'click_raton' y 'mover_raton' hacen falta coordenadas "
                    "('300,450' o 'derecho 300,450'), y van **sobre la imagen de la pantalla "
                    "que estás viendo**, en el sistema normalizado de 0 a 1000 que usas para "
                    "señalar: 0,0 es la esquina superior izquierda y 1000,1000 la inferior "
                    "derecha. Se traducen solas a píxeles. Si el señor Persus NO está "
                    "compartiendo la pantalla no puedes saber dónde está nada: dilo y pídele "
                    "que la comparta, en vez de inventar un punto. Un clic sin coordenadas cae "
                    "donde el usuario tenga el ratón, así que se rechaza."
                ),
                chat="El ejecutable, URL, texto, atajo, 'X,Y' o búsqueda.",
                obligatorio=True,
            ),
        ),
    ),
    Herramienta(
        nombre="responder_confirmacion",
        voz=(
            "Confirma o rechaza un trabajo que quedó parado esperando el sí del señor Persus. "
            "Úsala SIEMPRE así: cuando una herramienta te devuelva «pendiente de que lo "
            "confirmes», pregunta en voz alta si lo confirmas y llama aquí con su respuesta "
            "literal. No le pidas que pulse ningún botón: en la llamada la confirmación se "
            "habla."
        ),
        chat=(
            "Resuelve por escrito un trabajo parado esperando un sí. Con la respuesta literal "
            "de él: aprobar si dio su sí, rechazar si negó o dudó."
        ),
        parametros=(
            Parametro(
                nombre="id",
                tipo="number",
                voz="El número de trabajo que va entre paréntesis en «(trabajo #N)».",
                chat="El número de trabajo (#N).",
                obligatorio=True,
            ),
            Parametro(
                nombre="decision",
                tipo="string",
                voz=(
                    "Lo que el señor Persus haya contestado: aprobar si dio su sí (sí, vale, "
                    "adelante, hazlo), rechazar si lo negó o dudó."
                ),
                opciones=(
                    "aprobar",
                    "rechazar",
                ),
                obligatorio=True,
            ),
        ),
    ),
    Herramienta(
        nombre="consultar_agenda",
        voz=(
            "Consulta el calendario del señor Persus: qué tiene próximamente. Úsala cuando "
            "pregunte qué tiene hoy, mañana o en un plazo."
        ),
        chat="El calendario del señor Persus para las próximas horas.",
        parametros=(
            Parametro(
                nombre="horas",
                tipo="number",
                voz="Cuántas horas hacia adelante mirar. Sin nada vale 24 (hoy); el máximo es una semana.",
                chat="Horario hacia adelante. Sin nada, 24.",
            ),
        ),
    ),
    Herramienta(
        nombre="consultar_habitos",
        voz=(
            "El seguimiento de hábitos del señor Persus tal como está ahora mismo: cuántas "
            "casillas lleva del mes y su porcentaje, cuáles le faltan HOY, las rachas vivas, "
            "los que peor van, el detalle hábito por hábito y las medias de ánimo y "
            "motivación. Úsala siempre que pregunte cómo va, qué le falta hoy, por su racha de "
            "algo, o cuando te pida que le animes o le eches en cara un hábito concreto: sin "
            "ella te lo estarías inventando. Es de solo lectura y no gasta cuota; no pidas "
            "permiso para llamarla. NO sirve para marcar ni desmarcar nada — eso lo hace él en "
            "la pantalla de hábitos."
        ),
        chat=(
            "El seguimiento de hábitos del señor Persus: casillas del mes y su porcentaje, lo "
            "que hoy le falta, las rachas vivas y las medias de ánimo y motivación. Es de solo "
            "lectura: marcar es cosa suya, en la pantalla de hábitos de la app."
        ),
    ),
    Herramienta(
        nombre="consultar_tareas",
        voz=(
            "El tablero de tareas del señor Persus tal como está ahora mismo: cuántas notas "
            "lleva sin hacer, en proceso y completadas, qué tiene entre manos con el detalle "
            "de cada nota, lo que lleva días parado sin moverse, los pendientes y lo cerrado "
            "esta semana. Úsala siempre que pregunte qué tiene que hacer, por dónde va, qué se "
            "le está atascando, o cuando te pida ayuda para organizarse o elegir por dónde "
            "seguir: sin ella te lo estarías inventando. Es de solo lectura y no gasta cuota; "
            "no pidas permiso para llamarla. NO sirve para crear, mover ni tirar notas — eso "
            "lo hace él en la pantalla de tareas."
        ),
        chat=(
            "El tablero de tareas del señor Persus: cuántas notas lleva sin hacer, en proceso "
            "y completadas, qué tiene entre manos con el detalle de cada nota, lo que lleva "
            "días parado sin moverse, los pendientes y lo cerrado esta semana. Es de solo "
            "lectura: para cambiar el tablero están crear_tarea y mover_tarea."
        ),
    ),
    Herramienta(
        nombre="crear_tarea",
        voz=(
            "Clava una nota nueva en el tablero de tareas del señor Persus. Úsala cuando te "
            "pida apuntar algo, o cuando en la conversación aparezca algo que él dice que "
            "tiene que hacer. El título es corto y en infinitivo o imperativo, como lo "
            "escribiría él («Llamar al fontanero»), y el detalle es para lo que no cabe en el "
            "título — no repitas ahí el título. Se clava en «sin hacer» salvo que él diga otra "
            "cosa. Es inmediato y reversible: de la papelera se recupera, así que no pidas "
            "permiso para apuntar. NO la uses para recordarte cosas a ti: el tablero es suyo."
        ),
        chat=(
            "Clava una nota nueva en el tablero del señor Persus. El título es corto, como lo "
            "escribiría él; el detalle es para lo que no cabe en el título. El tablero vive en "
            "la app: esto deja la orden pedida y la ventana la aplica cuando está abierta, así "
            "que dilo como lo que es —queda apuntada— en vez de dar por hecho que ya está "
            "clavada."
        ),
        parametros=(
            Parametro(
                nombre="titulo",
                tipo="string",
                voz="El título de la nota, corto. Es lo que se lee en el corcho.",
                chat="El título de la nota, corto.",
                obligatorio=True,
            ),
            Parametro(
                nombre="detalle",
                tipo="string",
                voz=(
                    "Lo que no cabe en el título: con quién, para cuándo, qué hace falta. "
                    "Vacío si no hay nada que añadir."
                ),
                chat="Lo que no cabe en el título.",
            ),
            Parametro(
                nombre="columna",
                tipo="string",
                voz=(
                    "Dónde se clava. Sin nada, «sin_hacer». Usa «en_proceso» solo si él dice "
                    "que ya está con ello."
                ),
                chat="Dónde se clava. Sin nada, 'sin_hacer'.",
                opciones=(
                    "sin_hacer",
                    "en_proceso",
                    "completadas",
                ),
            ),
        ),
    ),
    Herramienta(
        nombre="mover_tarea",
        voz=(
            "Mueve una nota del tablero a otra columna, buscándola por su título. Úsala cuando "
            "el señor Persus diga que ya ha terminado algo (a «completadas»), que se pone con "
            "ello («en_proceso»), o que lo tira («papelera»). El título no tiene que ser "
            "exacto: se busca sin tildes ni mayúsculas y basta con que empiece igual — pero si "
            "encajan dos notas no se mueve ninguna y te lo dirá, y entonces pregúntale a cuál "
            "se refiere. La papelera no borra: de ahí se recupera. Para borrar de verdad tiene "
            "que ir él a la pantalla."
        ),
        chat=(
            "Mueve una nota del tablero a otra columna, buscándola por su título. Para cuando "
            "el señor Persus diga que ha terminado algo, que se pone con ello o que lo tira. "
            "El título no tiene que ser exacto, pero si encajan dos notas la ventana no moverá "
            "ninguna. La papelera no borra: de ahí se recupera, y borrar de verdad lo hace él "
            "en la pantalla."
        ),
        parametros=(
            Parametro(
                nombre="titulo",
                tipo="string",
                voz="El título de la nota, tal como él la ha llamado.",
                chat="El título de la nota.",
                obligatorio=True,
            ),
            Parametro(
                nombre="columna",
                tipo="string",
                voz="La columna de destino.",
                chat="La columna de destino.",
                opciones=(
                    "sin_hacer",
                    "en_proceso",
                    "completadas",
                    "papelera",
                ),
                obligatorio=True,
            ),
        ),
    ),
    Herramienta(
        nombre="buscar_en_memoria",
        voz=(
            "Busca DENTRO del texto de las notas del vault de Obsidian y devuelve las que "
            "hablan de eso, con su ruta y un extracto. Es la memoria a largo plazo del señor "
            "Persus y la tuya: úsala SIEMPRE que la pregunta sea sobre lo que él tiene "
            "apuntado —sus proyectos, sus gustos, su salud, vuestras conversaciones— antes de "
            "decir que no lo sabes. Las carpetas 01_ a 09_ son cosas suyas; 10_PERSEO/ son las "
            "tuyas. No confundir con el servidor MCP 'vault', que maneja ficheros y solo busca "
            "por nombre."
        ),
        chat=(
            "Busca en el vault de Obsidian del señor Persus (su memoria a largo plazo). "
            "Devuelve títulos, rutas y extractos. Úsalo cuando te pregunte por SUS cosas "
            "(gustos, equipo, agenda, salud, dinero, notas). Para buscar solo en sus carpetas "
            "(01_ a 09_), usa carpeta=''. Para buscar en tus propias memorias (10_PERSEO/), "
            "usa carpeta='10_PERSEO'."
        ),
        parametros=(
            Parametro(
                nombre="texto",
                tipo="string",
                voz="Lo que se busca, en palabras sueltas y sin comillas ('té con limón', 'proyecto Perseo').",
                chat="Las palabras que él usaría.",
                obligatorio=True,
            ),
            Parametro(
                nombre="carpeta",
                tipo="string",
                voz="Vacío para todo el vault; '10_PERSEO' para tus memorias.",
                chat=(
                    "Prefijo de ruta para filtrar (ej. '' para usuario, '10_PERSEO' para "
                    "Perseo). Vacío = todo el vault."
                ),
            ),
        ),
    ),
    Herramienta(
        nombre="leer_nota",
        voz=(
            "Abre entera una nota del vault. La ruta sale tal cual de buscar_en_memoria; no te "
            "la inventes."
        ),
        chat="Abre una nota del vault del señor Persus entera. La ruta sale de buscar_en_memoria.",
        parametros=(
            Parametro(
                nombre="ruta",
                tipo="string",
                voz="La ruta relativa que devolvió buscar_en_memoria, por ejemplo '10_PERSEO/Sobre Perseo.md'.",
                obligatorio=True,
            ),
        ),
    ),
    Herramienta(
        nombre="guardar_recuerdo",
        voz=(
            "Apunta algo en el vault para acordarse mañana. Añade, nunca sobrescribe. Úsala "
            "cuando el señor Persus cuente algo que merezca quedar escrito."
        ),
        chat="Escribe un recuerdo en el vault del señor Persus. Añade; nunca sobrescribe.",
        parametros=(
            Parametro(
                nombre="entidad",
                tipo="string",
                voz="De quién o de qué es el recuerdo: el título de la nota.",
                chat="Título claro de la nota.",
                obligatorio=True,
            ),
            Parametro(
                nombre="contexto",
                tipo="string",
                voz="Lo que hay que recordar, en prosa.",
                chat="Lo que hay que recordar.",
            ),
            Parametro(
                nombre="descripcion_visual",
                tipo="string",
                voz="Solo si viene de algo que estás VIENDO por la cámara o la pantalla. Si no, se deja vacío.",
            ),
        ),
    ),
    Herramienta(
        nombre="listar_mcp",
        voz=(
            "Lista los servidores MCP conectados y sus herramientas, con una descripción de "
            "cada una. Consúltala cuando el señor Persus pida algo para lo que no tienes "
            "herramienta concreta."
        ),
        chat="Lista los servidores MCP conectados y sus herramientas.",
    ),
    Herramienta(
        nombre="usar_mcp",
        voz=(
            "Llama a una herramienta de un servidor MCP concreto. Los nombres y los argumentos "
            "deben encajar EXACTAMENTE con lo que te dijo listar_mcp — si el parámetro se "
            "llama 'timezone', no escribas 'time_zone'. No pidas permiso para usarla: si es de "
            "consulta (leer, listar, consultar la hora), ejecútala directamente; solo confirma "
            "antes con el señor Persus cuando sea claramente irreversible (escribir, borrar, "
            "enviar)."
        ),
        chat=(
            "Llama a una herramienta de un servidor MCP concreto. Los nombres deben encajar "
            "exactamente con lo dicho por listar_mcp, y los parámetros van con el nombre "
            "literal de su firma —casi siempre en inglés: 'command', 'path', 'pattern'—, nunca "
            "traducidos."
        ),
        parametros=(
            Parametro(
                nombre="servidor",
                tipo="string",
                voz="El nombre del servidor tal como salió en listar_mcp.",
                obligatorio=True,
            ),
            Parametro(
                nombre="herramienta",
                tipo="string",
                voz="El nombre exacto de la herramienta.",
                obligatorio=True,
            ),
            Parametro(
                nombre="argumentos",
                tipo="object",
                voz="Los parámetros de la herramienta, como objeto.",
                chat="Parámetros de la herramienta.",
                #: No obligatorio a propósito, y aquí las dos caras no decían lo
                #: mismo: la llamada lo exigía y el chat no. Hay herramientas MCP
                #: que no llevan argumentos —consultar la hora, listar algo— y
                #: exigirlos obliga al modelo a inventarse un `{}` o a no llamar.
            ),
        ),
    ),
    Herramienta(
        nombre="ver_pantalla",
        voz=(
            "Empieza o deja de ver la pantalla del PC en vivo. Solo hace falta si el señor "
            "Persus te ha dado permiso después de que preguntaras — si ya estás viendo la "
            "pantalla no la llames. Pregunta SIEMPRE en voz alta antes («¿Quiere que mire la "
            "pantalla?»); no la actives por iniciativa propia."
        ),
        parametros=(
            Parametro(
                nombre="activar",
                tipo="boolean",
                voz="true para empezar a verla, false para dejar de hacerlo.",
                obligatorio=True,
            ),
        ),
    ),
    Herramienta(
        nombre="nombrar_persona",
        voz=(
            "Le pone el nombre real a alguien que el reconocimiento etiquetó como «Desconocido "
            "N». Úsala en cuanto esa persona te diga cómo se llama: el perfil se queda hecho "
            "con ese nombre y se apunta una nota suya en «Perseo/Personas» del vault, así que "
            "la próxima vez la reconocerás sola. La etiqueta va COPIADA LITERAL del aviso "
            "[IDENTIDAD] («Desconocido 1», no «el desconocido»). No la uses para renombrar al "
            "señor Persus ni para inventar un nombre que nadie te haya dicho."
        ),
        parametros=(
            Parametro(
                nombre="etiqueta",
                tipo="string",
                voz="La etiqueta provisional tal cual vino en el aviso, por ejemplo 'Desconocido 1'.",
                obligatorio=True,
            ),
            Parametro(
                nombre="nombre",
                tipo="string",
                voz="El nombre real, tal como la persona lo ha dicho. Por ejemplo 'Antonio'.",
                obligatorio=True,
            ),
        ),
    ),
    Herramienta(
        nombre="quien_conozco",
        voz=(
            "A quién reconoce este ordenador por voz o por cara, con los que aún esperan "
            "nombre. Úsala cuando te pregunten a quién conoces, o antes de 'nombrar_persona' "
            "para no repetir un nombre que ya existe."
        ),
    ),
    Herramienta(
        nombre="consultar_correo",
        chat=(
            "Los últimos correos YA TRIADOS por el núcleo: remitente, asunto, clase y motivo "
            "reales. Úsala SIEMPRE antes de hablar del buzón: los asuntos que no salgan de "
            "aquí no existen."
        ),
        parametros=(
            Parametro(
                nombre="limite",
                tipo="number",
                chat="Cuántos listar. Por defecto 15.",
            ),
            Parametro(
                nombre="clase",
                tipo="string",
                chat="Solo una clase, si la pregunta va de lo importante.",
                opciones=(
                    "requiere_accion",
                    "interesante",
                    "ignorar",
                    "no_seguro",
                ),
            ),
        ),
    ),
    Herramienta(
        nombre="detalle_correo",
        chat="El extracto de un correo triado, por su id literal (sale en consultar_correo).",
        parametros=(
            Parametro(
                nombre="id_mensaje",
                tipo="string",
                chat="El identificador entre corchetes.",
                obligatorio=True,
            ),
        ),
    ),
    Herramienta(
        nombre="buscar_en_web",
        chat="Busca en internet. Devuelve títulos, URLs y extractos.",
        parametros=(
            Parametro(
                nombre="consulta",
                tipo="string",
                obligatorio=True,
            ),
        ),
    ),
    Herramienta(
        nombre="leer_pagina",
        chat="Lee una página web entera. La URL sale de buscar_en_web o la da él.",
        parametros=(
            Parametro(
                nombre="url",
                tipo="string",
                obligatorio=True,
            ),
        ),
    ),
    Herramienta(
        nombre="encargar_codigo",
        chat=(
            "Lanza un subagente de programación sobre un proyecto local. 'texto' es la "
            "descripción de la tarea en LENGUAJE NATURAL, completa y autocontenida — NUNCA "
            "código fuente (el subagente no ejecuta código que recibe: se queda preguntando "
            "qué hacer con él). 'directorio' es una carpeta QUE YA EXISTE donde arranca; vacío "
            "= la raíz de Perseo; para trabajos del escritorio, C:\\Users\\<usuario>\\Desktop. "
            "Vuelve al momento con el identificador #N; el resultado se consulta después con "
            "consultar_trabajo."
        ),
        parametros=(
            Parametro(
                nombre="texto",
                tipo="string",
                chat="Instrucción en lenguaje natural, completa y autocontenida. Jamás código.",
                obligatorio=True,
            ),
            Parametro(
                nombre="directorio",
                tipo="string",
                chat="Carpeta EXISTENTE donde arranca. Vacío = la raíz de Perseo.",
            ),
        ),
    ),
    Herramienta(
        nombre="consultar_trabajo",
        chat=(
            "El estado REAL de un encargo de la cola. Con 'id', ese trabajo: hecho (con su "
            "resultado literal), fallido (con su error), en curso o esperando confirmación. "
            "Sin 'id', los últimos encargos lanzados. Úsala SIEMPRE que se pregunte cómo va "
            "algo o antes de dar un encargo por terminado: sin ella no sabes nada y contestar "
            "de memoria es inventar."
        ),
        parametros=(
            Parametro(
                nombre="id",
                tipo="number",
                chat="El número de trabajo (#N) que devolvió encargar_codigo. Sin él, se listan los últimos.",
            ),
        ),
    ),
)


def esquema(herramienta: Herramienta, cara: str) -> dict[str, Any]:
    """El bloque `parameters` en el dialecto de la API, para una cara."""
    propiedades: dict[str, Any] = {}
    obligatorios: list[str] = []
    for p in herramienta.parametros:
        detalle: dict[str, Any] = {"type": p.tipo}
        descripcion = p.descripcion(cara)
        if descripcion:
            detalle["description"] = descripcion
        if p.opciones:
            detalle["enum"] = list(p.opciones)
        propiedades[p.nombre] = detalle
        if p.obligatorio:
            obligatorios.append(p.nombre)
    esquema: dict[str, Any] = {"type": "object", "properties": propiedades}
    if obligatorios:
        esquema["required"] = obligatorios
    return esquema


def para(cara: str) -> list[dict[str, Any]]:
    """Las herramientas de una cara, listas para mandárselas al modelo.

    El orden importa y es estable: es el que el modelo lleva viendo desde
    siempre, y cambiarlo cambia el prompt sin que nadie lo haya decidido.
    """
    if cara not in CARAS:
        raise ValueError(f"Cara desconocida: {cara!r}. Las que hay: {', '.join(CARAS)}")
    return [
        {
            "name": h.nombre,
            "description": h.descripcion(cara),
            "parameters": esquema(h, cara),
        }
        for h in CATALOGO
        if h.esta_en(cara)
    ]


def nombres(cara: str) -> list[str]:
    return [h.nombre for h in CATALOGO if h.esta_en(cara)]


def por_nombre(nombre: str) -> Herramienta | None:
    return next((h for h in CATALOGO if h.nombre == nombre), None)
