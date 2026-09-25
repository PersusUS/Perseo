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

from typing import Any

from .catalogo_recados import RECADOS
from .catalogo_tipos import Herramienta, Parametro

#: Las dos caras que hablan con un modelo. La PWA y Telegram no declaran
#: herramientas: encolan trabajos y ya.
CARAS = ("voz", "chat")


_BASE: tuple[Herramienta, ...] = (
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
        nombre="parte_del_dia",
        voz=(
            "El parte del día de una vez: lo que queda hoy en la agenda, los correos que "
            "piden algo, los recordatorios de hoy y cómo van las tareas y los hábitos. "
            "Úsala cuando pregunte «¿qué tengo hoy?», «¿cómo va el día?» o te pida el parte. "
            "Cuéntalo resumido y por orden de urgencia, no lo leas entero. Si una sección "
            "dice que no se pudo mirar, dilo: no es un día vacío. Solo lectura."
        ),
        chat=(
            "El parte del día: agenda de hoy, correos que piden algo, recordatorios de hoy, "
            "tareas y hábitos. Resúmelo por urgencia; si una sección no se pudo mirar, dilo."
        ),
    ),
    Herramienta(
        nombre="crear_recordatorio",
        voz=(
            "Apunta un recordatorio y Perseo avisará a esa hora: por el móvil, y en la "
            "cola del panel. Úsala cuando el señor Persus diga «recuérdame», «avísame» o "
            "«que no se me olvide» con una hora o un plazo. Di cuándo de UNA forma: "
            "en_minutos para «dentro de veinte minutos»; hora con dias o dia_semana para "
            "«mañana a las nueve» (hora 09:00, dias 1) o «el jueves a las cinco» (hora "
            "17:00, dia_semana jueves) —no hace falta que sepas la fecha de hoy—; o "
            "fecha_hora solo para una fecha concreta. La respuesta repite la hora ya "
            "resuelta: DILA tal cual, para que él oiga si se entendió mal. Es reversible y "
            "no gasta cuota: no pidas permiso para apuntar. NO la uses para cosas que haya "
            "que hacer sin hora: eso es crear_tarea."
        ),
        chat=(
            "Apunta un recordatorio que avisará a esa hora (móvil y panel). Cuándo, de una "
            "forma: en_minutos; hora con dias o dia_semana («mañana a las 9» es hora 09:00 "
            "y dias 1); o fecha_hora en hora local (AAAA-MM-DDTHH:MM) para una fecha "
            "concreta. Repite la hora que devuelve, que es la ya resuelta. Para lo que no "
            "tiene hora está crear_tarea."
        ),
        parametros=(
            Parametro(
                nombre="texto",
                tipo="string",
                voz="Qué hay que recordarle, corto y como lo diría él («llamar al fontanero»).",
                chat="Qué hay que recordar, corto.",
                obligatorio=True,
            ),
            Parametro(
                nombre="en_minutos",
                tipo="number",
                voz="Dentro de cuántos minutos. Para plazos cortos: «en media hora» son 30.",
                chat="Dentro de cuántos minutos.",
            ),
            Parametro(
                nombre="hora",
                tipo="string",
                voz="A qué hora, HH:MM en 24 horas. Sola es la próxima vez que llega esa hora.",
                chat="A qué hora, HH:MM. Sola es la próxima vez que llega.",
            ),
            Parametro(
                nombre="dias",
                tipo="number",
                voz="Con hora: dentro de cuántos días. 0 hoy, 1 mañana, 2 pasado mañana.",
                chat="Con hora: dentro de cuántos días (0 hoy, 1 mañana).",
            ),
            Parametro(
                nombre="dia_semana",
                tipo="string",
                voz="Con hora: qué día de la semana. Si es hoy, el de la semana que viene.",
                chat="Con hora: qué día de la semana.",
                opciones=("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"),
            ),
            Parametro(
                nombre="fecha_hora",
                tipo="string",
                voz="Solo para una fecha concreta: AAAA-MM-DDTHH:MM, hora local.",
                chat="Una fecha concreta: AAAA-MM-DDTHH:MM, hora local.",
            ),
            Parametro(
                nombre="repetir",
                tipo="string",
                voz="Si se repite. Sin nada, una sola vez.",
                chat="Si se repite. Sin nada, una sola vez.",
                opciones=("nunca", "diario", "laborables", "semanal"),
            ),
        ),
    ),
    Herramienta(
        nombre="consultar_recordatorios",
        voz=(
            "Los recordatorios pendientes, con su hora. Úsala cuando pregunte qué tiene "
            "apuntado o antes de quitar uno. Es de solo lectura."
        ),
        chat="Los recordatorios pendientes, con su hora. Solo lectura.",
    ),
    Herramienta(
        nombre="cancelar_recordatorio",
        voz=(
            "Quita un recordatorio pendiente, buscándolo por el principio de su texto. Si "
            "encajan dos no quita ninguno y te dirá cuáles: entonces pregúntale a cuál se "
            "refiere."
        ),
        chat=(
            "Quita un recordatorio pendiente por el principio de su texto. Si encajan dos no "
            "quita ninguno."
        ),
        parametros=(
            Parametro(
                nombre="texto",
                tipo="string",
                voz="El texto del recordatorio, o su principio.",
                chat="El texto del recordatorio, o su principio.",
                obligatorio=True,
            ),
        ),
    ),
    Herramienta(
        nombre="redactar_borrador",
        voz=(
            "Deja un BORRADOR de correo en el Gmail del señor Persus; no lo envía. Úsala "
            "cuando te dicte un correo o te pida contestar a uno. Si es una respuesta, el "
            "destinatario sale del correo que estabais comentando: no lo inventes. Léele el "
            "texto antes si te lo pide. Si además quiere que salga, después enviar_borrador "
            "con lo que te devuelva esta."
        ),
        chat=(
            "Deja un borrador de correo en Gmail; no lo envía. El destinatario de una "
            "respuesta sale del correo triado: no lo inventes. Para que salga, después "
            "enviar_borrador con el id que devuelve."
        ),
        parametros=(
            Parametro(
                nombre="para",
                tipo="string",
                voz="La dirección de correo del destinatario, completa.",
                chat="La dirección del destinatario.",
                obligatorio=True,
            ),
            Parametro(
                nombre="asunto",
                tipo="string",
                voz="El asunto. En una respuesta, «Re: » y el asunto original.",
                chat="El asunto.",
            ),
            Parametro(
                nombre="texto",
                tipo="string",
                voz="El cuerpo del correo, ya redactado, con saludo y despedida.",
                chat="El cuerpo del correo, ya redactado.",
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

#: El catálogo entero. Los recados y las vigilancias viven en
#: `catalogo_recados.py` por el techo de líneas, no por ser de otra familia: se
#: declaran una vez igual, y aquí se juntan en el orden en que las ve el modelo.
CATALOGO: tuple[Herramienta, ...] = _BASE[:-1] + RECADOS + _BASE[-1:]


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
