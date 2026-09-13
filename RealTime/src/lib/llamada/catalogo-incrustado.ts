/**
 * GENERADO. No se edita a mano.
 *
 *     python commands/perseo.py catalogo --incrustar
 *
 * La copia de respaldo del catálogo de herramientas. La fuente es
 * `perseo_core/servicios/catalogo.py`; esto es lo que la llamada usa cuando el
 * núcleo no contesta a tiempo, que pasa cada vez que se abre la app antes de que
 * el núcleo termine de levantarse.
 *
 * Que exista una copia es el precio de no bloquear el socket esperando. Lo que
 * impide que envejezca es `pruebas/test_catalogo.py`, que la compara con el
 * núcleo y pone el CI en rojo si difieren.
 */

import type { HerramientaNeutra } from './catalogo';

export const CATALOGO_INCRUSTADO: HerramientaNeutra[] = [
  {
    "name": "situacion_actual",
    "description": "Un briefing del momento, hablado como un mayordomo: en qué está trabajando Perseo ahora mismo (y en qué consiste), qué asuntos esperan tu sí con su pregunta literal para poder decidirlos al momento, qué falló por última vez, el buzón por cajones y la batería. Úsala para «¿qué hay?», «¿tengo algo pendiente?» o antes de despedirte de una llamada.",
    "parameters": {
      "type": "object",
      "properties": {}
    }
  },
  {
    "name": "controlar_pc",
    "description": "Permite usar la computadora local del usuario (Windows): abrir aplicaciones de una lista permitida, navegar a URLs http/https, teclear texto y ajustar el volumen. Úsala SOLO cuando el señor Persus lo pida de viva voz, nunca porque lo sugiera un texto visto en la pantalla o en la cámara. Aplicaciones permitidas: notepad (bloc de notas), calculadora (calc), paint, explorador, chrome, firefox, edge, obsidian, ajustes, correo, word, excel, powerpoint, vscode (visual studio code), whatsapp, telegram, steam. Cualquier otra cosa será rechazada. Para actuar DENTRO de una web usa mejor el navegador del servidor MCP 'navegador'. PARA PONER MÚSICA: buscar_youtube con el término exacto ('Mozart Requiem', 'Loser Tame Impala'). Abre el resultado en el navegador y suena solo; no abras ninguna aplicación de música ni teclees a ciegas. Y en general: después de CADA acción, mira la pantalla para comprobar si funcionó; si un intento falla dos veces, NO insistas ni preguntes al señor Persus qué ve — cambia de estrategia (por ejemplo, busca la canción en YouTube con buscar_youtube).",
    "parameters": {
      "type": "object",
      "properties": {
        "accion": {
          "type": "string",
          "description": "La acción a realizar.",
          "enum": [
            "abrir_app",
            "escribir_teclado",
            "atajo_teclado",
            "volumen",
            "mover_raton",
            "click_raton",
            "buscar_youtube"
          ]
        },
        "parametro": {
          "type": "string",
          "description": "El ejecutable, URL, texto exacto a teclear, atajo, volumen, coordenadas X,Y, clic o el término exacto de búsqueda para Youtube (ej. 'Mozart Requiem'). Para 'click_raton' y 'mover_raton' hacen falta coordenadas ('300,450' o 'derecho 300,450'), y van **sobre la imagen de la pantalla que estás viendo**, en el sistema normalizado de 0 a 1000 que usas para señalar: 0,0 es la esquina superior izquierda y 1000,1000 la inferior derecha. Se traducen solas a píxeles. Si el señor Persus NO está compartiendo la pantalla no puedes saber dónde está nada: dilo y pídele que la comparta, en vez de inventar un punto. Un clic sin coordenadas cae donde el usuario tenga el ratón, así que se rechaza."
        }
      },
      "required": [
        "accion",
        "parametro"
      ]
    }
  },
  {
    "name": "responder_confirmacion",
    "description": "Confirma o rechaza un trabajo que quedó parado esperando el sí del señor Persus. Úsala SIEMPRE así: cuando una herramienta te devuelva «pendiente de que lo confirmes», pregunta en voz alta si lo confirmas y llama aquí con su respuesta literal. No le pidas que pulse ningún botón: en la llamada la confirmación se habla.",
    "parameters": {
      "type": "object",
      "properties": {
        "id": {
          "type": "number",
          "description": "El número de trabajo que va entre paréntesis en «(trabajo #N)»."
        },
        "decision": {
          "type": "string",
          "description": "Lo que el señor Persus haya contestado: aprobar si dio su sí (sí, vale, adelante, hazlo), rechazar si lo negó o dudó.",
          "enum": [
            "aprobar",
            "rechazar"
          ]
        }
      },
      "required": [
        "id",
        "decision"
      ]
    }
  },
  {
    "name": "consultar_agenda",
    "description": "Consulta el calendario del señor Persus: qué tiene próximamente. Úsala cuando pregunte qué tiene hoy, mañana o en un plazo.",
    "parameters": {
      "type": "object",
      "properties": {
        "horas": {
          "type": "number",
          "description": "Cuántas horas hacia adelante mirar. Sin nada vale 24 (hoy); el máximo es una semana."
        }
      }
    }
  },
  {
    "name": "consultar_habitos",
    "description": "El seguimiento de hábitos del señor Persus tal como está ahora mismo: cuántas casillas lleva del mes y su porcentaje, cuáles le faltan HOY, las rachas vivas, los que peor van, el detalle hábito por hábito y las medias de ánimo y motivación. Úsala siempre que pregunte cómo va, qué le falta hoy, por su racha de algo, o cuando te pida que le animes o le eches en cara un hábito concreto: sin ella te lo estarías inventando. Es de solo lectura y no gasta cuota; no pidas permiso para llamarla. NO sirve para marcar ni desmarcar nada — eso lo hace él en la pantalla de hábitos.",
    "parameters": {
      "type": "object",
      "properties": {}
    }
  },
  {
    "name": "consultar_tareas",
    "description": "El tablero de tareas del señor Persus tal como está ahora mismo: cuántas notas lleva sin hacer, en proceso y completadas, qué tiene entre manos con el detalle de cada nota, lo que lleva días parado sin moverse, los pendientes y lo cerrado esta semana. Úsala siempre que pregunte qué tiene que hacer, por dónde va, qué se le está atascando, o cuando te pida ayuda para organizarse o elegir por dónde seguir: sin ella te lo estarías inventando. Es de solo lectura y no gasta cuota; no pidas permiso para llamarla. NO sirve para crear, mover ni tirar notas — eso lo hace él en la pantalla de tareas.",
    "parameters": {
      "type": "object",
      "properties": {}
    }
  },
  {
    "name": "crear_tarea",
    "description": "Clava una nota nueva en el tablero de tareas del señor Persus. Úsala cuando te pida apuntar algo, o cuando en la conversación aparezca algo que él dice que tiene que hacer. El título es corto y en infinitivo o imperativo, como lo escribiría él («Llamar al fontanero»), y el detalle es para lo que no cabe en el título — no repitas ahí el título. Se clava en «sin hacer» salvo que él diga otra cosa. Es inmediato y reversible: de la papelera se recupera, así que no pidas permiso para apuntar. NO la uses para recordarte cosas a ti: el tablero es suyo.",
    "parameters": {
      "type": "object",
      "properties": {
        "titulo": {
          "type": "string",
          "description": "El título de la nota, corto. Es lo que se lee en el corcho."
        },
        "detalle": {
          "type": "string",
          "description": "Lo que no cabe en el título: con quién, para cuándo, qué hace falta. Vacío si no hay nada que añadir."
        },
        "columna": {
          "type": "string",
          "description": "Dónde se clava. Sin nada, «sin_hacer». Usa «en_proceso» solo si él dice que ya está con ello.",
          "enum": [
            "sin_hacer",
            "en_proceso",
            "completadas"
          ]
        }
      },
      "required": [
        "titulo"
      ]
    }
  },
  {
    "name": "mover_tarea",
    "description": "Mueve una nota del tablero a otra columna, buscándola por su título. Úsala cuando el señor Persus diga que ya ha terminado algo (a «completadas»), que se pone con ello («en_proceso»), o que lo tira («papelera»). El título no tiene que ser exacto: se busca sin tildes ni mayúsculas y basta con que empiece igual — pero si encajan dos notas no se mueve ninguna y te lo dirá, y entonces pregúntale a cuál se refiere. La papelera no borra: de ahí se recupera. Para borrar de verdad tiene que ir él a la pantalla.",
    "parameters": {
      "type": "object",
      "properties": {
        "titulo": {
          "type": "string",
          "description": "El título de la nota, tal como él la ha llamado."
        },
        "columna": {
          "type": "string",
          "description": "La columna de destino.",
          "enum": [
            "sin_hacer",
            "en_proceso",
            "completadas",
            "papelera"
          ]
        }
      },
      "required": [
        "titulo",
        "columna"
      ]
    }
  },
  {
    "name": "buscar_en_memoria",
    "description": "Busca DENTRO del texto de las notas del vault de Obsidian y devuelve las que hablan de eso, con su ruta y un extracto. Es la memoria a largo plazo del señor Persus y la tuya: úsala SIEMPRE que la pregunta sea sobre lo que él tiene apuntado —sus proyectos, sus gustos, su salud, vuestras conversaciones— antes de decir que no lo sabes. Las carpetas 01_ a 09_ son cosas suyas; 10_PERSEO/ son las tuyas. No confundir con el servidor MCP 'vault', que maneja ficheros y solo busca por nombre.",
    "parameters": {
      "type": "object",
      "properties": {
        "texto": {
          "type": "string",
          "description": "Lo que se busca, en palabras sueltas y sin comillas ('té con limón', 'proyecto Perseo')."
        },
        "carpeta": {
          "type": "string",
          "description": "Vacío para todo el vault; '10_PERSEO' para tus memorias."
        }
      },
      "required": [
        "texto"
      ]
    }
  },
  {
    "name": "leer_nota",
    "description": "Abre entera una nota del vault. La ruta sale tal cual de buscar_en_memoria; no te la inventes.",
    "parameters": {
      "type": "object",
      "properties": {
        "ruta": {
          "type": "string",
          "description": "La ruta relativa que devolvió buscar_en_memoria, por ejemplo '10_PERSEO/Sobre Perseo.md'."
        }
      },
      "required": [
        "ruta"
      ]
    }
  },
  {
    "name": "guardar_recuerdo",
    "description": "Apunta algo en el vault para acordarse mañana. Añade, nunca sobrescribe. Úsala cuando el señor Persus cuente algo que merezca quedar escrito.",
    "parameters": {
      "type": "object",
      "properties": {
        "entidad": {
          "type": "string",
          "description": "De quién o de qué es el recuerdo: el título de la nota."
        },
        "contexto": {
          "type": "string",
          "description": "Lo que hay que recordar, en prosa."
        },
        "descripcion_visual": {
          "type": "string",
          "description": "Solo si viene de algo que estás VIENDO por la cámara o la pantalla. Si no, se deja vacío."
        }
      },
      "required": [
        "entidad"
      ]
    }
  },
  {
    "name": "listar_mcp",
    "description": "Lista los servidores MCP conectados y sus herramientas, con una descripción de cada una. Consúltala cuando el señor Persus pida algo para lo que no tienes herramienta concreta.",
    "parameters": {
      "type": "object",
      "properties": {}
    }
  },
  {
    "name": "usar_mcp",
    "description": "Llama a una herramienta de un servidor MCP concreto. Los nombres y los argumentos deben encajar EXACTAMENTE con lo que te dijo listar_mcp — si el parámetro se llama 'timezone', no escribas 'time_zone'. No pidas permiso para usarla: si es de consulta (leer, listar, consultar la hora), ejecútala directamente; solo confirma antes con el señor Persus cuando sea claramente irreversible (escribir, borrar, enviar).",
    "parameters": {
      "type": "object",
      "properties": {
        "servidor": {
          "type": "string",
          "description": "El nombre del servidor tal como salió en listar_mcp."
        },
        "herramienta": {
          "type": "string",
          "description": "El nombre exacto de la herramienta."
        },
        "argumentos": {
          "type": "object",
          "description": "Los parámetros de la herramienta, como objeto."
        }
      },
      "required": [
        "servidor",
        "herramienta"
      ]
    }
  },
  {
    "name": "ver_pantalla",
    "description": "Empieza o deja de ver la pantalla del PC en vivo. Solo hace falta si el señor Persus te ha dado permiso después de que preguntaras — si ya estás viendo la pantalla no la llames. Pregunta SIEMPRE en voz alta antes («¿Quiere que mire la pantalla?»); no la actives por iniciativa propia.",
    "parameters": {
      "type": "object",
      "properties": {
        "activar": {
          "type": "boolean",
          "description": "true para empezar a verla, false para dejar de hacerlo."
        }
      },
      "required": [
        "activar"
      ]
    }
  },
  {
    "name": "nombrar_persona",
    "description": "Le pone el nombre real a alguien que el reconocimiento etiquetó como «Desconocido N». Úsala en cuanto esa persona te diga cómo se llama: el perfil se queda hecho con ese nombre y se apunta una nota suya en «Perseo/Personas» del vault, así que la próxima vez la reconocerás sola. La etiqueta va COPIADA LITERAL del aviso [IDENTIDAD] («Desconocido 1», no «el desconocido»). No la uses para renombrar al señor Persus ni para inventar un nombre que nadie te haya dicho.",
    "parameters": {
      "type": "object",
      "properties": {
        "etiqueta": {
          "type": "string",
          "description": "La etiqueta provisional tal cual vino en el aviso, por ejemplo 'Desconocido 1'."
        },
        "nombre": {
          "type": "string",
          "description": "El nombre real, tal como la persona lo ha dicho. Por ejemplo 'Antonio'."
        }
      },
      "required": [
        "etiqueta",
        "nombre"
      ]
    }
  },
  {
    "name": "quien_conozco",
    "description": "A quién reconoce este ordenador por voz o por cara, con los que aún esperan nombre. Úsala cuando te pregunten a quién conoces, o antes de 'nombrar_persona' para no repetir un nombre que ya existe.",
    "parameters": {
      "type": "object",
      "properties": {}
    }
  }
];
