"""Proceso persistente de herramientas de Perseo.

Rust lo arranca una vez, al abrir la aplicación, y le habla por tuberías con
JSON delimitado por líneas. Sustituye al esquema anterior, que lanzaba un
intérprete de Python nuevo en cada llamada.

Por qué importa: importar llama_index y chromadb cuesta ~3,7 s, y validar el
modelo de embeddings añade una petición de red. Medido, cada llamada tardaba
7,5 s contra un timeout de 10 s — el 75 % del presupuesto, siempre, no solo la
primera vez. Con el proceso vivo, ese coste se paga una vez mientras el usuario
todavía está conectando. Ver H-12.

Por qué tuberías y no HTTP: un servidor en localhost que expone `controlar_pc`
sería alcanzable por cualquier proceso de la máquina. Las tuberías son privadas
del proceso padre, así que no abren superficie nueva.

PROTOCOLO
---------
Entrada (una línea por petición):
    {"id": 1, "tool": "consultar_base_vectorial", "args": {"query": "..."}}
Salida (una línea por respuesta):
    {"id": 1, "ok": true,  "result": "..."}
    {"id": 1, "ok": false, "error": "..."}

stdout es EXCLUSIVAMENTE el canal del protocolo. Todo registro va a stderr.
"""

import json
import logging
import sys
import threading
import traceback

# stdout y stderr en UTF-8 explícito. Sin esto, Python en Windows escribe en la
# página de códigos del sistema (cp1252) y Rust lo lee como UTF-8: los acentos
# de las respuestas llegaban corruptos al modelo. Ver H-13.
sys.stdout.reconfigure(encoding="utf-8", newline="\n")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - servidor - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

HERRAMIENTAS = {}


def _cargar_herramientas() -> None:
    """Importa los módulos de herramientas. Barato: nada de RAG todavía."""
    from memory_tool import guardar_recuerdo
    from pc_tool import controlar_pc
    from rag_tool import consultar_base_vectorial

    HERRAMIENTAS.update({
        "consultar_base_vectorial": consultar_base_vectorial,
        "guardar_recuerdo": guardar_recuerdo,
        "controlar_pc": controlar_pc,
    })


def _precalentar() -> None:
    """Carga el índice vectorial en segundo plano, nada más arrancar.

    Es el motivo de existir de este proceso: cuando llegue la primera consulta
    real, el índice ya está en memoria y la respuesta es inmediata.
    """
    try:
        from rag_tool import _get_lazy_index

        _get_lazy_index()
        logger.info("Índice vectorial precargado y listo.")
    except Exception as e:
        # No es fatal: las herramientas que no son de RAG siguen funcionando, y
        # la consulta reintentará la carga por su cuenta.
        logger.warning("No se pudo precargar el índice vectorial: %s", e)


def _atender(peticion: dict) -> dict:
    id_peticion = peticion.get("id")
    nombre = peticion.get("tool")
    argumentos = peticion.get("args") or {}

    funcion = HERRAMIENTAS.get(nombre)
    if funcion is None:
        return {"id": id_peticion, "ok": False, "error": f"Herramienta desconocida: {nombre}"}

    try:
        return {"id": id_peticion, "ok": True, "result": funcion(**argumentos)}
    except TypeError as e:
        return {"id": id_peticion, "ok": False, "error": f"Argumentos inválidos para {nombre}: {e}"}
    except Exception as e:
        logger.error("Error en %s: %s\n%s", nombre, e, traceback.format_exc())
        return {"id": id_peticion, "ok": False, "error": f"Error ejecutando {nombre}: {e}"}


def _responder(respuesta: dict) -> None:
    sys.stdout.write(json.dumps(respuesta, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> None:
    _cargar_herramientas()
    threading.Thread(target=_precalentar, daemon=True).start()

    logger.info("Servidor de herramientas listo (%d disponibles).", len(HERRAMIENTAS))
    _responder({"id": 0, "ok": True, "result": "listo"})

    for linea in sys.stdin:
        linea = linea.strip()
        if not linea:
            continue
        try:
            peticion = json.loads(linea)
        except json.JSONDecodeError as e:
            _responder({"id": None, "ok": False, "error": f"JSON inválido: {e}"})
            continue

        _responder(_atender(peticion))

    logger.info("Entrada cerrada; el servidor termina.")


if __name__ == "__main__":
    main()
