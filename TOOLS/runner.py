"""Ejecución puntual de una herramienta desde la línea de comandos.

Ya NO es el camino que usa la aplicación: Rust habla con `server.py`, un proceso
persistente, porque arrancar un intérprete por llamada costaba 7,5 s. Este script
se mantiene para pruebas manuales y diagnóstico:

    python TOOLS/runner.py consultar_base_vectorial '{"query": "quién es Javi"}'
"""

import io
import json
import logging
import sys

# UTF-8 explícito: en Windows, Python escribe por defecto en la página de códigos
# del sistema (cp1252), y quien lea esta salida como UTF-8 recibe los acentos
# corruptos. Ver H-13.
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")
if isinstance(sys.stderr, io.TextIOWrapper):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Los registros van a stderr para no contaminar el resultado en stdout.
logging.basicConfig(level=logging.ERROR, stream=sys.stderr)

HERRAMIENTAS = {
    "consultar_base_vectorial": ("rag_tool", "consultar_base_vectorial"),
    "guardar_recuerdo": ("memory_tool", "guardar_recuerdo"),
    "guardar_conversacion": ("memory_tool", "guardar_conversacion"),
    "controlar_pc": ("pc_tool", "controlar_pc"),
}


def _error(mensaje: str) -> None:
    """Informa de un fallo por stderr y termina.

    Antes esto se imprimía en stdout y se salía con código 1, así que Rust
    devolvía `stderr`, que estaba vacío: el modelo recibía literalmente
    "Error en script Python: " sin ninguna causa. Ver H-14.
    """
    print(mensaje, file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if len(sys.argv) < 3:
        _error(f"Uso: python runner.py <herramienta> '<json>'\n"
               f"Herramientas: {', '.join(HERRAMIENTAS)}")

    nombre, args_json = sys.argv[1], sys.argv[2]

    if nombre not in HERRAMIENTAS:
        _error(f"Herramienta desconocida: {nombre}. Opciones: {', '.join(HERRAMIENTAS)}")

    try:
        argumentos = json.loads(args_json)
    except json.JSONDecodeError as e:
        _error(f"Error parseando los argumentos JSON: {e}")

    modulo, funcion = HERRAMIENTAS[nombre]
    try:
        importado = __import__(modulo, fromlist=[funcion])
        print(getattr(importado, funcion)(**argumentos))
    except TypeError as e:
        _error(f"Argumentos inválidos para {nombre}: {e}")
    except Exception as e:
        _error(f"Error ejecutando {nombre}: {e}")


if __name__ == "__main__":
    main()
