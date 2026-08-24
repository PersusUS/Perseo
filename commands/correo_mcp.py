"""Servidor MCP de correo: lo que el triaje ya clasificó, listo para la voz.

El hueco que cierra este servidor tiene nombre y fecha: el 2026-08-24 el señor
Persus pidió por voz «listarme los correos» y Perseo se los INVENTÓ — dos
asuntos que no existían — porque ninguna herramienta de la llamada devolvía
correos de verdad. `situacion_actual` solo dice recuentos («2 requieren
acción»), y los clasificados del agente `correo` vivían solo en la cola, que la
voz no lee. Ante un hueco así, el modelo rellena con ficción: no es maldad, es
lo que hace un modelo al que se le pregunta algo para lo que no tiene manos.

La regla de casa para esto ya estaba escrita (bitacora, N-3): lo que falta no
se parchea con promesas en el prompt, se cierra con una herramienta de verdad.
Y como Perseo ya habla MCP con medio mundo, el camino corto es este servidor.

**La lógica vive en el núcleo** (`perseo_core/correo_lectura.py`) y no aquí:
el chat escrito necesita leer exactamente lo mismo, y dos copias de una lectura
acaban discrepando. Este fichero es la cáscara que habla el protocolo.

Dos herramientas, las dos SOLO LECTURA:

1. `correos_triados`: los últimos correos triados con su clase, remitente,
   asunto y motivo — lo mínimo para responder «¿hay algo importante?». Opcional
   filtro por clase y tope de lista.
2. `detalle_correo`: el extracto completo de uno, por su identificador.

Por qué nivel `libre` en `mcp.json`: leer es lo más libre que hay según la
política de §7, igual que «leer correo» en la tabla del README. Marcar un
correo como atendido o descartado NO está aquí a propósito: eso es decisión de
persona y vive en el panel.

Seguridad: sin shell, sin escritura, sin red. La base se abre en modo
`read-only` por URI; aunque alguien encadenara una inyección en los argumentos,
lo único alcanzable es SELECT. Los argumentos van siempre ligados, nunca
interpolados en el SQL.

    Perseo Core lo lanza desde <datos>/mcp.json:
    ["python", "-X", "utf8", "...\\commands\\correo_mcp.py"]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

VERSION_PROTOCOLO = "2024-11-05"

RAIZ = Path(__file__).resolve().parent.parent
DATOS = Path(os.environ.get("PERSEO_DATOS", "") or (RAIZ / "perseo_core" / "datos"))
RUTA_DB = DATOS / "estado.sqlite3"

# El servidor es un proceso hijo lanzado con la ruta del script, así que el
# paquete del núcleo no está en su `sys.path`. Se añade aquí y no arriba:
# antes de esta línea no hay nada que lo necesite.
sys.path.insert(0, str(RAIZ))

from perseo_core import correo_lectura  # noqa: E402


def correos_triados(limite: int = 15, clase: str = "") -> str:
    return correo_lectura.correos_triados(RUTA_DB, limite, clase)


def detalle_correo(id_mensaje: str) -> str:
    return correo_lectura.detalle_correo(RUTA_DB, id_mensaje)


# --------------------------------------------------------------------------- #
# Servidor JSON-RPC sobre stdio, una línea por mensaje
# --------------------------------------------------------------------------- #

HERRAMIENTAS = [
    {
        "name": "correos_triados",
        "description": (
            "Lista REAL de los últimos correos que el núcleo trió: remitente, "
            "asunto, clase (requiere acción / interesante / ignorable / sin "
            "decidir), motivo y qué se hizo con cada uno. Úsala SIEMPRE antes de "
            "hablar del buzón: los asuntos que no salgan de aquí no existen. "
            "Para el contenido de uno, detalle_correo con su id."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limite": {
                    "type": "number",
                    "description": "Cuántos correos listar como mucho. Por defecto 15.",
                },
                "clase": {
                    "type": "string",
                    "enum": ["requiere_accion", "interesante", "ignorar", "no_seguro"],
                    "description": "Si quieres solo una clase: 'requiere_accion' para lo importante.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "detalle_correo",
        "description": (
            "El extracto completo de un correo triado, con su clase y motivo. "
            "El identificador es el literal entre corchetes que devolvió "
            "correos_triados."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "id_mensaje": {
                    "type": "string",
                    "description": "El identificador literal del correo (p. ej. 'nuevo-1').",
                }
            },
            "required": ["id_mensaje"],
        },
    },
]

_ACCIONES = {
    "correos_triados": lambda a: correos_triados(
        a.get("limite") if isinstance(a.get("limite"), (int, float)) else 15,
        str(a.get("clase", "") or ""),
    ),
    "detalle_correo": lambda a: detalle_correo(str(a.get("id_mensaje", "") or "")),
}


def _responder(identificador, resultado) -> None:
    print(json.dumps({"jsonrpc": "2.0", "id": identificador, "result": resultado}), flush=True)


def main() -> None:
    for linea in sys.stdin:
        linea = linea.strip()
        if not linea:
            continue
        try:
            mensaje = json.loads(linea)
        except json.JSONDecodeError:
            continue
        metodo = str(mensaje.get("method", ""))
        identificador = mensaje.get("id")
        if identificador is None:
            continue  # notificación: nada que contestar

        if metodo == "initialize":
            _responder(
                identificador,
                {
                    "protocolVersion": VERSION_PROTOCOLO,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "correo", "version": "1.0"},
                },
            )
        elif metodo == "tools/list":
            _responder(identificador, {"tools": HERRAMIENTAS})
        elif metodo == "tools/call":
            parametros = mensaje.get("params") or {}
            nombre = str(parametros.get("name") or "")
            argumentos = parametros.get("arguments") or {}
            accion = _ACCIONES.get(nombre)
            if accion is None:
                _responder(
                    identificador,
                    {
                        "content": [{"type": "text", "text": f"herramienta desconocida: {nombre}"}],
                        "isError": True,
                    },
                )
                continue
            try:
                _responder(identificador, {"content": [{"type": "text", "text": accion(argumentos)}]})
            except Exception as e:  # noqa: BLE001 — el error viaja al modelo, no tumba el servidor
                _responder(
                    identificador,
                    {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True},
                )
        else:
            print(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": identificador,
                        "error": {"code": -32601, "message": f"no sé hacer {metodo}"},
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
