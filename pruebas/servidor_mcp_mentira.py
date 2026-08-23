"""Un servidor MCP de mentira, para probar el cliente sin descargar nada.

Habla el protocolo de verdad —JSON-RPC 2.0 sobre stdio, una línea por mensaje—
pero con dos herramientas de juguete: `eco`, que devuelve lo que le llegue, y
`tarda`, que se duerme antes de contestar. Además mete ruido a propósito: una
notificación suelta y una petición que no es para nosotros, porque los
servidores reales hacen cosas así y el cliente no puede quedarse colgado
esperando la respuesta de su propia petición.

    python pruebas/servidor_mcp_mentira.py
"""

from __future__ import annotations

import json
import sys
import time


def responder(identificador: int, resultado) -> None:
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

        # Ruido voluntario: llega sin que nadie lo pida. El cliente debe
        # ignorarlo y seguir esperando SU respuesta.
        if mensaje.get("id") is None and metodo != "notifications/initialized":
            print(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 999,
                        "method": "sampling/createMessage",
                        "params": {},
                    }
                ),
                flush=True,
            )

        if identificador is None:
            continue  # notificación: no se contesta nunca

        if metodo == "initialize":
            responder(
                identificador,
                {"protocolVersion": "2024-11-05", "capabilities": {}, "serverInfo": {"name": "mentira"}},
            )
        elif metodo == "tools/list":
            responder(
                identificador,
                {
                    "tools": [
                        {"name": "eco", "description": "Devuelve lo que le mandes."},
                        {"name": "tarda", "description": "Se duerme antes de contestar."},
                    ]
                },
            )
        elif metodo == "tools/call":
            nombre = (mensaje.get("params") or {}).get("name")
            argumentos = (mensaje.get("params") or {}).get("arguments") or {}
            if nombre == "eco":
                responder(
                    identificador,
                    {"content": [{"type": "text", "text": "eco: " + json.dumps(argumentos)}]},
                )
            elif nombre == "tarda":
                time.sleep(30)
                responder(identificador, {"content": [{"type": "text", "text": "demasiado tarde"}]})
            else:
                responder(
                    identificador,
                    {
                        "content": [{"type": "text", "text": f"no conozco {nombre}"}],
                        "isError": True,
                    },
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
