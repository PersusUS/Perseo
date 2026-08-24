"""Verificación del servidor MCP de correo: la voz lee lo que el triaje escribió.

Este servidor existe por una escena concreta (2026-08-24): el señor Persus
pidió por voz sus correos y Perseo se inventó dos asuntos que no existían,
porque ninguna herramienta de la llamada devolvía correos de verdad. Aquí se
comprueba que el camino nuevo dice SIEMPRE la verdad o calla:

    - sin base de datos, error claro (nada de improvisar);
    - `correos_triados` devuelve los asuntos REALES que dejó el agente `correo`;
    - gana la clasificación más reciente cuando un correo salió en dos lotes;
    - el orden es el del mayordomo: primero lo que pide acción;
    - las marcas del panel (atendido/descartado) se ven;
    - `detalle_correo` trae el extracto completo de uno;
    - un identificador desconocido contesta con gracia, sin error rojo.

Contra el proceso real: se le arranca igual que lo lanza el núcleo desde
`mcp.json` y se le habla con el mismo cliente MCP de `perseo_core`.

    python perseo_core/verificar_correo_mcp.py
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perseo_core import almacen, mcp  # noqa: E402
from perseo_core.arnes_pruebas import comprobar, resumir  # noqa: E402

SERVIDOR = Path(__file__).resolve().parent.parent / "commands" / "correo_mcp.py"

# Dos lotes. El mensaje "factura-10" aparece en los DOS con clasificaciones
# distintas: manda el trabajo más reciente, que es como lo ve una persona.
LOTE_VIEJO = {
    "accion": "triar",
    "mensajes": [
        {
            "id": "factura-10",
            "remitente": "administracion@comunidad.example",
            "asunto": "Derrama ordinaria octubre",
            "extracto": "La junta aprueba una derrama de 120 euros para la fachada.",
            "fecha": "2026-08-20",
        },
        {
            "id": "webinar-7",
            "remitente": "no-reply@eventos.example",
            "asunto": "Webinar de productividad mañana",
            "extracto": "Reserva tu plaza para la sesión gratuita.",
            "fecha": "2026-08-21",
        },
    ],
}
RESULTADO_VIEJO = {
    "recuento": {},
    "clasificados": [
        {
            "id": "factura-10",
            "remitente": "administracion@comunidad.example",
            "asunto": "Derrama ordinaria octubre",
            "clase": "interesante",
            "motivo": "clasificacion vieja",
        },
        {
            "id": "webinar-7",
            "remitente": "no-reply@eventos.example",
            "asunto": "Webinar de productividad mañana",
            "clase": "no_seguro",
            "motivo": "no lo tengo claro",
        },
    ],
    "titular": "2 correos",
}

LOTE_NUEVO = {
    "accion": "triar",
    "mensajes": [
        LOTE_VIEJO["mensajes"][0],
        {
            "id": "newsletter-3",
            "remitente": "hola@tienda.example",
            "asunto": "Rebajas de verano hasta el 70%",
            "extracto": "Ultimas horas para aprovechar los descuentos.",
            "fecha": "2026-08-22",
        },
    ],
}
RESULTADO_NUEVO = {
    "recuento": {},
    "clasificados": [
        {
            "id": "factura-10",
            "remitente": "administracion@comunidad.example",
            "asunto": "Derrama ordinaria octubre",
            "clase": "requiere_accion",
            "motivo": "pide pago con plazo",
        },
        {
            "id": "newsletter-3",
            "remitente": "hola@tienda.example",
            "asunto": "Rebajas de verano hasta el 70%",
            "clase": "ignorar",
            "motivo": "publicidad",
        },
    ],
    "titular": "2 correos",
}


def montar_datos(directorio: Path) -> None:
    """La base del núcleo con dos trabajos de correo hechos, como los deja."""
    conexion = sqlite3.connect(directorio / "estado.sqlite3")
    try:
        conexion.executescript(almacen._ESQUEMA)
        for numero, (peticion, resultado) in enumerate(
            [(LOTE_VIEJO, RESULTADO_VIEJO), (LOTE_NUEVO, RESULTADO_NUEVO)], start=1
        ):
            conexion.execute(
                "INSERT INTO trabajos (estado, agente, origen, peticion, resultado, "
                "intentos, creado_en, actualizado_en) VALUES (?,?,?,?,?,?,?,?)",
                (
                    "hecho",
                    "correo",
                    "disparador",
                    json.dumps(peticion, ensure_ascii=False),
                    json.dumps(resultado, ensure_ascii=False),
                    0,
                    f"2026-08-2{numero}T10:00:00",
                    f"2026-08-2{numero}T10:00:05",
                ),
            )
        # La newsletter ya la resolvió una persona en el panel.
        conexion.execute(
            "INSERT INTO correos (id_mensaje, estado, actualizado_en) VALUES (?,?,?)",
            ("newsletter-3", "descartado", "2026-08-23T09:00:00"),
        )
        conexion.commit()
    finally:
        conexion.close()


async def guion() -> None:
    temporal = tempfile.TemporaryDirectory(prefix="perseo_correo_mcp_")
    directorio = Path(temporal.name)
    montar_datos(directorio)

    mcp.definiciones.clear()
    mcp.definiciones["correo"] = {
        "comando": [sys.executable, "-X", "utf8", str(SERVIDOR)],
        "nivel": "libre",
        "herramientas": [],
        "env": {"PERSEO_DATOS": str(directorio)},
        "tope_segundos": 30.0,
    }
    servidor = mcp.ServidorMcp("correo", mcp.definiciones["correo"])
    await servidor.arrancar()
    try:
        nombres = sorted(h.get("name") for h in servidor.herramientas)
        comprobar("El saludo lista sus dos herramientas", nombres == ["correos_triados", "detalle_correo"], str(nombres))

        lista = await servidor.llamar("correos_triados", {})
        comprobar("El asunto real de la derrama está", "Derrama ordinaria octubre" in lista)
        comprobar("Y también la rebaja real", "Rebajas de verano" in lista)
        comprobar(
            "Lo urgente va antes que lo dudoso",
            lista.index("requiere acción") < lista.index("sin decidir"),
            lista.splitlines()[0],
        )
        comprobar("Gana la clasificación más nueva", "pide pago con plazo" in lista)
        comprobar("La marca del panel se ve", "descartado" in lista)
        comprobar("Cada correo sale una sola vez", lista.count("Derrama ordinaria octubre") == 1)

        solo_urgentes = await servidor.llamar("correos_triados", {"clase": "requiere_accion"})
        comprobar("El filtro por clase deja solo lo urgente", "Rebajas" not in solo_urgentes and "Derrama" in solo_urgentes)

        detalle = await servidor.llamar("detalle_correo", {"id_mensaje": "factura-10"})
        comprobar("El extracto llega entero", "derrama de 120 euros" in detalle)

        fantasma = await servidor.llamar("detalle_correo", {"id_mensaje": "no-existo"})
        # El cliente convierte isError en excepción; llegar aquí con texto
        # educado significa que el servidor NO se cayó ni inventó nada.
        comprobar("Un id desconocido se contesta con gracia", "No encuentro" in fantasma or fantasma == "")

        vacia = await servidor.llamar("correos_triados", {"clase": "interesante"})
        comprobar("Una clase sin correos se dice sin rodeos", "No hay ningún correo triado como" in vacia)
    finally:
        await servidor.detener()
        temporal.cleanup()

    # Sin base de datos: error claro en vez de un buzón imaginario. Es la
    # comprobación que importa después de lo que pasó el 2026-08-24.
    mcp.definiciones["correo"] = {
        **mcp.definiciones["correo"],
        "env": {"PERSEO_DATOS": str(Path(tempfile.gettempdir()) / "perseo_no_existe_nunca")},
    }
    solitario = mcp.ServidorMcp("correo", mcp.definiciones["correo"])
    await solitario.arrancar()
    try:
        try:
            respuesta = await solitario.llamar("correos_triados", {})
            texto = respuesta
        except mcp.ErrorMcp as e:
            texto = str(e)
        comprobar(
            "Sin base de datos se admite el límite, no se inventa",
            "No hay base de datos" in texto,
            texto[:120],
        )
    finally:
        await solitario.detener()


def main() -> None:
    print("--- el servidor MCP de correo, contra su proceso real ---\n")
    asyncio.run(guion())
    print()
    resumir()


if __name__ == "__main__":
    main()
