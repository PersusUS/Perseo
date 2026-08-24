"""Lo que el triaje dejó escrito, leído de una sola manera.

El 2026-08-24 Perseo se inventó dos asuntos de correo que no existían porque
ninguna cara tenía a mano los clasificados reales. La lectura se escribió
entonces dentro del servidor MCP de correo (`commands/correo_mcp.py`) y vivía
allí solo. Desde que el chat también necesita mirar el buzón triado, la lógica
vive **aquí** — en el núcleo, donde está la regla de que las caras no piensan —
y el servidor MCP es una cáscara fina sobre estas funciones.

Solo stdlib y SOLO LECTURA: las dos funciones reciben la ruta de la base y no
saben escribir ni tocar Gmail. Marcar un correo como atendido es cosa del panel,
por su propia ruta.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

#: Cuántos trabajos de correo hechos se miran hacia atrás como mucho. Un lote
#: son 20 mensajes (TOPE_LOTE en perseo_core/correo.py); con 40 trabajos hay
#: correo de sobra y la consulta sigue siendo instantánea.
TOPE_TRABAJOS = 40

#: Orden en que se presentan las clases: primero lo que pide una acción, último
#: lo ignorable. Es el orden en que un mayordomo lee una bandeja.
ORDEN_CLASES = ("requiere_accion", "no_seguro", "interesante", "ignorar")

_ETIQUETAS = {
    "requiere_accion": "requiere acción",
    "interesante": "interesante",
    "ignorar": "ignorable",
    "no_seguro": "sin decidir",
}

_ESTADOS_LEGIBLES = {
    "atendido": "ya atendido",
    "descartado": "descartado",
}


def abrir_db(ruta_db: Path) -> sqlite3.Connection:
    """La base del núcleo en modo lectura. Si no está, error claro."""
    if not Path(ruta_db).is_file():
        raise FileNotFoundError(
            f"No hay base de datos del núcleo en {ruta_db}. Arranca `python -m perseo_core` "
            "y espera a que el disparador de correo trié algo."
        )
    # mode=ro: aquí se mira, jamás se escribe. Con la base viva en WAL el lector
    # externo ve lo último confirmado sin estorbar al núcleo.
    return sqlite3.connect(f"file:{Path(ruta_db)}?mode=ro", uri=True)


def cargar_correos(ruta_db: Path, tope_trabajos: int = TOPE_TRABAJOS) -> list[dict[str, Any]]:
    """Los correos triados, deduplicados, el más reciente primero.

    Un mismo mensaje puede aparecer en varios lotes si el buzón lo repitió;
    gana la clasificación más reciente. Junto a cada uno, qué se hizo con él
    según la tabla `correos` — lo que no está ahí está pendiente.
    """
    conexion = abrir_db(ruta_db)
    try:
        filas = conexion.execute(
            "SELECT peticion, resultado FROM trabajos "
            "WHERE agente = 'correo' AND estado = 'hecho' "
            "ORDER BY id DESC LIMIT ?",
            (tope_trabajos,),
        ).fetchall()
        marcas = dict(conexion.execute("SELECT id_mensaje, estado FROM correos").fetchall())
    finally:
        conexion.close()

    vistos: dict[str, dict[str, Any]] = {}
    for peticion_cruda, resultado_crudo in filas:
        try:
            resultado = json.loads(resultado_crudo or "{}")
            peticion = json.loads(peticion_cruda or "{}")
        except json.JSONDecodeError:
            continue
        mensajes = {
            str(m.get("id", "")): m for m in (peticion.get("mensajes") or []) if isinstance(m, dict)
        }
        for clasificado in resultado.get("clasificados") or []:
            if not isinstance(clasificado, dict):
                continue
            id_mensaje = str(clasificado.get("id", ""))
            if not id_mensaje or id_mensaje in vistos:
                continue
            original = mensajes.get(id_mensaje) or {}
            vistos[id_mensaje] = {
                "id": id_mensaje,
                "remitente": str(clasificado.get("remitente") or original.get("remitente") or ""),
                "asunto": str(clasificado.get("asunto") or original.get("asunto") or ""),
                "extracto": str(original.get("extracto") or ""),
                "fecha": str(original.get("fecha") or ""),
                "clase": str(clasificado.get("clase") or ""),
                "motivo": str(clasificado.get("motivo") or ""),
                "hecho": marcas.get(id_mensaje, ""),
            }
    return list(vistos.values())


def ordenar(correos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Primero lo urgente; dentro de cada clase, el más reciente primero."""
    puesto = {nombre: i for i, nombre in enumerate(ORDEN_CLASES)}
    return sorted(correos, key=lambda c: puesto.get(c["clase"], len(ORDEN_CLASES)))


def linea(c: dict[str, Any]) -> str:
    etiqueta = _ETIQUETAS.get(c["clase"], c["clase"] or "sin clase")
    marca = _ESTADOS_LEGIBLES.get(c["hecho"], "")
    trozos = [f"[{etiqueta}] {c['remitente'] or '(sin remitente)'} — “{c['asunto'] or '(sin asunto)'}”"]
    if c["motivo"]:
        trozos.append(f"({c['motivo']})")
    if marca:
        trozos.append(f"— {marca}")
    trozos.append(f"[id: {c['id']}]")
    return " ".join(trozos)


def correos_triados(ruta_db: Path, limite: int = 15, clase: str = "") -> str:
    """La lista real de lo triado. Sin datos no hay poesía: se dice vacío."""
    limite = max(1, min(int(limite or 15), 50))
    clase_limpia = (clase or "").strip().lower()

    todos = ordenar(cargar_correos(ruta_db))
    seleccion = [c for c in todos if not clase_limpia or c["clase"] == clase_limpia]
    if not seleccion:
        if clase_limpia:
            return (
                f"No hay ningún correo triado como '{_ETIQUETAS.get(clase_limpia, clase_limpia)}'. "
                "Con correos_triados sin filtro se ve todo lo hay."
            )
        return (
            "El buzón está al día: no hay ningún correo triado todavía. "
            "El disparador del núcleo mira el buzón cada pocos minutos y lo "
            "clasifica solo."
        )

    visibles = seleccion[:limite]
    lineas = [linea(c) for c in visibles]
    cabecera = f"{len(seleccion)} correo(s) triado(s)"
    if len(visibles) < len(seleccion):
        cabecera += f" (los {len(visibles)} más relevantes; sube 'limite' para ver más)"
    if clase_limpia:
        cabecera += f" — clase '{_ETIQUETAS.get(clase_limpia, clase_limpia)}'"
    return cabecera + ":\n" + "\n".join(lineas)


def detalle_correo(ruta_db: Path, id_mensaje: str) -> str:
    """El extracto de un correo concreto, por el id que dio correos_triados."""
    id_buscado = str(id_mensaje or "").strip().strip("\"'[]")
    if not id_buscado:
        raise ValueError("Falta el identificador del correo (sale en correos_triados).")

    for c in cargar_correos(ruta_db):
        if c["id"] != id_buscado:
            continue
        etiqueta = _ETIQUETAS.get(c["clase"], c["clase"] or "sin clase")
        marca = _ESTADOS_LEGIBLES.get(c["hecho"], "pendiente de que lo mires")
        partes = [
            f"Asunto: {c['asunto'] or '(sin asunto)'}",
            f"De: {c['remitente'] or '(sin remitente)'}",
            f"Clase: {etiqueta} ({marca})",
        ]
        if c["fecha"]:
            partes.append(f"Fecha: {c['fecha']}")
        if c["motivo"]:
            partes.append(f"Motivo del triaje: {c['motivo']}")
        extracto = (c["extracto"] or "").strip()
        partes.append(
            "Extracto: " + (extracto[:2000] if extracto else "(el núcleo no bajó el cuerpo: "
            "del correo solo se tría lo que dicen las cabeceras)")
        )
        return "\n".join(partes)

    return (
        f"No encuentro ningún correo con el identificador '{id_buscado}' en lo "
        "triado hasta ahora. El identificador literal sale entre corchetes en "
        "correos_triados; ojo con no confundirlo con un número de trabajo de la cola."
    )
