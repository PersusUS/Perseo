"""El servidor MCP de correo y la lectura compartida del núcleo.

Lo que se prueba aquí son las decisiones —deduplicar lotes, ordenar como lee
una persona, decir vacío sin rodeos— contra una base temporal. La lógica vive
en `perseo_core.correo_lectura` (el chat la usa por su lado y el servidor MCP
es una cáscara); el proceso entero lo comprueba
`verificadores/verificar_correo_mcp.py`.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "commands"))

import correo_mcp  # noqa: E402
from perseo_core import correo_lectura  # noqa: E402


@pytest.fixture()
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Una base del núcleo con dos lotes de correo ya hechos."""
    ruta = tmp_path / "estado.sqlite3"
    conexion = sqlite3.connect(ruta)
    try:
        conexion.executescript(
            """
            CREATE TABLE trabajos (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                estado         TEXT NOT NULL,
                agente         TEXT NOT NULL,
                origen         TEXT NOT NULL,
                peticion       TEXT NOT NULL,
                resultado      TEXT,
                error          TEXT,
                confirmacion   TEXT,
                intentos       INTEGER NOT NULL DEFAULT 0,
                creado_en      TEXT NOT NULL,
                actualizado_en TEXT NOT NULL,
                reclamado_en   TEXT
            );
            CREATE TABLE correos (
                id_mensaje     TEXT PRIMARY KEY,
                estado         TEXT NOT NULL,
                actualizado_en TEXT NOT NULL
            );
            """
        )
        viejo = {
            "mensajes": [
                {"id": "a-1", "remitente": "x@ejemplo.es", "asunto": "Viejo", "extracto": "cuerpo viejo"},
            ]
        }
        nuevo = {
            "mensajes": [
                {"id": "a-1", "remitente": "x@ejemplo.es", "asunto": "Viejo", "extracto": "cuerpo nuevo"},
                {"id": "b-2", "remitente": "y@ejemplo.es", "asunto": "Oferta", "extracto": "compre ahora"},
            ]
        }
        filas = [
            ("hecho", json.dumps(viejo), json.dumps({"clasificados": [
                {"id": "a-1", "remitente": "x@ejemplo.es", "asunto": "Viejo",
                 "clase": "interesante", "motivo": "viejo motivo"}]})),
            ("hecho", json.dumps(nuevo), json.dumps({"clasificados": [
                {"id": "a-1", "remitente": "x@ejemplo.es", "asunto": "Viejo",
                 "clase": "requiere_accion", "motivo": "nuevo motivo"},
                {"id": "b-2", "remitente": "y@ejemplo.es", "asunto": "Oferta",
                 "clase": "ignorar", "motivo": "publicidad"}]})),
            # Un trabajo fallido no se cuenta: su clasificación no existe.
            ("fallido", json.dumps({"mensajes": []}), None),
        ]
        for estado, peticion, resultado in filas:
            conexion.execute(
                "INSERT INTO trabajos (estado, agente, origen, peticion, resultado, "
                "creado_en, actualizado_en) VALUES (?, 'correo', 'disparador', ?, ?, ?, ?)",
                (estado, peticion, resultado, "2026-08-24", "2026-08-24"),
            )
        conexion.commit()
    finally:
        conexion.close()
    monkeypatch.setattr(correo_mcp, "RUTA_DB", ruta)
    return ruta


# --------------------------------------------------------------------------- #
# Lo que se lee
# --------------------------------------------------------------------------- #


def test_gana_la_clasificacion_mas_reciente(db: Path) -> None:
    correos = {c["id"]: c for c in correo_lectura.cargar_correos(db)}
    assert correos["a-1"]["clase"] == "requiere_accion"
    assert correos["a-1"]["motivo"] == "nuevo motivo"
    assert correos["a-1"]["extracto"] == "cuerpo nuevo"


def test_lo_que_pide_accion_va_primero(db: Path) -> None:
    orden = [c["clase"] for c in correo_lectura.ordenar(correo_lectura.cargar_correos(db))]
    assert orden.index("requiere_accion") < orden.index("ignorar")


def test_la_lista_dice_asuntos_reales(db: Path) -> None:
    texto = correo_mcp.correos_triados()
    assert "Viejo" in texto and "Oferta" in texto
    assert texto.count("Viejo") >= 1


def test_el_filtro_por_clase(db: Path) -> None:
    texto = correo_mcp.correos_triados(clase="requiere_accion")
    assert "requiere acción" in texto and "Oferta" not in texto


def test_una_clase_vacia_se_dice_sin_rodeos(db: Path) -> None:
    texto = correo_mcp.correos_triados(clase="no_seguro")
    assert "No hay ningún correo triado como" in texto


def test_detalle_trae_extracto(db: Path) -> None:
    detalle = correo_mcp.detalle_correo("a-1")
    assert "cuerpo nuevo" in detalle
    assert "requiere acción" in detalle


def test_id_desconocido_contesta_con_gracia(db: Path) -> None:
    assert "No encuentro" in correo_mcp.detalle_correo("fantasma")


def test_sin_base_de_datos_error_claro(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(correo_mcp, "RUTA_DB", tmp_path / "no_esta.sqlite3")
    with pytest.raises(FileNotFoundError) as fallo:
        correo_mcp.correos_triados()
    assert "No hay base de datos" in str(fallo.value)


def test_lotes_rotos_no_tumban_la_lectura(db: Path) -> None:
    """Un resultado que no es JSON se salta; lo demás sigue ahí."""
    conexion = sqlite3.connect(db)
    try:
        conexion.execute(
            "INSERT INTO trabajos (estado, agente, origen, peticion, resultado, "
            "creado_en, actualizado_en) VALUES ('hecho', 'correo', 'disparador', '{}', '{roto', ?, ?)",
            ("2026-08-24", "2026-08-24"),
        )
        conexion.commit()
    finally:
        conexion.close()
    assert "Viejo" in correo_mcp.correos_triados()
