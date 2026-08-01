"""Rutas canónicas del proyecto Perseo.

Único sitio donde se resuelven las rutas del vault y de la base vectorial.
Antes vivían duplicadas en `automator.py`, `memory_tool.py` y `rag_tool.py`, y
no coincidían: unas se resolvían desde `__file__` y otras desde el directorio
de trabajo, así que arrancar el demonio desde otra carpeta hacía que vigilase
un sitio distinto del que escribían las herramientas. Ver H-22.

Todas las rutas se devuelven absolutas y normalizadas, porque ChromaDB filtra
los metadatos por igualdad exacta de cadena: una ruta relativa y una absoluta
apuntando al mismo archivo son dos claves distintas. Ver H-20.
"""

import os

# RAG/paths.py -> RAG/ -> raíz del proyecto
RAIZ_PROYECTO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

VAULT_PATH = os.path.abspath(
    os.getenv("OBSIDIAN_VAULT_PATH", os.path.join(RAIZ_PROYECTO, "obsidian_vault"))
)

DB_PATH = os.path.abspath(os.path.join(RAIZ_PROYECTO, "RAG", "chroma_db"))

COLLECTION_NAME = "obsidian_vault"


def normalizar(ruta: str) -> str:
    """Forma canónica de una ruta, para comparar metadatos con fiabilidad."""
    return os.path.normcase(os.path.abspath(ruta))
