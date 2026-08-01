"""Consulta de la memoria a largo plazo de Perseo (base vectorial).

El índice se carga de forma diferida: importar ChromaDB y validar el modelo de
embeddings cuesta varios segundos, y no tiene sentido pagarlo hasta que el
modelo pida de verdad un recuerdo.
"""

import logging
import os
import sys
from typing import Optional

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

# El paquete RAG no está instalado, se importa por ruta.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "RAG"))

from paths import COLLECTION_NAME, DB_PATH  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_index: Optional[VectorStoreIndex] = None


def _get_lazy_index() -> VectorStoreIndex:
    """Carga diferida del índice vectorial. Solo la primera consulta la paga."""
    global _index
    if _index is None:
        from embedding_manager import configurar_embeddings

        configurar_embeddings()

        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(
                f"No hay base vectorial en {DB_PATH}. Ejecute primero "
                "'python RAG/automator.py --solo-barrido' para indexar el vault."
            )

        db = chromadb.PersistentClient(path=DB_PATH)
        coleccion = db.get_collection(COLLECTION_NAME)
        _index = VectorStoreIndex.from_vector_store(
            ChromaVectorStore(chroma_collection=coleccion)
        )
    return _index


def consultar_base_vectorial(query: str, top_k: int = 5) -> str:
    """Busca por similitud semántica y devuelve los fragmentos encontrados.

    Args:
        query: la consulta en lenguaje natural.
        top_k: número máximo de fragmentos a recuperar.

    Returns:
        Los documentos relevantes formateados, o un mensaje explicativo.
    """
    try:
        nodos = _get_lazy_index().as_retriever(similarity_top_k=top_k).retrieve(query)

        if not nodos:
            return "No se encontraron documentos relevantes en la base de conocimiento."

        bloques = []
        for nodo in nodos:
            ruta = nodo.metadata.get("file_path", "ruta_desconocida")
            nombre = os.path.basename(ruta)
            score = nodo.score if nodo.score is not None else 0.0
            bloques.append(
                f"--- Documento: {nombre} (similitud: {score:.4f}) ---\n"
                f"{nodo.get_content().strip()}"
            )
        return "\n\n".join(bloques)

    except Exception as e:
        logger.error("Error consultando la base vectorial: %s", e, exc_info=True)
        return f"Error interno en la consulta RAG: {e}"


if __name__ == "__main__":
    consulta = sys.argv[1] if len(sys.argv) > 1 else "¿quién es Javi?"
    print(f"Consulta: {consulta}\n")
    print(consultar_base_vectorial(consulta, top_k=2))
