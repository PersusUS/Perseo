import os
import sys
import logging
from typing import Optional

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

# Añadir el directorio RAG al PATH temporal para importar módulos si se ejecuta desde TOOLS
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "RAG"))

logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Variables globales para Lazy Loading
_index: Optional[VectorStoreIndex] = None

def _get_lazy_index() -> VectorStoreIndex:
    """
    Carga diferida del índice vectorial LlamaIndex/ChromaDB.
    Se ejecuta únicamente en la primera consulta para no bloquear el hilo principal en el arranque.
    """
    global _index
    if _index is None:
        try:
            from embedding_manager import configurar_embeddings
            configurar_embeddings()
            
            db_path = os.path.join(os.path.dirname(__file__), "..", "RAG", "chroma_db")
            
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"Directorio ChromaDB no encontrado en: {db_path}")

            db = chromadb.PersistentClient(path=db_path)
            chroma_collection = db.get_collection("obsidian_vault")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            
            _index = VectorStoreIndex.from_vector_store(vector_store)
        except Exception as e:
            logger.error(f"Falla al cargar el índice vectorial en diferido: {e}")
            raise e
            
    return _index

def consultar_base_vectorial(query: str, top_k: int = 5) -> str:
    """
    Realiza una inferencia de similitud semántica contra la base vectorial y retorna
    los fragmentos encontrados en un string estandarizado.

    Args:
        query (str): La consulta del usuario.
        top_k (int): Número máximo de fragmentos a recuperar.

    Returns:
        str: Cadena formateada con los documentos y sus respectivos textos.
    """
    try:
        index = _get_lazy_index()
        retriever = index.as_retriever(similarity_top_k=top_k)
        nodos_recuperados = retriever.retrieve(query)
        
        if not nodos_recuperados:
            return "No se encontraron documentos relevantes en la base de conocimiento."

        resultado_formateado = []
        for nodo in nodos_recuperados:
            file_path = nodo.metadata.get("file_path", "Ruta_desconocida")
            score = nodo.score if nodo.score is not None else 0.0
            texto = nodo.get_content().strip()
            
            bloque = f"--- Documento: {file_path} (Similitud: {score:.4f}) ---\n{texto}"
            resultado_formateado.append(bloque)
            
        return "\n\n".join(resultado_formateado)
        
    except Exception as e:
        logger.error(f"Error consultando la base vectorial: {e}", exc_info=True)
        return f"Error interno en la consulta RAG: {str(e)}"