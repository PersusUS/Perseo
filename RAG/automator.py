import os
import time
import logging
from typing import List, Dict, Any
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core import SimpleDirectoryReader

# Importar configuración de embeddings
from embedding_manager import configurar_embeddings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH: str = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME: str = "obsidian_vault"
VAULT_PATH: str = os.getenv("OBSIDIAN_VAULT_PATH", "../obsidian_vault")

class ObsidianEventHandler(FileSystemEventHandler):
    """
    Manejador de eventos del sistema de archivos para el vault de Obsidian.
    Intercepta creaciones y modificaciones de archivos Markdown.
    """
    
    def __init__(self, index: VectorStoreIndex, chroma_collection: Any):
        super().__init__()
        self.index = index
        self.chroma_collection = chroma_collection
        self.node_parser = MarkdownNodeParser()

    def process_file(self, file_path: str) -> None:
        """
        Procesa un archivo Markdown individual: extrae, fragmenta e indexa.
        Elimina vectores previos asociados al archivo para evitar duplicidad.
        """
        if not file_path.endswith(".md"):
            return
            
        logger.info(f"Procesando archivo: {file_path}")
        
        try:
            # 1. Eliminar vectores preexistentes basados en la ruta del documento
            try:
                self.chroma_collection.delete(where={"file_path": file_path})
                logger.debug(f"Vectores previos eliminados para {file_path}")
            except Exception as e:
                logger.debug(f"No se encontraron vectores previos o ocurrió un error al borrar: {e}")

            # 2. Cargar documento
            reader = SimpleDirectoryReader(input_files=[file_path])
            documents: List[Document] = reader.load_data()
            
            if not documents:
                return

            # 3. Fragmentar respetando jerarquía Markdown
            nodes = self.node_parser.get_nodes_from_documents(documents)
            
            # Adicionar metadatos explícitos a nodos para asegurar el filtrado futuro
            for node in nodes:
                if 'file_path' not in node.metadata:
                    node.metadata['file_path'] = file_path

            # 4. Insertar en el índice
            if nodes:
                self.index.insert_nodes(nodes)
                logger.info(f"Archivo indexado exitosamente: {file_path} ({len(nodes)} nodos)")

        except PermissionError:
            logger.error(f"Error de permisos al leer el archivo (posible bloqueo I/O): {file_path}")
        except Exception as e:
            logger.error(f"Error procesando el archivo {file_path}: {e}", exc_info=True)

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            # Ligero delay para asegurar que la escritura inicial haya finalizado
            time.sleep(1)
            self.process_file(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            time.sleep(1)
            self.process_file(event.src_path)

def inicializar_sistema_rag() -> tuple[VectorStoreIndex, Any]:
    """
    Inicializa la base de datos ChromaDB y el índice vectorial de LlamaIndex.
    """
    configurar_embeddings()
    
    os.makedirs(DB_PATH, exist_ok=True)
    db = chromadb.PersistentClient(path=DB_PATH)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Crea el índice (o lo carga si ya existen datos)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
    )
    
    return index, chroma_collection

def iniciar_demonio() -> None:
    """
    Función principal que arranca el demonio de monitorización.
    """
    logger.info("Inicializando sistema RAG y base de datos vectorial...")
    try:
        index, chroma_collection = inicializar_sistema_rag()
    except Exception as e:
        logger.critical(f"Falla al inicializar RAG: {e}")
        return

    event_handler = ObsidianEventHandler(index, chroma_collection)
    observer = Observer()
    
    if not os.path.exists(VAULT_PATH):
        logger.warning(f"La ruta del vault no existe: {VAULT_PATH}. Esperando creación...")
        os.makedirs(VAULT_PATH, exist_ok=True)

    observer.schedule(event_handler, VAULT_PATH, recursive=True)
    observer.start()
    
    logger.info(f"Monitorización iniciada en: {os.path.abspath(VAULT_PATH)}")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Monitorización detenida por el usuario.")
    
    observer.join()

if __name__ == "__main__":
    iniciar_demonio()