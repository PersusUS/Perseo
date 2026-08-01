"""Indexador del vault de Obsidian para la memoria a largo plazo de Perseo.

Hace dos cosas: un barrido completo al arrancar y una vigilancia continua del
vault para reindexar lo que cambie.

El barrido inicial es lo que faltaba (H-02). La versión anterior solo indexaba
dentro de los eventos de watchdog, así que los archivos que ya existían antes
de arrancar el demonio no se indexaban nunca. Como el demonio tampoco se
lanzaba en ningún sitio, la base vectorial llevaba meses con cero embeddings
mientras el vault acumulaba memorias.
"""

import logging
import os
import threading
import time
from typing import Any, Dict, List

import chromadb
from llama_index.core import Document, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from embedding_manager import configurar_embeddings
from paths import COLLECTION_NAME, DB_PATH, VAULT_PATH, normalizar

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Margen tras el último evento antes de reindexar. Un solo guardado de Obsidian
# dispara varios `on_modified`; sin este margen el archivo se reindexaba una vez
# por evento. Ver H-19.
RETARDO_ANTIRREBOTE_S = 1.5


class Indexador:
    """Encapsula el índice vectorial y las operaciones sobre un archivo."""

    def __init__(self, index: VectorStoreIndex, coleccion: Any):
        self.index = index
        self.coleccion = coleccion
        self.node_parser = MarkdownNodeParser()

    def _mtimes_indexados(self) -> Dict[str, float]:
        """Mapa {ruta normalizada -> mtime} de lo que ya está en la base."""
        try:
            datos = self.coleccion.get(include=["metadatas"])
        except Exception as e:
            logger.warning("No se pudieron leer los metadatos existentes: %s", e)
            return {}

        mapa: Dict[str, float] = {}
        for metadatos in datos.get("metadatas") or []:
            if not metadatos:
                continue
            ruta, mtime = metadatos.get("file_path"), metadatos.get("mtime")
            if ruta and mtime is not None:
                mapa[ruta] = max(mapa.get(ruta, 0.0), float(mtime))
        return mapa

    def eliminar(self, ruta: str) -> None:
        """Borra los vectores asociados a un archivo.

        La ruta se normaliza antes de filtrar: ChromaDB compara los metadatos
        por igualdad exacta de cadena, así que una ruta relativa y una absoluta
        del mismo archivo serían dos claves distintas y los vectores viejos se
        acumularían en cada reindexado. Ver H-20.
        """
        try:
            self.coleccion.delete(where={"file_path": normalizar(ruta)})
        except Exception as e:
            logger.debug("Sin vectores previos para %s (%s)", ruta, e)

    def indexar_archivo(self, ruta: str) -> bool:
        """Reindexa un archivo Markdown. Devuelve True si se indexó algo."""
        if not ruta.lower().endswith(".md") or not os.path.isfile(ruta):
            return False

        ruta_norm = normalizar(ruta)
        try:
            self.eliminar(ruta_norm)

            documentos: List[Document] = SimpleDirectoryReader(input_files=[ruta]).load_data()
            if not documentos:
                return False

            nodos = self.node_parser.get_nodes_from_documents(documentos)
            if not nodos:
                return False

            mtime = os.path.getmtime(ruta)
            for nodo in nodos:
                nodo.metadata["file_path"] = ruta_norm
                nodo.metadata["mtime"] = mtime
                # Estos metadatos son para el filtrado, no para el significado:
                # si entran en el texto que se vectoriza, ensucian la búsqueda.
                nodo.excluded_embed_metadata_keys = ["file_path", "mtime"]
                nodo.excluded_llm_metadata_keys = ["mtime"]

            self.index.insert_nodes(nodos)
            logger.info("Indexado: %s (%d nodos)", os.path.basename(ruta), len(nodos))
            return True

        except PermissionError:
            logger.error("Archivo bloqueado por otro proceso: %s", ruta)
        except Exception as e:
            logger.error("Error procesando %s: %s", ruta, e, exc_info=True)
        return False

    def indexar_todo(self, forzar: bool = False) -> int:
        """Barrido completo del vault. Salta lo que no ha cambiado.

        Args:
            forzar: reindexa todo aunque el mtime coincida.

        Returns:
            Número de archivos indexados.
        """
        indexados_previos = self._mtimes_indexados()
        conocidos = {} if forzar else indexados_previos
        indexados = omitidos = 0
        en_disco = set()

        for carpeta, _, archivos in os.walk(VAULT_PATH):
            for nombre in archivos:
                if not nombre.lower().endswith(".md"):
                    continue
                ruta = os.path.join(carpeta, nombre)
                ruta_norm = normalizar(ruta)
                en_disco.add(ruta_norm)

                previo = conocidos.get(ruta_norm)
                if previo is not None and abs(previo - os.path.getmtime(ruta)) < 1e-6:
                    omitidos += 1
                    continue

                if self.indexar_archivo(ruta):
                    indexados += 1

        # Vectores de archivos que ya no existen. Ocurre cuando se borra una
        # memoria con el demonio apagado: sin esta purga quedarían huérfanos y
        # Perseo seguiría "recordando" algo que el usuario eliminó del vault.
        huerfanos = set(indexados_previos) - en_disco
        for ruta in huerfanos:
            self.eliminar(ruta)
            logger.info("Purgado del índice (ya no existe): %s", os.path.basename(ruta))

        logger.info(
            "Barrido completo: %d indexados, %d sin cambios, %d purgados.",
            indexados, omitidos, len(huerfanos),
        )
        return indexados


class ObsidianEventHandler(FileSystemEventHandler):
    """Encola los cambios; el trabajo real lo hace un hilo aparte.

    La versión anterior llamaba a `time.sleep(1)` dentro del propio callback,
    que se ejecuta en el hilo del observer: durante ese segundo no se procesaba
    ningún otro evento. Ver H-19.
    """

    def __init__(self, indexador: Indexador):
        super().__init__()
        self.indexador = indexador
        self._pendientes: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._parar = threading.Event()
        self._worker = threading.Thread(target=self._procesar_pendientes, daemon=True)
        self._worker.start()

    def _encolar(self, ruta: str) -> None:
        if ruta.lower().endswith(".md"):
            with self._lock:
                self._pendientes[ruta] = time.time()

    def _procesar_pendientes(self) -> None:
        while not self._parar.is_set():
            ahora = time.time()
            with self._lock:
                maduros = [
                    r for r, t in self._pendientes.items()
                    if ahora - t >= RETARDO_ANTIRREBOTE_S
                ]
                for ruta in maduros:
                    del self._pendientes[ruta]

            for ruta in maduros:
                if os.path.isfile(ruta):
                    self.indexador.indexar_archivo(ruta)
                else:
                    self.indexador.eliminar(ruta)
                    logger.info("Eliminado del índice: %s", os.path.basename(ruta))

            self._parar.wait(0.5)

    def detener(self) -> None:
        self._parar.set()

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._encolar(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._encolar(event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._encolar(event.src_path)


def inicializar_sistema_rag() -> Indexador:
    """Prepara el modelo de embeddings, ChromaDB y el índice."""
    configurar_embeddings()

    os.makedirs(DB_PATH, exist_ok=True)
    db = chromadb.PersistentClient(path=DB_PATH)
    coleccion = db.get_or_create_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=coleccion)
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=StorageContext.from_defaults(vector_store=vector_store)
    )
    return Indexador(index, coleccion)


def iniciar_demonio() -> None:
    """Barrido inicial del vault y vigilancia continua."""
    logger.info("Inicializando sistema RAG...")

    # No se crea el vault si no existe: la versión anterior lo hacía, y si el
    # demonio se arrancaba desde otro directorio acababa vigilando una carpeta
    # vacía recién creada mientras las herramientas escribían en la de verdad.
    if not os.path.isdir(VAULT_PATH):
        logger.critical("El vault no existe: %s", VAULT_PATH)
        logger.critical("Cree la carpeta o defina OBSIDIAN_VAULT_PATH.")
        return

    try:
        indexador = inicializar_sistema_rag()
    except Exception as e:
        logger.critical("Falla al inicializar RAG: %s", e)
        return

    logger.info("Barrido inicial de %s...", VAULT_PATH)
    indexador.indexar_todo()

    manejador = ObsidianEventHandler(indexador)
    observer = Observer()
    observer.schedule(manejador, VAULT_PATH, recursive=True)
    observer.start()
    logger.info("Vigilancia activa sobre %s", VAULT_PATH)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Detenido por el usuario.")
    finally:
        manejador.detener()
        observer.stop()
        observer.join()


if __name__ == "__main__":
    import sys

    if "--solo-barrido" in sys.argv:
        # Indexa y sale. Útil para poblar la base sin dejar el demonio corriendo.
        if not os.path.isdir(VAULT_PATH):
            logger.critical("El vault no existe: %s", VAULT_PATH)
            raise SystemExit(1)
        indexador = inicializar_sistema_rag()
        indexador.indexar_todo(forzar="--forzar" in sys.argv)
    else:
        iniciar_demonio()
