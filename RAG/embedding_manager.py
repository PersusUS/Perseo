"""Configuración del modelo de embeddings del RAG de Perseo.

Intenta Gemini primero y cae a Ollama en local si no hay clave o si la red
falla. Ambos caminos se validan con una petición real, no solo comprobando que
exista la configuración.

Historial de esta pieza (ver H-02): durante meses el RAG no indexó nada, y una
de las tres causas estaba aquí. El modelo `text-embedding-004` fue retirado y
devolvía 404, así que la ruta de Gemini fallaba siempre y la de Ollama solo
funciona si hay un servidor local levantado. El resultado neto era que
`configurar_embeddings()` lanzaba RuntimeError en cualquier circunstancia.
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.base.embeddings.base import BaseEmbedding

from paths import RAIZ_PROYECTO

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# `gemini-embedding-001` es el modelo de embeddings vigente. NO volver a
# `text-embedding-004`: fue retirado y responde 404.
MODELO_GEMINI = "models/gemini-embedding-001"
MODELO_OLLAMA = "nomic-embed-text"
URL_OLLAMA = "http://localhost:11434"


def _leer_clave() -> Optional[str]:
    """Busca la clave de Gemini en el entorno y en los .env del proyecto.

    Se aceptan los dos nombres habituales para que la clave sea la misma que
    usa la aplicación de escritorio y no haya que mantenerla por duplicado.
    """
    load_dotenv()  # .env del directorio de trabajo, si lo hay
    load_dotenv(os.path.join(RAIZ_PROYECTO, ".env"))
    load_dotenv(os.path.join(RAIZ_PROYECTO, "RealTime", ".env"))

    for nombre in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "VITE_GEMINI_API_KEY"):
        valor = os.getenv(nombre)
        if valor:
            logger.info("Clave de embeddings tomada de %s.", nombre)
            return valor
    return None


def configurar_embeddings() -> BaseEmbedding:
    """Configura y valida el modelo de embeddings a nivel global.

    Returns:
        El modelo configurado, ya asignado a `Settings.embed_model`.

    Raises:
        RuntimeError: si ni Gemini ni Ollama están disponibles.
    """
    clave = _leer_clave()

    if clave:
        try:
            logger.info("Inicializando GoogleGenAIEmbedding (%s)...", MODELO_GEMINI)
            from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

            embed_model = GoogleGenAIEmbedding(model_name=MODELO_GEMINI, api_key=clave)
            embed_model.get_text_embedding("test")  # validación real de red y cuota
            Settings.embed_model = embed_model
            logger.info("GoogleGenAIEmbedding configurado y validado.")
            return embed_model
        except Exception as e:
            logger.error("Fallo en la validación de GoogleGenAIEmbedding: %s", e)
            logger.warning("Iniciando fallback a OllamaEmbedding...")
    else:
        logger.warning(
            "No se encontró clave de API (GOOGLE_API_KEY / GEMINI_API_KEY). "
            "Iniciando fallback a OllamaEmbedding..."
        )

    try:
        from llama_index.embeddings.ollama import OllamaEmbedding

        embed_model = OllamaEmbedding(model_name=MODELO_OLLAMA, base_url=URL_OLLAMA)
        embed_model.get_text_embedding("test")
        Settings.embed_model = embed_model
        logger.info("OllamaEmbedding configurado como fallback.")
        return embed_model
    except Exception as e:
        logger.critical("No se pudo inicializar el fallback de Ollama: %s", e)
        raise RuntimeError(
            "No hay modelos de embedding disponibles. Defina GEMINI_API_KEY "
            f"o levante Ollama en {URL_OLLAMA} con el modelo '{MODELO_OLLAMA}'."
        ) from e


if __name__ == "__main__":
    modelo = configurar_embeddings()
    vector = modelo.get_text_embedding("prueba de conectividad")
    print(f"OK: {type(modelo).__name__} operativo, {len(vector)} dimensiones.")
