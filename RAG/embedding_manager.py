import os
import logging
from typing import Optional
from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def configurar_embeddings() -> BaseEmbedding:
    """
    Configura y valida el modelo de embeddings a nivel global.
    
    Intenta inicializar el modelo de Gemini utilizando la API key. Si la validación falla
    por problemas de red o cuota, implementa un fallback automatizado utilizando Ollama en local.
    
    Returns:
        BaseEmbedding: El modelo de embeddings configurado y validado.
    """
    load_dotenv()
    
    google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
    
    if google_api_key:
        try:
            logger.info("Intentando inicializar GeminiEmbedding (text-embedding-004)...")
            embed_model = GeminiEmbedding(
                model_name="models/text-embedding-004",
                api_key=google_api_key
            )
            # Validación de red y cuota
            embed_model.get_text_embedding("test")
            Settings.embed_model = embed_model
            logger.info("GeminiEmbedding configurado y validado exitosamente.")
            return embed_model
            
        except Exception as e:
            logger.error(f"Fallo en la validación de GeminiEmbedding: {e}")
            logger.warning("Iniciando fallback a OllamaEmbedding...")
    else:
        logger.warning("GOOGLE_API_KEY no encontrada. Iniciando fallback a OllamaEmbedding...")

    try:
        embed_model = OllamaEmbedding(
            model_name="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        # Validación de instancia local
        embed_model.get_text_embedding("test")
        Settings.embed_model = embed_model
        logger.info("OllamaEmbedding configurado como fallback exitosamente.")
        return embed_model
    except Exception as e:
        logger.critical(f"Fallo crítico: No se pudo inicializar el modelo de fallback Ollama: {e}")
        raise RuntimeError("No hay modelos de embedding disponibles.") from e

if __name__ == "__main__":
    configurar_embeddings()