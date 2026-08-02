import os
import sys
import logging
from datetime import datetime

# El paquete RAG no está instalado, se importa por ruta. Las rutas del vault
# viven en un único sitio para que el indexador vigile exactamente la carpeta
# donde esta herramienta escribe. Ver H-22.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "RAG"))

from paths import VAULT_PATH  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MEMORIAS_FOLDER = os.path.join(VAULT_PATH, "Memorias_Sistema")
CONVERSACIONES_FOLDER = os.path.join(VAULT_PATH, "Conversaciones")


def _nombre_seguro(texto: str) -> str:
    """Deja solo caracteres válidos para un nombre de archivo."""
    return "".join(c for c in texto if c.isalnum() or c in (" ", "_", "-")).strip()

def guardar_recuerdo(entidad: str, descripcion_visual: str, contexto: str, tags: list = None) -> str:
    """
    Guarda un recuerdo en el vault de Obsidian como archivo Markdown.
    
    Args:
        entidad (str): El nombre de la persona, objeto o concepto (ej. 'Aurelio').
        descripcion_visual (str): La descripción visual detallada extraída de la cámara.
        contexto (str): Contexto relacional o información adicional (ej. 'Amigo del usuario').
        tags (list): Lista de etiquetas opcionales.

    Returns:
        str: Mensaje de confirmación o error.
    """
    try:
        # Asegurar que el directorio de memorias existe
        os.makedirs(MEMORIAS_FOLDER, exist_ok=True)
        
        filename = _nombre_seguro(entidad)
        file_path = os.path.join(MEMORIAS_FOLDER, f"{filename}.md")
        
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tags_str = ", ".join(tags) if tags else "memoria"
        
        content = f"""---
tipo: memoria_episodica
entidad: {entidad}
fecha_creacion: {date_str}
tags: [{tags_str}]
---
# {entidad}

## Contexto
{contexto}

## Descripción Visual (Patrón Físico)
{descripcion_visual}

---
*Generado automáticamente por el motor de memoria de Perseo el {date_str}*
"""
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        logger.info(f"Recuerdo guardado exitosamente: {file_path}")
        return f"Éxito: Recuerdo sobre '{entidad}' guardado correctamente. Ya está disponible en la memoria a largo plazo."
    except Exception as e:
        logger.error(f"Error al guardar recuerdo: {e}")
        return f"Error al guardar el recuerdo: {str(e)}"

def guardar_conversacion(mensajes: list, titulo: str = "") -> str:
    """Guarda una conversación como Markdown en el vault.

    Se escribe dentro del vault a propósito, y no en una base de datos aparte:
    así el indexador del RAG la recoge sin hacer nada más, y Perseo puede
    recordar en sesiones futuras lo que se habló en ésta. Ver H-07.

    Args:
        mensajes: lista de {"tipo": "ai"|"user"|"system", "texto": "..."}.
        titulo: título opcional; si falta, se usa la fecha y la hora.

    Returns:
        Mensaje de confirmación o error.
    """
    try:
        utiles = [
            m for m in mensajes
            if m.get("tipo") in ("ai", "user") and (m.get("texto") or "").strip()
        ]
        if not utiles:
            return "Aviso: la conversación no tenía contenido que guardar."

        os.makedirs(CONVERSACIONES_FOLDER, exist_ok=True)

        ahora = datetime.now()
        marca = ahora.strftime("%Y-%m-%d %H-%M")
        nombre = _nombre_seguro(titulo) or f"Conversación {marca}"
        file_path = os.path.join(CONVERSACIONES_FOLDER, f"{nombre}.md")

        lineas = [
            "---",
            "tipo: conversacion",
            f"fecha: {ahora.strftime('%Y-%m-%d %H:%M:%S')}",
            f"mensajes: {len(utiles)}",
            "tags: [conversacion, perseo]",
            "---",
            f"# {nombre}",
            "",
        ]
        for m in utiles:
            quien = "Perseo" if m["tipo"] == "ai" else "Señor Persus"
            lineas.append(f"**{quien}:** {m['texto'].strip()}")
            lineas.append("")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lineas))

        logger.info("Conversación guardada: %s", file_path)
        return f"Éxito: conversación guardada en '{nombre}.md' ({len(utiles)} mensajes)."
    except Exception as e:
        logger.error("Error al guardar la conversación: %s", e)
        return f"Error al guardar la conversación: {e}"


if __name__ == "__main__":
    # Test manual
    print(guardar_recuerdo(
        entidad="Aurelio_Test", 
        descripcion_visual="Hombre con gafas azules y barba.", 
        contexto="Amigo de pruebas del sistema."
    ))