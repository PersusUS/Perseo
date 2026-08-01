import os
import logging

logger = logging.getLogger(__name__)

def controlar_pc(accion: str, parametro: str = "") -> str:
    """
    Controla funcionalidades del PC bajo Windows.
    
    Args:
        accion (str): El tipo de acción ('abrir_app' o navegar).
        parametro (str): El nombre del proceso, programa o URL a abrir.
    """
    try:
        if accion == "abrir_app":
            # Limpiamos el texto que envíe Gemini (por si envía puntos o espacios raros)
            objetivo = parametro.strip().lower()
            
            # El comando 'start' nativo de Windows es mágico. Puede abrir URLs en el 
            # navegador por defecto, y programas si están registrados en el PATH.
            if objetivo == "spotify":
                os.system("start spotify:") # URI directa
            else:
                os.system(f"start {objetivo}")
                
            return f"Éxito: Orden ejecutada en Windows. Intentando abrir '{objetivo}'."
            
        elif accion == "escribir_teclado":
            try:
                import pyautogui
                import time
                # Pequeño delay de cortesía por si acaba de abrir una app
                time.sleep(1)
                pyautogui.write(parametro, interval=0.01)
                return f"Éxito: Se ha tecleado el texto '{parametro}' en la ventana actual."
            except ImportError:
                return "Error: La librería 'pyautogui' no está instalada. Ejecuta 'pip install pyautogui'."
                
        elif accion == "atajo_teclado":
            try:
                import pyautogui
                # Convertimos el parámetro (ej: "ctrl,c") a lista de teclas
                teclas = [t.strip() for t in parametro.split(",")]
                pyautogui.hotkey(*teclas)
                return f"Éxito: Se ha ejecutado el atajo de teclado '{parametro}'."
            except ImportError:
                return "Error: La librería 'pyautogui' no está instalada. Ejecuta 'pip install pyautogui'."
                
        elif accion == "volumen":
            # Usar PowerShell de fondo para controlar el volumen en Windows de forma pasiva 
            # Parametro debe ser "subir", "bajar" o "mutear"
            if parametro == "subir":
                for _ in range(5): # Aproximadamente +10%
                    os.system("nircmd.exe changesysvolume 6553") # Alternativa pro o se puede usar PowerShell: os.system("powershell -c (new-object -com wscript.shell).SendKeys([char]175)")
            else:
                os.system("powershell -c (new-object -com wscript.shell).SendKeys([char]173)") # bajar o mutear (apróx logic)
            # Para esto recomiendo pyautogui.press('volumeup') que es mucho más limpio:
            try:
                import pyautogui
                if parametro == "subir":
                    pyautogui.press('volumeup', presses=5)
                elif parametro == "bajar":
                    pyautogui.press('volumedown', presses=5)
                elif parametro == "mutear":
                    pyautogui.press('volumemute')
                return f"Éxito: Acción de volumen '{parametro}' enviada al sistema."
            except ImportError:
                pass
            
            return f"Éxito: Comandos de volumen intentados (se recomienda instalar pyautogui para más fiabilidad)."

        elif accion == "mover_raton":
            try:
                import pyautogui
                coords = parametro.split(",")
                if len(coords) == 2:
                    x, y = int(coords[0].strip()), int(coords[1].strip())
                    pyautogui.moveTo(x, y, duration=0.5)
                    return f"Éxito: Ratón movido a las coordenadas ({x}, {y})."
                return "Error: Formato de coordenadas inválido. Debe ser 'x,y'."
            except Exception as e:
                return f"Error moviendo ratón: {str(e)}"
                
        elif accion == "click_raton":
            try:
                import pyautogui
                tipo = parametro.strip().lower()
                if tipo == "derecho":
                    pyautogui.rightClick()
                elif tipo == "doble":
                    pyautogui.doubleClick()
                else:
                    pyautogui.click()
                return f"Éxito: Click '{tipo}' ejecutado en la posición actual."
            except Exception as e:
                return f"Error haciendo click: {str(e)}"
                
        elif accion == "buscar_youtube":
            try:
                import urllib.parse
                # Codificamos la búsqueda (ejemplo: "Mozart Requiem" -> "Mozart%20Requiem")
                termino = urllib.parse.quote(parametro.strip())
                
                # Le pedimos que abra directamente la URL de resultados de YouTube
                url_busqueda = f"https://www.youtube.com/results?search_query={termino}"
                os.system(f"start {url_busqueda}")
                
                # NOTA: Para hacer "auto-play" del primer video, podríamos requerir Selenium/Puppeteer, 
                # pero simplemente con que Gemini use 'atajo_teclado' luego de esto (Darle al botón TAB 
                # varias veces y ENTER en la pantalla) o click_raton visualmente, será suficiente.
                
                return f"Éxito: Se ha abierto YouTube buscando '{parametro}'."
            except Exception as e:
                return f"Error buscando en YouTube: {str(e)}"

        else:
            return f"Error: Acción '{accion}' no soportada por el momento."
            
    except Exception as e:
        logger.error(f"Error controlando PC: {e}")
        return f"Error del sistema al intentar ejecutar la acción: {str(e)}"

if __name__ == "__main__":
    # Prueba manual
    print(controlar_pc("abrir_app", "https://youtube.com"))
