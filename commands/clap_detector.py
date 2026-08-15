import sounddevice as sd
import numpy as np
import time
import subprocess
import os
import sys
import threading
import speech_recognition as sr
import pygame
import json

# Inicializamos el mixer de Pygame silenciosamente
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
try:
    pygame.mixer.init()
except:
    pass

# --- Configuración del Detector ---
# Ajusta este THRESHOLD dependiendo de la sensibilidad de tu micrófono
# Un valor común para un aplauso claro a medio metro es 15.0 - 30.0
THRESHOLD = 20.0  

CLAP_MIN_DELAY = 0.25  # Aumentado (antes 0.2) para evitar que el eco cuente como segundo aplauso
CLAP_MAX_DELAY = 1.3  # Tiempo máximo (segundos) para considerar que son 2 aplausos seguidos
COOLDOWN = 15.0       # Aumentado el cooldown a 15 segundos para bloquear gatillos fantasma

# Variables de estado
lap_count = 0
last_clap_time = 0
last_trigger_time = 0

# Variables de voz
awaiting_voice = False
voice_buffer = []
voice_frames_needed = 0

def is_app_running():
    try:
        # Revisa si la aplicación ya está viva en los procesos de Windows
        output = subprocess.check_output('tasklist /FI "IMAGENAME eq temp-app.exe"', shell=True).decode(errors='ignore')
        if "temp-app.exe" in output.lower():
            return True
            
        # También comprobamos Node por si el servidor Vite está levantado (Port 1420 ocupado) 
        # pero Tauri GUI no hubiera salido aún del todo
        # Es un bloqueo conservador
        return False
    except Exception:
        return False

def run_perseo_and_cleanup(realtime_path):
    print(f"[*] Levantando Perseo en: {realtime_path}")
    
    # Arrancamos npm run tauri dev
    process = subprocess.Popen(["npm", "run", "tauri", "dev"], cwd=realtime_path, shell=True)
    
    # 1. Esperamos a que Perseo inicie completamente leyendo el tasklist (hasta 120 segundos)
    app_started = False
    for _ in range(60):
        try:
            output = subprocess.check_output('tasklist /FI "IMAGENAME eq temp-app.exe"', shell=True).decode(errors='ignore')
            if "temp-app.exe" in output.lower():
                app_started = True
                print("[*] Interfaz gráfica de Perseo detectada.")
                
                # PARAR LA MÚSICA cuando aparezca la app gráfica (inicia la llamada)
                try:
                    if pygame.mixer.music.get_busy():
                        pygame.mixer.music.fadeout(1000)
                except:
                    pass
                    
                print("[*] Analizando cierre...")
                break
        except Exception:
            pass
        time.sleep(2)
        if process.poll() is not None:
            break
            
    # 2. Si la app gráfica inició, nos quedamos en bucle infinito revisando cuando se cierre
    if app_started:
        while True:
            try:
                output = subprocess.check_output('tasklist /FI "IMAGENAME eq temp-app.exe"', shell=True).decode(errors='ignore')
                if "temp-app.exe" not in output.lower():
                    print("\n[*] La ventana de Perseo se ha cerrado. Procediendo con el exterminio residual...")
                    break
            except Exception:
                break
            time.sleep(2)
    else:
        print("[!] No se detectó la ventana gráfica de Perseo en un límite de 2 minutos.")
        
    # 3. Limpieza: Matar el proceso principal root y todo subproceso residual
    try:
        # Matamos todo el árbol desde el CMD que lanzó NPM
        subprocess.call(f"taskkill /F /T /PID {process.pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        pass
        
    try:
        # Matamos forzosamente cualquier cosa ocupando el Puerto 1420 (Vite zombie)
        os.system('FOR /F "tokens=5" %a in (\'netstat -aon ^| findstr :1420\') do taskkill /F /PID %a >nul 2>&1')
        print("[*] Entorno totalmente saneado. Listo para el próximo encendido.")
    except:
        pass

def trigger_action():
    print("\n[!] ¡Activando Comando de Emergencia! Encendiendo...")
    
    # Señal de auto-conexión: un fichero marcador en la raíz del proyecto que la
    # aplicación borra al leerlo. Antes se escribía dentro de src/autocall.json,
    # que React importaba estáticamente: Vite congelaba el valor al compilar (el
    # disparo no funcionaba en producción) y nadie lo devolvía a false, así que
    # toda apertura manual entraba en llamada sola. Ver H-09.
    #
    # Se deja **siempre**, también con la app abierta. Desde que Perseo vive en
    # la bandeja del sistema, el caso normal es que ya esté corriendo: Rust
    # vigila este fichero, saca la ventana del escondite y entra en llamada.
    # Antes se salía antes de escribirlo, así que con la app abierta la palabra
    # clave no hacía absolutamente nada.
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    realtime_path = os.path.join(base_dir, "RealTime")
    marcador = os.path.join(base_dir, ".perseo-autollamada")
    try:
        with open(marcador, 'w', encoding='utf-8') as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
        print("[*] Señal de auto-conexión depositada.")
    except Exception as e:
        print(f"[-] No se pudo dejar la señal de auto-conexión: {e}")

    if is_app_running():
        print("[!] Perseo ya está corriendo: la señal basta, no se abre otra instancia.")
        return

    # 1. Abrir animacion de carga
    script_dir = os.path.dirname(os.path.abspath(__file__))
    splash_path = os.path.join(script_dir, "loading_splash.py")
    
    # Usamos pythonw para que la ventana de carga corra silenciosa por detras sin robar terminal
    pythonw_exe = sys.executable.replace("python.exe", "pythonw.exe")
    try:
        subprocess.Popen([pythonw_exe, splash_path])
    except Exception as e:
        print(f"[x] Error al mostrar carga: {e}")

    # 2. Abrir RealTime de Perseo y monitorizar cierre
    # Calculamos la ruta absoluta de la carpeta "RealTime" basándonos en la ubicación de este script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    realtime_path = os.path.join(base_dir, "RealTime")
    
    try:
        # Lanzamos el comando en un hilo separado para que el detector de aplausos siga funcionando 100% independiente
        threading.Thread(target=run_perseo_and_cleanup, args=(realtime_path,), daemon=True).start()
    except Exception as e:
        print(f"[x] Error al iniciar hilo de Perseo: {e}")

    # 3. Reproducir Opening localmente con PyGame
    script_dir = os.path.dirname(os.path.abspath(__file__))
    opening_path = os.path.join(script_dir, "opening.mp3")
    
    if os.path.exists(opening_path):
        try:
            print("[*] Reproduciendo opening base...")
            pygame.mixer.music.load(opening_path)
            pygame.mixer.music.set_volume(0.6)
            pygame.mixer.music.play()
        except:
            print("[-] No se pudo reproducir la cancion, asegurate de tener un mp3 compatible.")
    else:
        print("[i] (No se encontró el archivo opening.mp3 en la carpeta commands)")

def audio_callback(indata, frames, time_info, status):
    global lap_count, last_clap_time, last_trigger_time, awaiting_voice, voice_buffer, voice_frames_needed
    
    # Si estamos escuchando tu frase "Perseo", guardamos audio puro sin aplicar lógica de aplausos
    if awaiting_voice:
        if len(voice_buffer) < voice_frames_needed:
            voice_buffer.extend(indata[:, 0])
        return

    # Si detecta errores de desbordamiento en el audio de entrada
    if status:
        pass

    # Evitar procesamiento si acabamos de ejecutar el comando (cooldown)
    now = time.time()
    if now - last_trigger_time < COOLDOWN:
        return

    # Calcular el volumen (norma RMS)
    volume_norm = np.linalg.norm(indata) * 10
    
    # Si pasó mucho tiempo desde el primer aplauso, reiniciar el contador
    if lap_count == 1 and (now - last_clap_time) > CLAP_MAX_DELAY:
        lap_count = 0
        
    # Detectar pico de sonido
    if volume_norm > THRESHOLD:
        if now - last_clap_time > CLAP_MIN_DELAY:
            lap_count += 1
            last_clap_time = now
            print(f"~ Posible aplauso detectado ({lap_count}/2) - Volumen: {volume_norm:.1f}")
            
            if lap_count == 2:
                print("\n[?] ¡Doble aplauso detectado! Tienes 3 segundos. Habla ahora y di 'Perseo'...")
                lap_count = 0
                
                # 3 segundos a 44100 muestras por segundo (frecuencia de grabación)
                voice_buffer.clear()
                voice_frames_needed = int(3 * 44100)
                awaiting_voice = True

def process_voice_buffer():
    global voice_buffer, last_trigger_time
    
    print("    [~] Subiendo audio a la Inteligencia Artificial (Google Speech)...")
    
    import speech_recognition as sr
    
    # Transformamos el buffer continuo recogido por el callback en formato estándar
    import numpy as np
    audio_array = np.array(voice_buffer, dtype=np.float32)
    
    # Sounddevice emite en float32 nativamente. La IA lo necesita en un int16 estandarizado
    audio_int16 = (audio_array * 32767).astype(np.int16)
    
    # Construimos la envoltura AudioData para la librería SpeechRecognition
    audio_data = sr.AudioData(audio_int16.tobytes(), 44100, 2)
    recognizer = sr.Recognizer()
    
    try:
        # Intentamos hasta 3 veces por si Google nos corta la conexión (WinError 10054)
        text = ""
        for attempt in range(3):
            try:
                text = recognizer.recognize_google(audio_data, language="es-ES", show_all=False)
                break
            except sr.UnknownValueError:
                print("    [-] El audio era ininteligible. Secuencia anulada. (Si había ruido, intente otra vez el aplauso)")
                print("----------------------------------------")
                print("- Esperando doble aplauso de nuevo...")
                return
            except (sr.RequestError, Exception) as e:
                if attempt == 2:
                    raise e
                print(f"    [!] Conexión inestable con Google (Intento {attempt+1}/3... Reintentando...)")
                time.sleep(1.5)
                
        print(f"    [+] Transcripción captada: '{text}'")
        
        # Filtro de activación
        texto=text.lower()
        if "perseo" in texto:
            last_trigger_time = time.time()
            trigger_action()
        else:
            print("    [x] Secuencia anulada. La voz no dijo 'Perseo'. Volviendo a escuchar aplausos...")
            
    except sr.RequestError as e:
        print(f"    [!] Error de red persistente. No pudimos comprobar la frase: {e}")
    except Exception as general_error:
        print(f"    [!] Error interno general: {general_error}")
        
    print("----------------------------------------")
    print("- Esperando doble aplauso de nuevo...")

def start_listening():
    global awaiting_voice
    
    print("========================================")
    print(" Perseo Clap-Listener Iniciado en BG")
    print("========================================")
    print("- Esperando doble aplauso...")
    print("- Consumo de CPU minimizado.")
    
    try:
        # Se usa un samplerate bajo (44.1kHz es el mas generico compatible con windows) y captura mono (channels=1)
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=44100):
            while True:
                # Si el buffer de voz se ha llenado y el modo voz está activo, mandamos a analizar
                # Lo lanzamos en un TREAD para no bloquear el detector y evitar el Atasco Mágico de Google
                if awaiting_voice and len(voice_buffer) >= voice_frames_needed:
                    awaiting_voice = False # Desactiva inmediatamente la bandera para no relanzar threads
                    threading.Thread(target=process_voice_buffer, daemon=True).start()
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nDetenido por el usuario.")
    except Exception as e:
        print(f"\n[x] Error critico en el micrófono: {e}")
        print("    ! Asegúrate de tener al menos un micrófono enchufado.")

if __name__ == "__main__":
    start_listening()