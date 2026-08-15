import sounddevice as sd
import numpy as np
import time
import subprocess
import os
import sys
import threading
import pygame

from palabra_clave import DetectorPalabra

# Inicializamos el mixer de Pygame silenciosamente
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
try:
    pygame.mixer.init()
except Exception as e:
    # Sin mixer se pierde la música de apertura, pero el detector funciona igual.
    # Se avisa porque un fallo silencioso aquí parecía "el mp3 no existe".
    print(f"[-] No se pudo iniciar el audio de Pygame: {e}")

# --- Configuración del Detector ---
# Ajusta este THRESHOLD dependiendo de la sensibilidad de tu micrófono
# Un valor común para un aplauso claro a medio metro es 15.0 - 30.0
THRESHOLD = 20.0  

CLAP_MIN_DELAY = 0.25  # Aumentado (antes 0.2) para evitar que el eco cuente como segundo aplauso
CLAP_MAX_DELAY = 1.3  # Tiempo máximo (segundos) para considerar que son 2 aplausos seguidos
COOLDOWN = 15.0       # Aumentado el cooldown a 15 segundos para bloquear gatillos fantasma

# 44.1 kHz es lo más compatible en Windows, y THRESHOLD está calibrado a esta
# frecuencia: la norma que decide si un ruido es un aplauso depende del tamaño de
# bloque, y el tamaño de bloque depende del samplerate. Cambiar este número
# descalibra el detector de aplausos aunque no lo parezca. openWakeWord necesita
# 16 kHz, pero eso se resuelve remuestreando en palabra_clave.py, no aquí.
SAMPLERATE = 44100
VOICE_SECONDS = 3.0  # Cuánto se graba tras el doble aplauso para buscar la palabra

# El modelo tarda cerca de un segundo en cargar, y ese segundo no puede caer
# entre el aplauso y la respuesta: se construye una vez y se precarga al arrancar.
detector_palabra = DetectorPalabra()

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
    except Exception as e:
        # Si `tasklist` falla no sabemos si la app está viva. Se contesta que no,
        # que es lo conservador —la señal de autollamada basta si ya estaba
        # abierta—, pero se deja dicho: este camino explica un "se abrió una
        # segunda instancia" que si no parece cosa de magia.
        print(f"[-] No se pudo consultar la lista de procesos: {e}")
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
                except Exception as e:
                    print(f"[-] No se pudo parar la música de apertura: {e}")

                print("[*] Analizando cierre...")
                break
        except Exception as e:
            # Un fallo suelto de `tasklist` no es motivo para rendirse: quedan
            # más vueltas del bucle. Se avisa por si falla en todas.
            print(f"[-] No se pudo comprobar si la app ya arrancó: {e}")
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
            except Exception as e:
                # Aquí sí se sale del bucle: si no podemos vigilar el cierre, es
                # mejor limpiar ahora que quedarse mirando para siempre.
                print(f"[-] Se pierde de vista la ventana de Perseo: {e}")
                break
            time.sleep(2)
    else:
        print("[!] No se detectó la ventana gráfica de Perseo en un límite de 2 minutos.")
        
    # 3. Limpieza: Matar el proceso principal root y todo subproceso residual
    try:
        # Matamos todo el árbol desde el CMD que lanzó NPM
        subprocess.call(f"taskkill /F /T /PID {process.pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"[-] No se pudo matar el árbol de procesos de npm: {e}")

    try:
        # Matamos forzosamente cualquier cosa ocupando el Puerto 1420 (Vite zombie)
        os.system('FOR /F "tokens=5" %a in (\'netstat -aon ^| findstr :1420\') do taskkill /F /PID %a >nul 2>&1')
        print("[*] Entorno totalmente saneado. Listo para el próximo encendido.")
    except Exception as e:
        # Un Vite zombi en el 1420 hace que el arranque siguiente falle sin decir
        # por qué. Que se sepa aquí ahorra media hora la próxima vez.
        print(f"[-] No se pudo liberar el puerto 1420: {e}")

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
        except Exception as e:
            print(f"[-] No se pudo reproducir la canción, asegúrate de tener un mp3 compatible: {e}")
    else:
        print("[i] (No se encontró el archivo opening.mp3 en la carpeta commands)")

def audio_callback(indata, frames, time_info, status):
    global lap_count, last_clap_time, last_trigger_time, awaiting_voice, voice_buffer, voice_frames_needed
    
    # Si estamos escuchando tu frase "Perseo", guardamos audio puro sin aplicar lógica de aplausos
    if awaiting_voice:
        if len(voice_buffer) < voice_frames_needed:
            voice_buffer.extend(indata[:, 0])
        return

    # Los desbordamientos del audio de entrada se ignoran a propósito, y este es
    # el único sitio del fichero donde callarse está justificado: esto corre en el
    # hilo del micrófono, y escribir por consola desde aquí provoca justo el
    # desbordamiento que se está reportando. No es el H-25.
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
                print(f"\n[?] ¡Doble aplauso detectado! Tienes {VOICE_SECONDS:.0f} segundos. Habla ahora y di 'Perseo'...")
                lap_count = 0
                
                voice_buffer.clear()
                voice_frames_needed = int(VOICE_SECONDS * SAMPLERATE)
                awaiting_voice = True

def process_voice_buffer():
    global voice_buffer, last_trigger_time

    print("    [~] Escuchando la palabra clave (openWakeWord, en local)...")

    try:
        arranque = time.time()
        dijo_la_palabra, puntuacion = detector_palabra.escuchar(voice_buffer, SAMPLERATE)
        tardanza = (time.time() - arranque) * 1000

        if dijo_la_palabra:
            print(f"    [+] Palabra clave reconocida ({puntuacion:.2f}, {tardanza:.0f} ms)")
            last_trigger_time = time.time()
            trigger_action()
        else:
            print(f"    [x] Secuencia anulada: no se dijo la palabra clave "
                  f"({puntuacion:.2f} por debajo de {detector_palabra.puntuacion_minima:.2f}, "
                  f"{tardanza:.0f} ms). Volviendo a escuchar aplausos...")

    except Exception as error:
        # Aquí caben un modelo que no carga, un .onnx corrupto o un fallo del
        # remuestreo. Cualquiera de los tres deja el detector sin palabra clave,
        # así que se dice en voz alta en vez de tragárselo: un fallo silencioso
        # aquí se ve desde fuera como "Perseo ha dejado de responder".
        print(f"    [!] No se pudo comprobar la palabra clave: {error}")

    print("----------------------------------------")
    print("- Esperando doble aplauso de nuevo...")

def start_listening():
    global awaiting_voice
    
    print("========================================")
    print(" Perseo Clap-Listener Iniciado en BG")
    print("========================================")

    # Se carga antes de abrir el micrófono, no la primera vez que aplaudes.
    try:
        arranque = time.time()
        detector_palabra.cargar()
        print(f"- Palabra clave en local: {detector_palabra.descripcion()} "
              f"({(time.time() - arranque) * 1000:.0f} ms)")
    except Exception as e:
        # No se sale: los aplausos siguen funcionando y el fallo se vuelve a
        # intentar en cada activación, que es donde se explica con detalle.
        print(f"[-] No se pudo cargar el modelo de la palabra clave: {e}")
        print("    El detector sigue en pie, pero no reconocerá la palabra.")

    print("- Esperando doble aplauso...")
    print("- Consumo de CPU minimizado.")

    try:
        # Ver el comentario de SAMPLERATE: esta frecuencia y el umbral del aplauso
        # van juntos. Captura mono (channels=1).
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLERATE):
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