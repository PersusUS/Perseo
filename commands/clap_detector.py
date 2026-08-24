import sounddevice as sd
import numpy as np
import time
import subprocess
import os
import sys
import threading
import pygame

import presencia
from palabra_clave import crear_detector

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
# descalibra el detector de aplausos aunque no lo parezca. El motor local necesita
# 16 kHz, pero eso se resuelve remuestreando en palabra_clave.py, no aquí; a Google
# se le manda tal cual, que acepta 44.100.
SAMPLERATE = 44100
VOICE_SECONDS = 3.0  # Cuánto se graba tras el doble aplauso para buscar la palabra

# El detector se construye una vez y se precarga al arrancar: con el motor local
# el modelo tarda cerca de un segundo en cargar, y ese segundo no puede caer entre
# el aplauso y la respuesta. Cuál de los dos motores sale de PERSEO_PALABRA_MOTOR;
# el de por defecto es Google, que es el único que reconoce «Perseo».
detector_palabra = crear_detector()

# Variables de estado
lap_count = 0
last_clap_time = 0
last_trigger_time = 0

# Variables de voz
awaiting_voice = False
voice_buffer = []
voice_frames_needed = 0

def is_app_running():
    """Si Perseo ya está abierto.

    Se pregunta por el PID que deja la propia aplicación, no por el nombre del
    ejecutable en el `tasklist`. Atar esto al nombre del binario era lo que hacía
    que renombrar el paquete Rust rompiera el detector en silencio. Ver H-21.
    """
    try:
        return presencia.app_viva()
    except Exception as e:
        # Si no se puede averiguar, se contesta que no: es lo conservador —la
        # señal de autollamada basta si ya estaba abierta—, pero se deja dicho.
        # Este camino explica un "se abrió una segunda instancia" que si no
        # parece cosa de magia.
        print(f"[-] No se pudo comprobar si Perseo esta abierto: {e}")
        return False

def levantar_app():
    """Abre la app de voz por la misma puerta que `perseo on`.

    Antes esto lanzaba `npm run tauri dev`, que es modo desarrollo: minutos de
    compilación en frío, y el bucle que esperaba la ventana mataba el árbol a
    los dos minutos — la app jamás llegó a abrirse y el síntoma era «aplaudí y
    no pasó nada» (H-56). La app de verdad es el binario construido y sellado,
    el mismo que abre `perseo on`; ahí viven el despegue del job (H-53), el
    aviso si la construcción está vieja y el mensaje si no está construida.
    """
    import perseo

    perseo.arrancar_app()


def parar_musica_cuando_abra_la_app(tope_segundos: float = 90.0):
    """La intro suena mientras Perseo despierta, y calla cuando ya está despierto.

    Antes esta parada vivía dentro del bucle que vigilaba `npm run tauri dev`;
    al pasar el arranque a la app construida (H-56) se quedó fuera y la canción
    seguía entera por encima de la llamada. Se pregunta por la marca de
    presencia —lo mismo que mira el splash— y con tope, para que un arranque
    fallido no deje la música sonando sola en el salón.
    """
    empezó = time.time()
    while time.time() - empezó < tope_segundos:
        try:
            if presencia.app_viva():
                print("[*] Interfaz gráfica de Perseo detectada; callando la intro.")
                break
        except Exception:
            # Un fallo suelto al preguntar no cancela la espera: quedan vueltas.
            pass
        time.sleep(0.5)
    else:
        print("[!] La app no dejó marca de presencia; se calla la intro igualmente.")

    try:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.fadeout(1000)
    except Exception as e:
        print(f"[-] No se pudo parar la música de apertura: {e}")

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
    marcador = os.path.join(base_dir, ".perseo-autollamada")
    try:
        # VACÍO, no una fecha: el contenido del marcador es el MOTIVO de la
        # llamada y quien lo lee distingue así los dos caminos — vacío es el
        # aplauso (entra en llamada sin más) y con texto un subagente que
        # terminó (timbre y cartel «PERSEO LLAMA»). La marca de tiempo que
        # había aquí antes viajaba como motivo y Perseo entraba al aplauso
        # pasando por la notificación de aviso: los dos rituales mezclados.
        with open(marcador, 'w', encoding='utf-8') as f:
            f.write('')
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

    # 2. Abrir la app de voz construida. Vuelve enseguida: el arranque es un
    #    Popen despegado, no una espera.
    try:
        levantar_app()
    except Exception as e:
        print(f"[x] Error al iniciar Perseo: {e}")

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

    # 4. La intro calla cuando la app deje su marca de presencia — o al cabo
    #    del tope, para no quedársela si el arranque falla.
    threading.Thread(target=parar_musica_cuando_abra_la_app, daemon=True).start()

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

    print(f"    [~] Escuchando la palabra clave ({detector_palabra.descripcion()})...")

    try:
        arranque = time.time()
        dijo_la_palabra, puntuacion = detector_palabra.escuchar(voice_buffer, SAMPLERATE)
        tardanza = (time.time() - arranque) * 1000

        if dijo_la_palabra:
            print(f"    [+] Palabra clave reconocida ({puntuacion:.2f}, {tardanza:.0f} ms)")
            last_trigger_time = time.time()
            trigger_action()
        else:
            # Lo que se entendió, si el motor sabe decirlo: "no se dijo la palabra"
            # no ayuda a nadie, y "oí «apaga la luz»" explica la vez que no salta.
            oido = getattr(detector_palabra, "ultimo_texto", "")
            detalle = f'se oyó «{oido}»' if oido else f"{puntuacion:.2f} por debajo de {detector_palabra.puntuacion_minima:.2f}"
            print(f"    [x] Secuencia anulada: no se dijo la palabra clave "
                  f"({detalle}, {tardanza:.0f} ms). Volviendo a escuchar aplausos...")

    except Exception as error:
        # Aquí caben la red caída con el motor de Google, un modelo que no carga
        # con el local, y un fallo del remuestreo con cualquiera de los dos. Los
        # tres dejan al detector sin palabra clave, así que se dice en voz alta en
        # vez de tragárselo: un fallo silencioso aquí se ve desde fuera como
        # "Perseo ha dejado de responder".
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
        print(f"- Palabra clave: {detector_palabra.descripcion()} "
              f"({(time.time() - arranque) * 1000:.0f} ms)")
    except Exception as e:
        # No se sale: los aplausos siguen funcionando y el fallo se vuelve a
        # intentar en cada activación, que es donde se explica con detalle.
        print(f"[-] No se pudo preparar la palabra clave: {e}")
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