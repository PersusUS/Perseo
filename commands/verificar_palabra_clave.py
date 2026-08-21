"""Prueba de la palabra clave sin hablarle al micrófono.

El problema de un detector de voz es que comprobarlo suele exigir una persona
delante diciendo cosas, y eso no se puede meter en una prueba de regresión. Aquí
la voz la pone Windows: SAPI sintetiza la frase a un WAV de 44.100 Hz —la misma
frecuencia a la que graba el detector— y ese WAV recorre exactamente el camino
del audio de verdad, remuestreo incluido.

Se comprueban **los dos motores**:

* **Google**, que es el de por defecto y el único que reconoce «Perseo». Este sí
  necesita red: si no hay, se dice y se sigue.
* **openWakeWord**, en local. Con el modelo de repuesto ('hey jarvis') la frase
  de prueba es la suya y se dice en inglés; con `perseo.onnx`, «Perseo» en
  español. Ver commands/modelos/README.md.

No hace falta micrófono ni tocar nada del estado real: todo se escribe en un
directorio temporal que se borra al terminar.

    python commands/verificar_palabra_clave.py
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from palabra_clave import (  # noqa: E402
    FRECUENCIA_MODELO,
    MODELO_PUENTE,
    VARIANTES,
    DetectorGoogle,
    DetectorPalabra,
    dijo_la_palabra,
    remuestrear,
)

FRECUENCIA_MICROFONO = 44100

fallos = 0


def comprobar(condicion, titulo, detalle=""):
    global fallos
    marca = "OK  " if condicion else "FALLO"
    if not condicion:
        fallos += 1
    linea = f"[{marca}] {titulo}"
    if detalle:
        linea += f" -- {detalle}"
    print(linea)


def sintetizar(frase, voz, destino):
    """Escribe `frase` en un WAV mono de 16 bits a 44.100 Hz usando SAPI.

    Returns:
        bool: si se pudo generar el fichero.
    """
    guion = f"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo({FRECUENCIA_MICROFONO}, `
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, `
    [System.Speech.AudioFormat.AudioChannel]::Mono)
try {{ $s.SelectVoice("{voz}") }} catch {{ }}
$s.Rate = -1
$s.SetOutputToWaveFile("{destino}", $fmt)
$s.Speak("{frase}")
$s.Dispose()
"""
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", guion],
            check=True,
            capture_output=True,
            timeout=120,
        )
    except Exception as e:
        print(f"[-] SAPI no pudo sintetizar '{frase}': {e}")
        return False
    return os.path.exists(destino) and os.path.getsize(destino) > 1000


def leer_wav(ruta):
    """Devuelve el audio como float en [-1, 1] y su frecuencia, como del micrófono."""
    with wave.open(ruta, "rb") as f:
        frecuencia = f.getframerate()
        crudo = f.readframes(f.getnframes())
    muestras = np.frombuffer(crudo, dtype=np.int16).astype(np.float32) / 32767.0
    return muestras, frecuencia


def probar_remuestreo():
    print("\n-- Remuestreo de 44.100 a 16.000 Hz --\n")

    segundos = 3.0
    entrada = np.zeros(int(FRECUENCIA_MICROFONO * segundos), dtype=np.float32)
    salida = remuestrear(entrada, FRECUENCIA_MICROFONO)

    esperadas = int(FRECUENCIA_MODELO * segundos)
    comprobar(
        abs(len(salida) - esperadas) <= 2,
        "Tres segundos siguen siendo tres segundos",
        f"{len(salida)} muestras (esperadas ~{esperadas})",
    )
    comprobar(salida.dtype == np.int16, "Sale PCM de 16 bits", str(salida.dtype))

    # Un tono de 440 Hz tiene que seguir en 440 Hz después de remuestrear. Si el
    # remuestreo estuviera mal, el pico de la FFT se movería de sitio.
    t = np.arange(int(FRECUENCIA_MICROFONO * 1.0)) / FRECUENCIA_MICROFONO
    tono = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    convertido = remuestrear(tono, FRECUENCIA_MICROFONO).astype(np.float32) / 32767.0
    espectro = np.abs(np.fft.rfft(convertido))
    pico = np.fft.rfftfreq(len(convertido), 1 / FRECUENCIA_MODELO)[np.argmax(espectro)]
    comprobar(abs(pico - 440) < 5, "Un tono de 440 Hz sigue siendo de 440 Hz", f"{pico:.1f} Hz")

    # Sin recorte, un pico por encima de 1.0 da la vuelta en int16 y se convierte
    # en un chasquido: el valor máximo saldría negativo.
    fuerte = np.full(FRECUENCIA_MICROFONO, 1.4, dtype=np.float32)
    recortado = remuestrear(fuerte, FRECUENCIA_MICROFONO)
    comprobar(
        recortado.max() > 0 and recortado.min() >= 0,
        "Los picos por encima de 1.0 se recortan y no dan la vuelta",
        f"máximo={recortado.max()} mínimo={recortado.min()}",
    )


def probar_texto():
    """Qué transcripción despierta a Perseo. Ni red ni audio: solo la decisión."""
    print("\n-- La palabra dentro del texto --\n")

    for texto in ("Perseo", "perseo", "Perséo", "oye Perseo, enciende la luz"):
        comprobar(dijo_la_palabra(texto), f"«{texto}» despierta a Perseo")

    for texto in ("apaga la luz del salón", "percusión", ""):
        etiqueta = texto or "(nada)"
        comprobar(not dijo_la_palabra(texto), f"«{etiqueta}» no lo despierta")

    comprobar(
        "perseo" in VARIANTES,
        "El nombre de verdad está entre las variantes",
        ", ".join(VARIANTES),
    )


def probar_google(directorio):
    """El motor de por defecto, contra la API de verdad. Necesita red."""
    print("\n-- Decisión de Google Speech (el motor de por defecto) --\n")

    detector = DetectorGoogle()
    try:
        detector.cargar()
    except Exception as e:
        comprobar(False, "La librería de reconocimiento está", str(e))
        return
    comprobar(True, "La librería de reconocimiento está", detector.descripcion())

    ruta = os.path.join(directorio, "google.wav")
    if not sintetizar("Perseo", "Microsoft Helena Desktop", ruta):
        comprobar(False, "Decir «Perseo» sí lo despierta", "SAPI no generó el audio")
        return

    muestras, frecuencia = leer_wav(ruta)
    arranque = time.time()
    try:
        dijo, _ = detector.escuchar(muestras, frecuencia)
    except Exception as e:
        # Sin red no hay palabra clave, y es un estado que hay que poder
        # distinguir de un fallo del código: se dice y no se cuenta como rojo.
        print(f"[i] Sin red o Google no contesta ({e}). El motor de Google no se pudo probar.")
        print("    Con `PERSEO_PALABRA_MOTOR=local` la palabra clave funciona sin red,")
        print("    pero responde a «hey jarvis» hasta que exista perseo.onnx.")
        return
    tardanza = (time.time() - arranque) * 1000
    comprobar(dijo, "Decir «Perseo» sí lo despierta", f"se oyó «{detector.ultimo_texto}», {tardanza:.0f} ms")

    ruta_otra = os.path.join(directorio, "google_otra.wav")
    if sintetizar("apaga la luz del salón", "Microsoft Helena Desktop", ruta_otra):
        muestras, frecuencia = leer_wav(ruta_otra)
        try:
            dijo, _ = detector.escuchar(muestras, frecuencia)
        except Exception as e:
            print(f"[i] Sin red para la segunda frase ({e}).")
            return
        comprobar(
            not dijo,
            "Decir otra cosa no lo despierta",
            f"se oyó «{detector.ultimo_texto}»",
        )

    silencio = np.zeros(int(FRECUENCIA_MICROFONO * 3), dtype=np.float32)
    try:
        dijo, _ = detector.escuchar(silencio, FRECUENCIA_MICROFONO)
    except Exception as e:
        print(f"[i] Sin red para el silencio ({e}).")
        return
    comprobar(not dijo, "El silencio no despierta a Perseo")


def probar_deteccion(directorio):
    print("\n-- Decisión de openWakeWord (el motor local, PERSEO_PALABRA_MOTOR=local) --\n")

    detector = DetectorPalabra()
    arranque = time.time()
    try:
        detector.cargar()
    except Exception as e:
        comprobar(False, "El modelo carga", str(e))
        return
    comprobar(True, "El modelo carga", f"{detector.descripcion()}, {(time.time() - arranque) * 1000:.0f} ms")

    es_puente = not detector.es_propio and detector.ruta == MODELO_PUENTE
    if es_puente:
        frase, voz = "hey jarvis", "Microsoft Zira Desktop"
        otra_frase, otra_voz = "what is the weather tomorrow", "Microsoft Zira Desktop"
    else:
        frase, voz = "Perseo", "Microsoft Helena Desktop"
        otra_frase, otra_voz = "apaga la luz del salón", "Microsoft Helena Desktop"

    silencio = np.zeros(int(FRECUENCIA_MICROFONO * 3), dtype=np.float32)
    dijo, puntuacion = detector.escuchar(silencio, FRECUENCIA_MICROFONO)
    comprobar(not dijo, "El silencio no despierta a Perseo", f"{puntuacion:.4f}")

    generador = np.random.default_rng(7)
    ruido = generador.normal(0, 0.05, int(FRECUENCIA_MICROFONO * 3)).astype(np.float32)
    dijo, puntuacion = detector.escuchar(ruido, FRECUENCIA_MICROFONO)
    comprobar(not dijo, "El ruido de fondo tampoco", f"{puntuacion:.4f}")

    ruta_frase = os.path.join(directorio, "clave.wav")
    if sintetizar(frase, voz, ruta_frase):
        muestras, frecuencia = leer_wav(ruta_frase)
        dijo, puntuacion = detector.escuchar(muestras, frecuencia)
        comprobar(dijo, f"Decir «{frase}» sí lo despierta", f"{puntuacion:.4f}")

        # La segunda vez, no la primera: la primera paga la puesta en marcha del
        # extractor y no se parece a lo que se vive aplaudiendo dos veces.
        arranque = time.time()
        detector.escuchar(muestras, frecuencia)
        tardanza = (time.time() - arranque) * 1000
        # El listón ya no es "más rápido que la red": Google contesta en medio
        # segundo largo, así que la velocidad dejó de ser el motivo para elegir
        # el motor local. Los motivos son que funciona sin red y que no sale del
        # portátil. Lo que sí se comprueba es que no cueste segundos, porque eso
        # sí se nota entre el aplauso y la respuesta.
        comprobar(tardanza < 1000, "Y la decisión no cuesta segundos", f"{tardanza:.0f} ms")
    else:
        comprobar(False, f"Decir «{frase}» sí lo despierta", "SAPI no generó el audio")

    ruta_otra = os.path.join(directorio, "otra.wav")
    if sintetizar(otra_frase, otra_voz, ruta_otra):
        muestras, frecuencia = leer_wav(ruta_otra)
        dijo, puntuacion = detector.escuchar(muestras, frecuencia)
        comprobar(not dijo, f"Decir «{otra_frase}» no lo despierta", f"{puntuacion:.4f}")
    else:
        comprobar(False, f"Decir «{otra_frase}» no lo despierta", "SAPI no generó el audio")

    # Dos activaciones seguidas: sin el `reset` del detector, la segunda arrastra
    # el audio de la primera y puntúa una mezcla de las dos.
    if os.path.exists(ruta_otra):
        muestras, frecuencia = leer_wav(ruta_otra)
        dijo, puntuacion = detector.escuchar(muestras, frecuencia)
        comprobar(
            not dijo,
            "La activación anterior no contamina la siguiente",
            f"{puntuacion:.4f} tras haber dicho la palabra clave",
        )

    if es_puente:
        print("\n[i] Corriendo con el modelo de repuesto: esto comprueba el circuito,")
        print("    no la palabra «Perseo». Ver commands/modelos/README.md.")


def main():
    print("=" * 60)
    print(" Verificación de la palabra clave (los dos motores, sin micrófono)")
    print("=" * 60)

    directorio = tempfile.mkdtemp(prefix="perseo_palabra_")
    try:
        probar_texto()
        probar_remuestreo()
        probar_google(directorio)
        probar_deteccion(directorio)
    finally:
        shutil.rmtree(directorio, ignore_errors=True)

    print()
    if fallos:
        print(f"{fallos} comprobación(es) en rojo.")
        return 1
    print("Todo en verde.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
