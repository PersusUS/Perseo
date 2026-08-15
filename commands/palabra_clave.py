"""Reconocimiento de la palabra clave, en local y sin red.

Antes, los tres segundos que seguían al doble aplauso se subían a Google Speech,
se transcribían a texto y se buscaba la cadena "perseo" dentro. Eso significaba
mandar el audio del salón a un tercero, depender de la red en el momento exacto
en que quieres que Perseo te haga caso, y esperar entre uno y tres segundos a que
Google contestara — con reintentos cuando la conexión se caía a media petición.

Ahora la decisión la toma openWakeWord aquí mismo, sobre la CPU. Es un modelo
pequeño, corre en decenas de milisegundos, y no sale nada del portátil. La CPU es
además el único sitio donde cabe: los 6 GB de la RTX 4050 están comprometidos con
el modelo de 4B del router del núcleo.

Lo que NO cambia es el doble aplauso. La activación sigue pidiendo las dos cosas
—aplaudir y decir la palabra—, que es lo que hace usable un activador por sonido
sin falsos positivos constantes. Este módulo sustituye a Google, no al aplauso.

Configuración por entorno:

    PERSEO_MODELO_PALABRA   Ruta a un .onnx propio. Por defecto se busca
                            commands/modelos/perseo.onnx.
    PERSEO_UMBRAL_PALABRA   Puntuación mínima para dar la palabra por dicha.
                            Por defecto 0.5, que es el valor que recomienda
                            openWakeWord para sus modelos.
"""

import os

import numpy as np

# 16 kHz mono es lo único que come openWakeWord: el extractor de características
# está entrenado a esa frecuencia y no remuestrea por su cuenta.
FRECUENCIA_MODELO = 16000

UMBRAL_POR_DEFECTO = 0.5

# Modelo de repuesto mientras no exista el propio. Es uno de los pre-entrenados
# que trae openWakeWord; sirve para comprobar que el circuito entero funciona,
# pero la palabra que reconoce no es "Perseo". Ver commands/modelos/README.md.
MODELO_PUENTE = "hey_jarvis"

_DIRECTORIO = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO_PROPIO = os.path.join(_DIRECTORIO, "modelos", "perseo.onnx")


def ruta_modelo():
    """Devuelve qué modelo hay que cargar, y si es el propio o el de repuesto.

    Returns:
        tuple[str, bool]: la ruta (o el nombre corto de un pre-entrenado) y si es
        el modelo propio de Perseo.
    """
    configurado = os.environ.get("PERSEO_MODELO_PALABRA", "").strip()
    if configurado:
        return configurado, True
    if os.path.exists(RUTA_MODELO_PROPIO):
        return RUTA_MODELO_PROPIO, True
    return MODELO_PUENTE, False


def umbral():
    """Puntuación a partir de la cual se da la palabra por dicha."""
    crudo = os.environ.get("PERSEO_UMBRAL_PALABRA", "").strip()
    if not crudo:
        return UMBRAL_POR_DEFECTO
    try:
        return float(crudo)
    except ValueError:
        print(f"[-] PERSEO_UMBRAL_PALABRA no es un número ('{crudo}'), "
              f"se usa {UMBRAL_POR_DEFECTO}")
        return UMBRAL_POR_DEFECTO


def remuestrear(muestras, frecuencia_origen, frecuencia_destino=FRECUENCIA_MODELO):
    """Pasa un bloque de audio a la frecuencia que espera el modelo.

    El micrófono se lee a 44.100 Hz porque es lo más compatible en Windows y
    porque el umbral del aplauso está calibrado a esa frecuencia; tocar el
    `samplerate` del stream cambiaría el tamaño de bloque y, con él, la norma que
    decide si un ruido es un aplauso. Así que se remuestrea aquí y el detector de
    aplausos se queda como está.

    `resample_poly` filtra antes de decimar, que es justo lo que hace falta:
    interpolar a pelo de 44.1 kHz a 16 kHz mete aliasing en la banda de la voz.

    Args:
        muestras: audio mono, float en [-1, 1] o entero.
        frecuencia_origen (int): frecuencia a la que se grabó.
        frecuencia_destino (int): frecuencia de salida.

    Returns:
        np.ndarray: audio int16 a la frecuencia de destino.
    """
    from math import gcd

    from scipy.signal import resample_poly

    muestras = np.asarray(muestras, dtype=np.float32).reshape(-1)

    if frecuencia_origen != frecuencia_destino:
        divisor = gcd(int(frecuencia_origen), int(frecuencia_destino))
        subida = int(frecuencia_destino) // divisor
        bajada = int(frecuencia_origen) // divisor
        muestras = resample_poly(muestras, subida, bajada).astype(np.float32)

    # openWakeWord quiere PCM de 16 bits. El recorte importa: el remuestreo puede
    # sacar picos algo por encima de 1.0 y, sin recortar, int16 da la vuelta y
    # convierte un pico en un chasquido que el modelo no sabe leer.
    muestras = np.clip(muestras, -1.0, 1.0)
    return (muestras * 32767).astype(np.int16)


class DetectorPalabra:
    """Envuelve el modelo de openWakeWord y le da una respuesta de sí o no.

    Se construye una vez y se reutiliza: cargar el modelo cuesta cerca de un
    segundo, y ese segundo no puede caer entre el aplauso y la respuesta.
    """

    def __init__(self, ruta=None, puntuacion_minima=None):
        if ruta is None:
            ruta, self.es_propio = ruta_modelo()
        else:
            self.es_propio = True
        self.ruta = ruta
        self.puntuacion_minima = (
            puntuacion_minima if puntuacion_minima is not None else umbral()
        )
        self._modelo = None

    def cargar(self):
        """Carga el modelo. Vale llamarlo varias veces: solo la primera trabaja."""
        if self._modelo is not None:
            return self._modelo

        from openwakeword.model import Model

        # `onnx` explícito y no el automático: en Windows no hay `tflite-runtime`,
        # y dejar que openWakeWord elija termina en un ImportError que no dice
        # nada del problema real.
        self._modelo = Model(wakeword_models=[self.ruta], inference_framework="onnx")
        return self._modelo

    def escuchar(self, muestras, frecuencia_origen):
        """Decide si en este trozo de audio se ha dicho la palabra clave.

        Args:
            muestras: el audio grabado tras el doble aplauso.
            frecuencia_origen (int): a qué frecuencia se grabó.

        Returns:
            tuple[bool, float]: si se dijo la palabra, y con qué puntuación.
        """
        audio = remuestrear(muestras, frecuencia_origen)
        modelo = self.cargar()

        # Sin este `reset` la cola de audio de la activación anterior sigue dentro
        # del extractor: la segunda vez que aplaudes, el modelo estaría puntuando
        # una mezcla de lo que dices ahora y de lo que dijiste antes.
        modelo.reset()

        fotogramas = modelo.predict_clip(audio)
        if not fotogramas:
            return False, 0.0

        # El máximo sobre todas las salidas, no sobre una clave concreta: la clave
        # es el nombre del fichero del modelo, así que cambiar de perseo.onnx al
        # puente rompería cualquier nombre escrito a mano aquí.
        puntuacion = max(max(f.values()) for f in fotogramas)
        return puntuacion >= self.puntuacion_minima, float(puntuacion)

    def descripcion(self):
        """Una línea para el registro, para saber qué modelo está cargado."""
        if self.es_propio:
            return f"modelo propio '{os.path.basename(self.ruta)}'"
        return (f"modelo de repuesto '{self.ruta}' — NO reconoce \"Perseo\", "
                f"ver commands/modelos/README.md")
