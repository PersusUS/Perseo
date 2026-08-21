"""Reconocimiento de la palabra clave. Dos motores, y manda el de Google.

Hay dos formas de decidir si en los tres segundos que siguen al doble aplauso se
ha dicho «Perseo», y las dos están aquí:

* **Google Speech** (`DetectorGoogle`, **el de por defecto**). Se sube el audio,
  se transcribe a texto y se busca la palabra dentro. Es lo que hacía Perseo v1.
* **openWakeWord** (`DetectorPalabra`). Decide en local, sobre la CPU, en unos
  180 ms y sin que salga nada del portátil.

El local llegó el 2026-08-15 y era mejor en todo menos en una cosa, que resultó
ser la que importa: **no reconoce «Perseo»**. Necesita un modelo entrenado para
esa palabra y entrenarlo pide una GPU prestada, así que mientras tanto respondía
a «hey jarvis». Un asistente que se llama Perseo y atiende por otro nombre no es
un asistente a medias: es un asistente que no está.

**Decidido el 2026-08-21: se vuelve a Google** y se deja el modelo local escrito
para el día que exista `perseo.onnx`. Lo que se paga a cambio, dicho claro:

* entre uno y tres segundos de ida y vuelta, en vez de 180 ms;
* **tres segundos de audio del salón viajan a un tercero** cada vez que aplaudes
  dos veces;
* sin red no hay palabra clave.

Y lo que **no** cambia, que es lo que hace que ese precio sea asumible: **el
doble aplauso sigue mandando**. No hay micrófono subiendo nada mientras tanto;
solo los tres segundos que siguen a las dos palmadas. Este módulo sustituye al
aplauso en nada.

Configuración por entorno:

    PERSEO_PALABRA_MOTOR    'google' (por defecto) o 'local'.
    PERSEO_MODELO_PALABRA   Solo para el motor local: ruta a un .onnx propio.
                            Por defecto se busca commands/modelos/perseo.onnx.
    PERSEO_UMBRAL_PALABRA   Solo para el motor local: puntuación mínima. Por
                            defecto 0.5, que es lo que recomienda openWakeWord.
"""

import os
import unicodedata

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

    import numpy as np
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


# --------------------------------------------------------------------------- #
# El motor de Google: se transcribe y se busca la palabra en el texto
# --------------------------------------------------------------------------- #

#: Cómo puede salir «Perseo» de un transcriptor que no conoce el nombre. No es
#: una lista de sinónimos: son las cosas que Google ha devuelto de verdad al oír
#: la palabra. Se aceptan porque quien está delante ha aplaudido dos veces y ha
#: dicho algo — el aplauso ya filtró el ruido, y aquí ser estricto solo consigue
#: que haya que repetirse.
VARIANTES = ("perseo", "perseus", "perceo", "perseio", "perse o", "persero")

IDIOMA = "es-ES"

#: Cuántas veces se le pide a Google. Dos, no tres: cada intento son segundos con
#: una persona esperando delante, y si el primero falla por red el segundo suele
#: fallar igual.
INTENTOS = 2


def normalizar(texto):
    """Minúsculas y sin tildes, para comparar sin sorpresas.

    Google devuelve «Perseo» o «perseo» según el día, y a veces con tilde por
    medio. Comparar en crudo hace que la palabra clave funcione unas veces sí y
    otras no, que es la peor forma de fallar.
    """
    sin_tildes = unicodedata.normalize("NFD", str(texto or ""))
    sin_tildes = "".join(c for c in sin_tildes if unicodedata.category(c) != "Mn")
    return sin_tildes.lower().strip()


def dijo_la_palabra(texto):
    """Si en una transcripción está la palabra clave, en cualquiera de sus formas."""
    limpio = normalizar(texto)
    return any(variante in limpio for variante in VARIANTES)


class DetectorGoogle:
    """Sube los tres segundos a Google Speech y busca «Perseo» en el texto.

    Es lo que hacía Perseo v1 y a lo que se vuelve el 2026-08-21: reconoce la
    palabra de verdad, que es lo único que el modelo local no sabe hacer.

    No hay nada que cargar, así que `cargar()` está para que las dos clases se
    puedan usar igual desde `clap_detector.py`.
    """

    #: Para que quien lo use pueda decir lo mismo que con el otro motor.
    puntuacion_minima = 1.0

    def __init__(self, idioma=IDIOMA, intentos=INTENTOS):
        self.idioma = idioma
        self.intentos = intentos
        self.es_propio = True
        #: Lo último que se entendió. Es lo que se enseña cuando no despierta:
        #: «no se dijo la palabra» no ayuda; «oí "apaga la luz"» sí.
        self.ultimo_texto = ""

    def cargar(self):
        """Comprueba que la librería está, y nada más. Aquí no hay modelo."""
        import speech_recognition  # noqa: F401

        return None

    def escuchar(self, muestras, frecuencia_origen):
        """Transcribe y decide. Devuelve `(dijo_la_palabra, puntuación)`.

        La puntuación es 1 o 0: con un transcriptor no hay grados, o la palabra
        está en el texto o no está. Se devuelve igual que en el motor local para
        que quien llame no tenga que saber cuál está puesto.

        Un audio que no se entiende **no es un error**: es el caso más normal del
        mundo —ruido, alguien que se calló— y se contesta que no se dijo la
        palabra. Los fallos de red sí se lanzan, porque son otra cosa y quien
        llama los cuenta aparte.
        """
        import numpy as np
        import speech_recognition as sr

        audio = np.asarray(muestras, dtype=np.float32).reshape(-1)
        audio = np.clip(audio, -1.0, 1.0)
        # 16 bits, que es lo que declara `sample_width=2` justo debajo. Sin el
        # recorte de arriba, un pico por encima de 1.0 da la vuelta en int16 y se
        # convierte en un chasquido.
        entero = (audio * 32767).astype(np.int16)

        datos = sr.AudioData(entero.tobytes(), int(frecuencia_origen), 2)
        reconocedor = sr.Recognizer()

        ultimo_fallo = None
        for intento in range(self.intentos):
            try:
                texto = reconocedor.recognize_google(datos, language=self.idioma)
            except sr.UnknownValueError:
                self.ultimo_texto = ""
                return False, 0.0
            except sr.RequestError as e:
                # Google corta la conexión de vez en cuando (`WinError 10054`).
                # Se reintenta una vez y, si vuelve a fallar, que lo cuente quien
                # llama: sin red no hay palabra clave, y eso hay que decirlo.
                ultimo_fallo = e
                if intento + 1 >= self.intentos:
                    raise
                continue

            self.ultimo_texto = texto
            acerto = dijo_la_palabra(texto)
            return acerto, 1.0 if acerto else 0.0

        raise ultimo_fallo  # pragma: no cover — el bucle sale antes

    def descripcion(self):
        return f"Google Speech en {self.idioma} — reconoce «Perseo», y necesita red"


#: Qué motor usa `clap_detector.py`. El de por defecto es el de Google porque es
#: el único que reconoce la palabra: ver la cabecera de este módulo.
MOTOR_POR_DEFECTO = "google"


def motor():
    """Qué motor toca, según `PERSEO_PALABRA_MOTOR`."""
    elegido = os.environ.get("PERSEO_PALABRA_MOTOR", "").strip().lower()
    return elegido or MOTOR_POR_DEFECTO


def crear_detector():
    """El detector que toca. Los dos tienen `cargar`, `escuchar` y `descripcion`."""
    if motor() == "local":
        return DetectorPalabra()
    return DetectorGoogle()
