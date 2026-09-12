"""El motor de caras: YuNet detecta, SFace incrusta. Ambos por OpenCV.

Separado de `biometria.py` por el mismo motivo que `biometria_voz.py`: aquí
vive el único `import cv2` del núcleo, y si falta opencv-python el resto no se
entera.

Por qué esta pareja y no InsightFace/ArcFace: los dos modelos son ficheros ONNX
pequeños (~0,2 MB y ~37 MB) que corren **dentro de cv2.dnn**, sin onnxruntime,
sin torch y sin GPU; y son lo que OpenCV mantiene en su zoo oficial. Para saber
«quién sale por la cámara de casa» sobra: LFW da a SFace una precisión del 99,6%
en verificación 1:1, y aquí siempre se verifica contra pocos perfiles.

Los modelos se bajan la primera vez que hace falta un vector —no al construir el
motor—, desde el repositorio `opencv/opencv_zoo` de GitHub, y quedan cacheados
en `<datos>/modelos/`.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

YUNET_FICHERO = "face_detection_yunet_2023mar.onnx"
SFACE_FICHERO = "face_recognition_sface_2021dec.onnx"

ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
URL_YUNET = f"{ZOO}/face_detection_yunet/{YUNET_FICHERO}"
URL_SFACE = f"{ZOO}/face_recognition_sface/{SFACE_FICHERO}"

#: Confianza mínima del detector para considerar cara. Con cámaras de escritorio,
#: 0.6 quita casi todos los falsos positivos sin perder caras reales.
PUNTUACION_MINIMA = 0.6


class MotorCaraOpencv:
    """Detecta caras en un JPEG y entrega su vector de 128 números cada una."""

    def __init__(self, directorio_modelos: Path) -> None:
        self._raiz = Path(directorio_modelos)
        # La comprobación de dependencia es barata y va aquí: si falta cv2, el
        # ImportError sale al construir el motor y biometria.py lo convierte en
        # motivo claro. Bajar modelos NO pasa aquí: es lento y solo toca hacerse
        # cuando alguien usa de verdad la cámara.
        import cv2  # noqa: F401

        self._detector = None
        self._reconocedor = None

    def _asegurar(self):
        import cv2

        if self._detector is not None and self._reconocedor is not None:
            return cv2
        self._raiz.mkdir(parents=True, exist_ok=True)
        ruta_yunet = self._raiz / YUNET_FICHERO
        ruta_sface = self._raiz / SFACE_FICHERO
        for ruta, url in ((ruta_yunet, URL_YUNET), (ruta_sface, URL_SFACE)):
            if not ruta.is_file() or ruta.stat().st_size == 0:
                logger.info("Bajando modelo de caras %s…", ruta.name)
                with urllib.request.urlopen(url, timeout=60) as respuesta, ruta.open("wb") as destino:
                    destino.write(respuesta.read())

        self._detector = cv2.FaceDetectorYN.create(
            str(ruta_yunet),
            "",
            # El tamaño de entrada se reajusta en cada imagen (`setInputSize`);
            # este valor inicial es irrelevante pero obligatorio.
            (320, 320),
            score_threshold=PUNTUACION_MINIMA,
        )
        self._reconocedor = cv2.FaceRecognizerSF.create(str(ruta_sface), "")
        return cv2

    def detectar(self, jpeg: bytes) -> list[dict]:
        """Una entrada por cara: caja en píxeles y vector de SFace."""
        import numpy as np

        cv2 = self._asegurar()
        matriz = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if matriz is None:
            return []
        alto, ancho = matriz.shape[:2]
        self._detector.setInputSize((ancho, alto))
        caras = self._detector.detect(matriz)[1]
        salida = []
        if caras is None:
            return salida
        for fila in caras:
            # Fila: x, y, w, h, 5 puntos de referencia, puntuación.
            caja = [int(round(v)) for v in fila[:4]]
            alineada = self._reconocedor.alignCrop(matriz, fila)
            vector = self._reconocedor.feature(alineada).flatten()
            salida.append(
                {
                    "caja": caja,
                    "vector": [float(x) for x in vector.tolist()],
                }
            )
        return salida
