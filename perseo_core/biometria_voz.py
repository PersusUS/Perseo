"""El motor de voz: ECAPA-TDNN por SpeechBrain.

Separado de `biometria.py` a propósito: este fichero es el único que importa
torch y speechbrain, que son cientos de megas. Si no están instalados, el
`ImportError` se queda aquí y el resto del núcleo sigue sin enterarse — igual
que `pc.py` vive sin pyautogui.

Por qué ECAPA-TDNN y no otra cosa: es el estado del arte abierto en verificación
de hablante (VoxCeleb), su modelo preentrenado baja solo de HuggingFace
(~80 MB), y produce un vector de 192 dimensiones pensado justo para esto:
comparar «¿es la misma voz?» por similitud coseno. Los embeddings de VoxCeleb
aguantan bien audio comprimido tipo llamada (Opus, banda telefónica), que es
exactamente lo que entra por un micrófono con cancelación de eco.

El primer uso tarda: importa torch y baja el modelo. Se avisa en el registro y
se cachea para siempre en `<datos>/modelos/speechbrain`.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

#: Dónde se cachea el modelo descargado, relativo a `<datos>/modelos/`.
CARPETA_CACHE = "speechbrain"

MODELO = "speechbrain/spkrec-ecapa-voxceleb"

_bloqueo = threading.Lock()


class MotorVozEcapa:
    """Convierte PCM 16 kHz mono en un vector de 192 números."""

    def __init__(self, directorio_modelos: Path) -> None:
        #: `<datos>/modelos`, que es donde SpeechBrain deja su caché.
        self._raiz = Path(directorio_modelos)
        self._codificador = None

    def _asegurar(self):
        if self._codificador is not None:
            return self._codificador
        with _bloqueo:
            if self._codificador is not None:
                return self._codificador
            # torch se importa aquí y no arriba: si falta, el ImportError sale
            # con este módulo ya cargado y biometria.py lo convierte en motivo.
            import torch  # noqa: F401 - se importa para que falle AQUÍ si no está
            from speechbrain.inference.speaker import EncoderClassifier

            logger.info("Cargando ECAPA-TDNN (%s); la primera vez puede tardar.", MODELO)
            # En Windows los symlinks exigen privilegio de desarrollador y el
            # fetch de SpeechBrain revienta con WinError 1314 (pasó el
            # 2026-08-24). Ahí se COPIA el modelo; en Linux/macOS, enlace.
            import sys
            from speechbrain.utils.fetching import LocalStrategy

            estrategia = (
                LocalStrategy.COPY if sys.platform == "win32" else LocalStrategy.SYMLINK
            )
            self._codificador = EncoderClassifier.from_hparams(
                source=MODELO,
                savedir=str(self._raiz / CARPETA_CACHE),
                # CPU y no GPU a propósito: la GPU queda para Ollama, que es
                # quien de verdad la aprovecha. ECAPA en CPU tarda decenas de ms
                # por segundo de audio, sobrado para etiquetar hablantes.
                run_opts={"device": "cpu"},
                local_strategy=estrategia,
            )
            logger.info("ECAPA-TDNN cargado.")
            return self._codificador

    def incrustar(self, muestras: list[float]) -> list[float]:
        import torch

        codificador = self._asegurar()
        with torch.no_grad():
            senal = torch.tensor(muestras, dtype=torch.float32).unsqueeze(0)
            emb = codificador.encode_batch(senal)  # [1, 1, 192]
        return [float(x) for x in emb.squeeze().tolist()]
