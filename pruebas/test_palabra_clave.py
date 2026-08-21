"""La palabra clave: qué cuenta como «Perseo» y qué se hace cuando falla la red.

Nada de esto habla con Google ni carga un modelo: lo que se prueba aquí es la
decisión —qué texto despierta a Perseo y qué se contesta cuando el audio no se
entiende—, que es donde están los fallos que se notan usándolo.

La transcripción de verdad la comprueba `commands/verificar_palabra_clave.py`,
que sí sale a la red y necesita SAPI para poner la voz.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "commands"))

import palabra_clave  # noqa: E402


# --------------------------------------------------------------------------- #
# Qué cuenta como haber dicho la palabra
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "texto",
    [
        "Perseo",
        "perseo",
        "PERSEO",
        "Perséo",
        "oye Perseo, enciende la luz",
        "  perseo  ",
        "perseus",
        "perceo",
    ],
)
def test_estas_transcripciones_despiertan_a_perseo(texto: str) -> None:
    """Google devuelve el nombre de formas distintas según el día. Aceptar solo
    una hace que la palabra clave funcione unas veces sí y otras no, que es la
    peor manera de fallar: parece que el micrófono no oye."""
    assert palabra_clave.dijo_la_palabra(texto)


@pytest.mark.parametrize(
    "texto",
    ["apaga la luz del salón", "", "   ", "percusión", "por seso", "un perro"],
)
def test_estas_no(texto: str) -> None:
    assert not palabra_clave.dijo_la_palabra(texto)


def test_normalizar_quita_tildes_y_mayusculas() -> None:
    assert palabra_clave.normalizar("Perséo") == "perseo"
    assert palabra_clave.normalizar("  ÁÉÍÓÚ  ") == "aeiou"
    assert palabra_clave.normalizar(None) == ""


# --------------------------------------------------------------------------- #
# Qué motor se usa
# --------------------------------------------------------------------------- #


def test_por_defecto_manda_google(monkeypatch: pytest.MonkeyPatch) -> None:
    """Es el único que reconoce «Perseo». El local responde a «hey jarvis» hasta
    que exista perseo.onnx, y un asistente que atiende por otro nombre no está."""
    monkeypatch.delenv("PERSEO_PALABRA_MOTOR", raising=False)
    assert palabra_clave.motor() == "google"
    assert isinstance(palabra_clave.crear_detector(), palabra_clave.DetectorGoogle)


def test_el_entorno_puede_pedir_el_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERSEO_PALABRA_MOTOR", "local")
    assert isinstance(palabra_clave.crear_detector(), palabra_clave.DetectorPalabra)


def test_los_dos_motores_se_usan_igual() -> None:
    """`clap_detector.py` no sabe cuál tiene puesto: los llama igual."""
    for clase in (palabra_clave.DetectorGoogle, palabra_clave.DetectorPalabra):
        for metodo in ("cargar", "escuchar", "descripcion"):
            assert callable(getattr(clase, metodo)), f"{clase.__name__} sin {metodo}"


# --------------------------------------------------------------------------- #
# Lo que pasa cuando Google no colabora
# --------------------------------------------------------------------------- #


def _falso_speech_recognition(respuesta):
    """Un `speech_recognition` de mentira que contesta lo que se le diga.

    `respuesta` puede ser un texto o una excepción a lanzar; si es una lista, se
    va gastando una por intento.
    """
    modulo = types.ModuleType("speech_recognition")

    class UnknownValueError(Exception):
        pass

    class RequestError(Exception):
        pass

    class AudioData:
        def __init__(self, datos, frecuencia, ancho):
            self.datos, self.frecuencia, self.ancho = datos, frecuencia, ancho

    pendientes = list(respuesta) if isinstance(respuesta, list) else [respuesta]
    intentos = []

    class Recognizer:
        def recognize_google(self, datos, language=""):
            intentos.append(language)
            valor = pendientes.pop(0) if len(pendientes) > 1 else pendientes[0]
            if isinstance(valor, Exception):
                raise valor
            return valor

    modulo.UnknownValueError = UnknownValueError
    modulo.RequestError = RequestError
    modulo.AudioData = AudioData
    modulo.Recognizer = Recognizer
    modulo.intentos = intentos
    return modulo


def _audio():
    numpy = pytest.importorskip("numpy")
    return numpy.zeros(1600, dtype=numpy.float32)


def test_lo_que_se_entiende_se_guarda_para_poder_contarlo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """«No se dijo la palabra» no ayuda a nadie; «se oyó "apaga la luz"» explica
    la vez que no saltó."""
    falso = _falso_speech_recognition("apaga la luz")
    monkeypatch.setitem(sys.modules, "speech_recognition", falso)

    detector = palabra_clave.DetectorGoogle()
    dijo, puntuacion = detector.escuchar(_audio(), 44100)

    assert (dijo, puntuacion) == (False, 0.0)
    assert detector.ultimo_texto == "apaga la luz"


def test_si_se_dijo_la_palabra_despierta(monkeypatch: pytest.MonkeyPatch) -> None:
    falso = _falso_speech_recognition("Perseo")
    monkeypatch.setitem(sys.modules, "speech_recognition", falso)

    dijo, puntuacion = palabra_clave.DetectorGoogle().escuchar(_audio(), 44100)
    assert (dijo, puntuacion) == (True, 1.0)


def test_un_audio_ininteligible_no_es_un_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Es el caso más normal del mundo —ruido, alguien que se calló— y tratarlo
    como fallo llenaría la pantalla de avisos que no significan nada."""
    falso = _falso_speech_recognition(None)
    monkeypatch.setitem(sys.modules, "speech_recognition", falso)
    detector = palabra_clave.DetectorGoogle()

    def revienta(self, datos, language=""):
        raise falso.UnknownValueError()

    falso.Recognizer.recognize_google = revienta

    assert detector.escuchar(_audio(), 44100) == (False, 0.0)
    assert detector.ultimo_texto == ""


def test_un_corte_de_red_se_reintenta_una_vez(monkeypatch: pytest.MonkeyPatch) -> None:
    """Google corta la conexión de vez en cuando. Un reintento, no tres: cada uno
    son segundos con una persona esperando delante."""
    falso = _falso_speech_recognition(None)
    monkeypatch.setitem(sys.modules, "speech_recognition", falso)

    llamadas = {"n": 0}

    def a_la_segunda(self, datos, language=""):
        llamadas["n"] += 1
        if llamadas["n"] == 1:
            raise falso.RequestError("conexión cortada")
        return "Perseo"

    falso.Recognizer.recognize_google = a_la_segunda

    assert palabra_clave.DetectorGoogle().escuchar(_audio(), 44100) == (True, 1.0)
    assert llamadas["n"] == 2


def test_si_la_red_no_vuelve_se_dice(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin red no hay palabra clave, y eso hay que decirlo: tragárselo se ve
    desde fuera como «Perseo ha dejado de responder»."""
    falso = _falso_speech_recognition(None)
    monkeypatch.setitem(sys.modules, "speech_recognition", falso)

    def siempre_falla(self, datos, language=""):
        raise falso.RequestError("sin red")

    falso.Recognizer.recognize_google = siempre_falla

    with pytest.raises(falso.RequestError):
        palabra_clave.DetectorGoogle().escuchar(_audio(), 44100)
