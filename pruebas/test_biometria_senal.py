"""Qué parte de una ventana de micrófono es voz.

El detector de antes vivía en la app y recalculaba el ruido de fondo mientras
hablabas: a los 300 ms de empezar una frase ya no dejaba pasar casi nada. Estas
pruebas fijan lo contrario — que la voz seguida pasa entera, que las pausas se
quitan y que lo que no es voz (silencio, ruido, un zumbido plano) no llega.
"""

from __future__ import annotations

import math
import random

from perseo_core.servicios import biometria_senal as senal

FRECUENCIA = senal.FRECUENCIA


def _voz(segundos: float, amplitud: float = 0.25) -> list[float]:
    """Tono con envolvente de sílabas: sube y baja tres veces por segundo."""
    return [
        amplitud * (0.2 + 0.8 * abs(math.sin(2 * math.pi * 3 * i / FRECUENCIA))) * math.sin(i * 0.06)
        for i in range(int(FRECUENCIA * segundos))
    ]


def _silencio(segundos: float) -> list[float]:
    return [0.0] * int(FRECUENCIA * segundos)


def test_la_voz_seguida_pasa_entera() -> None:
    """El fallo de antes: a los 300 ms el suelo alcanzaba a la voz y la cortaba."""
    for segundos in (1.0, 2.0, 3.0):
        voz = senal.recortar_voz(_voz(segundos))
        assert senal.segundos(voz) >= segundos * 0.95


def test_la_voz_baja_tambien_pasa() -> None:
    assert senal.segundos(senal.recortar_voz(_voz(2.0, amplitud=0.05))) >= 1.9


def test_las_pausas_entre_frases_se_quitan() -> None:
    ventana = _voz(0.8) + _silencio(0.6) + _voz(0.8) + _silencio(0.6)
    voz = senal.segundos(senal.recortar_voz(ventana))
    # 1,6 s de voz y como mucho el colgado detrás de cada frase.
    assert 1.6 <= voz <= 1.6 + 2 * senal.TRAMAS_COLGADO * 0.02 + 0.01


def test_el_silencio_no_es_voz() -> None:
    assert senal.recortar_voz(_silencio(3.0)) == []


def test_el_ruido_de_fondo_no_es_voz() -> None:
    aleatorio = random.Random(7)
    ruido = [aleatorio.gauss(0, 0.02) for _ in range(FRECUENCIA * 3)]
    assert senal.recortar_voz(ruido) == []


def test_un_zumbido_plano_no_es_voz() -> None:
    """Un ventilador o un pitido no suben y bajan con las sílabas."""
    zumbido = [0.25 * math.sin(i * 0.06) for i in range(FRECUENCIA * 3)]
    assert senal.recortar_voz(zumbido) == []


def test_la_voz_sobre_ruido_se_separa_del_ruido() -> None:
    aleatorio = random.Random(3)
    ruido = lambda n: [aleatorio.gauss(0, 0.02) for _ in range(n)]  # noqa: E731
    ventana = [x + r for x, r in zip(_voz(0.8), ruido(12800))] + ruido(8000)
    ventana += [x + r for x, r in zip(_voz(0.8), ruido(12800))]
    voz = senal.segundos(senal.recortar_voz(ventana))
    assert 1.5 <= voz <= 1.9


def test_un_chasquido_suelto_se_tira() -> None:
    ventana = _silencio(1.0) + [0.5 * math.sin(i) for i in range(int(FRECUENCIA * 0.03))] + _silencio(1.0)
    assert senal.recortar_voz(ventana) == []


def test_pcm_de_la_app_se_lee_en_su_escala() -> None:
    import struct

    pcm = struct.pack("<3h", 0, 16384, -32768)
    assert senal.pcm16_a_flotantes(pcm) == [0.0, 0.5, -1.0]
    # Un byte suelto al final no revienta: se ignora.
    assert senal.pcm16_a_flotantes(pcm + b"\x01") == [0.0, 0.5, -1.0]
