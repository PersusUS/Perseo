"""La señal antes del vector: PCM a flotantes y qué parte de un trozo es voz.

**Por qué el detector de voz vive aquí y no en la app.** Hasta el 2026-09-23 lo
decidía `identidad.ts` con un suelo de ruido que se recalculaba *también
mientras hablabas*. Con trozos de 64 ms el suelo alcanzaba al propio volumen
de la voz en unos 300 ms, el umbral —tres veces el suelo— se quedaba por
encima de casi todo lo dicho, y a ECAPA solo le llegaban picos de sílaba
pegados unos con otros. Simulado sobre voz sintética, pasaba entre un 6 y un
18 % de la voz y ningún trozo llegaba a los 0,4 s que este núcleo exige: casi
todo lo que viajaba se tiraba al llegar. El perfil `Persus` sumaba ocho
muestras en un mes de llamadas.

Así que la app ahora solo **transporta** ventanas seguidas de micrófono, sin
recortar, y la decisión de qué es voz se toma aquí, que es donde se decide
todo lo demás. Las caras no piensan.

**Cómo se decide.** Energía por tramas de 20 ms, con dos umbrales:

  · el suelo se mide con el percentil 10 de la propia ventana, no con una
    media que corre detrás de la voz: en una ventana de dos o tres segundos
    siempre hay huecos entre palabras, y esos huecos son el ruido;
  · una ventana **plana** —el percentil 90 no llega al doble del 10— no es voz:
    la voz sube y baja con cada sílaba, y lo que no sube ni baja es un
    ventilador, un zumbido o un pitido. Sin esta regla, en una habitación
    ruidosa y sin supresión de ruido el ventilador pasaría por voz;
  · los umbrales se acotan también por arriba con el percentil 90: en una
    ventana de habla seguida, sin un solo silencio, el percentil 10 ya es voz,
    y tres veces eso dejaría fuera casi todo;
  · para *empezar* a contar voz hay que pasar el umbral alto, y para *seguir*
    basta con el bajo (histéresis). Así las colas de las sílabas, que tienen
    poca energía y mucha identidad, no se cortan;
  · un colgado de 120 ms cose las pausas cortas dentro de una palabra, y los
    chasquidos de menos de 60 ms se tiran.

Sin numpy a propósito, como el resto de la biometría: este fichero corre en
cualquier Python 3.11 pelado y las pruebas no necesitan nada instalado.
"""

from __future__ import annotations

import math
import struct

FRECUENCIA = 16000

#: Tramas de 20 ms: lo bastante cortas para no tragarse una pausa entre
#: palabras y lo bastante largas para que su energía signifique algo.
MUESTRAS_TRAMA = FRECUENCIA // 50

#: Mínimos absolutos, en escala [-1, 1]. Con la supresión de ruido del
#: navegador el silencio queda casi en cero, y el percentil 10 también: sin un
#: mínimo, cualquier roce del micrófono pasaría el umbral. 0,012 son unos 390
#: en int16, un poco por debajo de los 500 que la app usaba como suelo fijo.
UMBRAL_ALTO_ABSOLUTO = 0.012
UMBRAL_BAJO_ABSOLUTO = 0.006

#: Cuántas veces por encima del ruido de la ventana tiene que estar una trama.
FACTOR_ALTO = 3.0
FACTOR_BAJO = 1.8

#: Y como mucho, qué fracción de lo más fuerte de la ventana (percentil 90).
TECHO_ALTO = 0.5
TECHO_BAJO = 0.3

#: Por debajo de este cociente entre percentiles 90 y 10, la ventana es plana.
RANGO_MINIMO = 2.0

#: Tramas de colgado tras la última con voz (6 × 20 ms = 120 ms).
TRAMAS_COLGADO = 6

#: Tramas seguidas mínimas para que un tramo cuente (3 × 20 ms = 60 ms).
TRAMAS_MINIMAS = 3


def pcm16_a_flotantes(pcm: bytes) -> list[float]:
    """PCM little-endian int16 mono a flotantes en [-1, 1]. Lo que manda la app."""
    utiles = pcm[: (len(pcm) // 2) * 2]
    enteros = struct.unpack(f"<{len(utiles) // 2}h", utiles)
    return [e / 32768.0 for e in enteros]


def _energias(muestras: list[float]) -> list[float]:
    """RMS de cada trama completa. La última, si se queda corta, no cuenta."""
    energias = []
    for inicio in range(0, len(muestras) - MUESTRAS_TRAMA + 1, MUESTRAS_TRAMA):
        trama = muestras[inicio : inicio + MUESTRAS_TRAMA]
        energias.append(math.sqrt(sum(x * x for x in trama) / MUESTRAS_TRAMA))
    return energias


def _percentil(valores: list[float], fraccion: float) -> float:
    ordenados = sorted(valores)
    return ordenados[min(len(ordenados) - 1, int(len(ordenados) * fraccion))]


def tramas_con_voz(muestras: list[float]) -> list[bool]:
    """Una marca por trama de 20 ms: si ahí hay voz o no."""
    energias = _energias(muestras)
    if not energias:
        return []
    suelo = _percentil(energias, 0.10)
    fuerte = _percentil(energias, 0.90)
    if fuerte < suelo * RANGO_MINIMO:
        return [False] * len(energias)
    alto = max(UMBRAL_ALTO_ABSOLUTO, min(suelo * FACTOR_ALTO, fuerte * TECHO_ALTO))
    bajo = max(UMBRAL_BAJO_ABSOLUTO, min(suelo * FACTOR_BAJO, fuerte * TECHO_BAJO))

    # Histéresis: se entra por el alto y se sigue mientras no se baje del bajo.
    marcas = [False] * len(energias)
    dentro = False
    for i, energia in enumerate(energias):
        dentro = energia >= alto if not dentro else energia >= bajo
        marcas[i] = dentro

    # Fuera los chasquidos ANTES del colgado: si no, el colgado estira un clic
    # de 20 ms hasta los 140 y ya parece una sílaba.
    limpias = list(marcas)
    i = 0
    while i < len(marcas):
        if not marcas[i]:
            i += 1
            continue
        fin = i
        while fin < len(marcas) and marcas[fin]:
            fin += 1
        if fin - i < TRAMAS_MINIMAS:
            for j in range(i, fin):
                limpias[j] = False
        i = fin

    # Colgado: la pausa corta dentro de una palabra sigue siendo esa palabra.
    cosidas = list(limpias)
    ultima = -TRAMAS_COLGADO - 1
    for i, marca in enumerate(limpias):
        if marca:
            ultima = i
        elif i - ultima <= TRAMAS_COLGADO:
            cosidas[i] = True
    return cosidas


def recortar_voz(muestras: list[float]) -> list[float]:
    """Solo lo que es voz, en el orden en que se dijo.

    Los tramos se juntan: dentro de una ventana de dos o tres segundos habla
    una sola persona casi siempre, y lo que se quita son silencios que a ECAPA
    solo le restan. Si en la ventana no hay voz, lista vacía.
    """
    marcas = tramas_con_voz(muestras)
    voz: list[float] = []
    for i, marca in enumerate(marcas):
        if marca:
            voz.extend(muestras[i * MUESTRAS_TRAMA : (i + 1) * MUESTRAS_TRAMA])
    return voz


def segundos(muestras: list[float]) -> float:
    return len(muestras) / FRECUENCIA
