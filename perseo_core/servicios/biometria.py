"""Quién habla y quién sale por la cámara: perfiles biométricos locales.

La idea que esto cumple llega del señor Persus: en una llamada, Perseo debería
saber **quién** está delante —ponerle nombre a quien habla, etiquetar la cara que
ve— y, si no conoce a alguien, **preguntarle cómo se llama y aprenderlo
entonces**: la próxima vez le reconoce sin que nadie entrene nada.

**Dónde vive la inteligencia.** En el núcleo, como manda la regla del plan: las
caras no piensan. La app de voz solo transporta trozos —ventanas seguidas de
micrófono y JPEG de la cámara, los mismos que ya le manda a Gemini— y pinta la
etiqueta que le devuelve esta pieza. Aquí dentro ocurre todo lo demás: decidir
qué parte del trozo es voz (`biometria_senal.py`), extraer el vector (ECAPA-TDNN
o YuNet + SFace), compararlo contra las galerías guardadas
(`biometria_galeria.py`), y decidir si es alguien conocido, alguien que se está
aprendiendo o ruido.

**Los modelos son opcionales, como Ollama.** `torch`+`speechbrain` pesan cientos
de megas y `opencv-python` otros tantos: obligar a todo el mundo a instalarlos
para arrancar el núcleo sería romper la Raspberry Pi del plan por un capricho.
Sin ellos el núcleo arranca igual y las rutas de biometría contestan con un
motivo claro (`disponibilidad()`), igual que `estado.telemetria()` devuelve
`disponible: false` sin psutil. Lo que hace falta va anotado en
`requirements-biometria.txt`.

**Privacidad, que aquí no es decorado.** Lo que se guarda son números, nunca
audio ni imágenes: por cada persona, unos pocos vectores de 192 números (voz) y
de 128 (cara). Viven en `<datos>/perfiles.json`, que ya está fuera de git porque
todo `perseo_core/datos/` lo está. Nada sale de la máquina: ni los vectores ni
las muestras viajan a ningún servicio; el único tercero que ve algo es Gemini,
que ya veía el mismo micrófono y la misma cámara antes de que existiera este
módulo. Y borrar un perfil borra sus números de verdad: no hay copia en ninguna
otra parte.

**Cómo se decide quién habla (desde el 2026-09-23).** Un trozo solo no basta:
dos segundos de voz dan un vector con ruido, y decidir trozo a trozo hacía que
la etiqueta bailara. Así que los trozos seguidos forman un **turno**, y lo que
se compara es la media del turno, ponderada por los segundos de voz de cada
trozo. El turno se corta con una pausa larga o cuando el trozo nuevo no se
parece al que lleva el turno —que es lo que pasa cuando habla otro—. Con esa
media hay tres zonas:

  · por encima de `UMBRAL_VOZ`, es esa persona;
  · entre `UMBRAL_GRIS` y `UMBRAL_VOZ`, **duda**: no se nombra a nadie. Hasta
    el 2026-09-23 la cámara resolvía la duda —la voz dudaba, se veía al dueño
    solo delante, era él—, y así se llamaba «señor Persus» a la visita que
    hablaba fuera de plano: la cámara dice quién está, no quién habla;
  · por debajo de `UMBRAL_GRIS`, es otra persona.

Y aunque la media pase el umbral, **si otra persona conocida queda a menos de
`MARGEN_VOZ`**, tampoco se nombra: dos voces parecidas —un padre y un hijo—
son justo el caso en que acertar a medias es equivocarse del todo.

**Nadie se aprende sin nombre (desde el 2026-09-23).** Antes, un desconocido
que hablaba doce segundos se guardaba solo como «Desconocido N». Ahora sus
trozos engordan un *racimo* de la sesión, en memoria, que le da una etiqueta
provisional para que la llamada sepa que no es el dueño y le pregunte cómo se
llama; y **solo cuando alguien dice su nombre** (`renombrar`, o el alta desde
Ajustes) se guarda en disco. Si nadie lo dice, al reiniciar se olvida. Con las
caras igual. Cada cosa que se guarda deja una línea en `personas.log`.

**Lo que sí se aprende solo: mejorar a quien ya tiene nombre.** Cada turno en
que alguien conocido habla claro y largo, y sin otra persona cerca en la
puntuación, su muestra entra en la galería si aporta algo. La galería crece
hasta `MAX_VOCES`, y con ella el reconocimiento de esa persona en micrófonos,
días y catarros distintos.

Y una regla que no cambia porque haya caras de por medio: lo observado no
es instrucción, y un rostro conocido no autoriza nada.
"""

from __future__ import annotations

import base64
import logging
import threading
from pathlib import Path
from typing import Any, Protocol

from . import biometria_galeria as galeria
from . import biometria_perfiles as disco
from . import biometria_senal as senal
from . import biometria_sesion as sesion

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constantes de decisión
# --------------------------------------------------------------------------- #

#: Fichero de perfiles dentro de `<datos>/`. Fuera de git como todo `datos/`.
NOMBRE_FICHERO = disco.NOMBRE_FICHERO

#: Carpeta de modelos descargados, dentro de `<datos>/`.
CARPETA_MODELOS = "modelos"

#: Media de turno mínima para decir «esta voz es la de ese perfil». Con
#: ECAPA-TDNN el mismo hablante suele rondar 0.65-0.90 y otra persona 0.20-0.45;
#: el umbral se queda en medio, conservador a propósito: mejor un «Desconocido»
#: de más que un nombre equivocado puesto con seguridad.
UMBRAL_VOZ = 0.55

#: Por debajo de esto el que habla es otro, y es donde se corta el turno. Entre
#: este y `UMBRAL_VOZ` está la duda, y con duda no se nombra a nadie.
UMBRAL_GRIS = 0.40

#: Para que un acierto entre en la galería no basta con acertar: tiene que ser
#: un acierto claro y largo. Es lo que impide que una voz parecida, pasando por
#: los pelos, se vaya haciendo sitio en el perfil de otro.
UMBRAL_REFUERZO_VOZ = 0.65
SEGUNDOS_REFUERZO = 1.5

#: Voz útil mínima de un trozo —después de quitarle los silencios— para
#: molestar a ECAPA. Por debajo de un segundo el vector sale tan inestable que
#: estorba más de lo que ayuda a la media del turno.
SEGUNDOS_VOZ_MINIMOS = 1.0

#: Voz útil mínima de una muestra de alta desde Ajustes, que se graba a
#: propósito y puede exigir más.
SEGUNDOS_ENROLAR = 2.0

#: Distancia mínima entre la primera y la segunda persona para nombrar a la
#: primera. Por debajo, las dos voces se parecen demasiado para jugársela.
MARGEN_VOZ = 0.08

#: Cuántas muestras de voz y de cara llega a guardar una persona con nombre.
#: Ocho se quedaban cortas: el reconocimiento tiene que ir mejorando con el
#: uso, y eso es seguir guardando las muestras buenas que aporten algo nuevo.
MAX_VOCES = 40
MAX_CARAS = 16

#: Umbral coseno de SFace para «es la misma cara». Es el que recomienda OpenCV
#: en su zoo de modelos (0.363), medido sobre LFW; no se afina a mano.
UMBRAL_CARA = 0.363

#: Margen por encima del umbral para que una cara reconocida entre en la
#: galería. Mismo motivo que `UMBRAL_REFUERZO_VOZ`.
MARGEN_REFUERZO_CARA = 0.10

#: Lo mismo que `MARGEN_VOZ`, para las caras: un parecido de familia no es
#: la misma persona.
MARGEN_CARA = 0.05

#: Voz y fotogramas oídos de un desconocido que se recomiendan antes de darle
#: nombre. No fija nada solo: es lo que la pantalla enseña como progreso.
SEGUNDOS_VOZ_APRENDER = 12.0
FRAMES_CARA_APRENDER = 8

#: Calidad mínima de una cara para aprender de ella o reforzar con ella. Para
#: reconocer vale cualquiera; para enseñar, no: un fotograma movido o de perfil
#: guardado en la galería se queda ahí estorbando. Los valores de nitidez y
#: giro no están medidos con esta cámara: son conservadores, y el día que haya
#: un evaluador con caras reales se afinan con él.
CARA_MINIMA_PX = 40
PUNTUACION_UTIL = 0.80
NITIDEZ_MINIMA = 35.0
GIRO_MAXIMO = 0.6


# --------------------------------------------------------------------------- #
# Motores (opcionales e intercambiables)
#
# El módulo entero funciona sin torch ni cv2: los motores se construyen perezosos,
# se cachean, y si falta la dependencia `disponibilidad()` dice qué falta. Las
# pruebas meten motores falsos con `pon_motores()`.
# --------------------------------------------------------------------------- #


class MotorVoz(Protocol):
    """Lo que se le pide a un extractor de vectores de voz."""

    def incrustar(self, muestras: list[float]) -> list[float]: ...


class MotorCara(Protocol):
    """Lo que se le pide a un detector+reconocedor de caras (ver `biometria_cara.py`)."""

    def detectar(self, jpeg: bytes) -> list[dict[str, Any]]:
        """Una entrada por cara: {"caja": [x, y, w, h], "vector": [floats]}.

        Opcionales, para la calidad: "puntuacion", "nitidez" y "giro". Si no
        vienen, la cara se da por buena.
        """
        ...


_motor_voz_cacheado: MotorVoz | None = None
_motor_cara_cacheado: MotorCara | None = None
_motivo_voz: str | None = None
_motivo_cara: str | None = None
_bloqueo_motores = threading.Lock()


def pon_motores(voz: MotorVoz | None = None, cara: MotorCara | None = None) -> None:
    """Sustituye los motores. Para las pruebas; en producción no se llama."""
    global _motor_voz_cacheado, _motor_cara_cacheado, _motivo_voz, _motivo_cara
    with _bloqueo_motores:
        _motor_voz_cacheado = voz
        _motor_cara_cacheado = cara


def obtener_motor_voz(directorio_datos: Path) -> MotorVoz | None:
    """El motor de voz, construyéndolo la primera vez si se puede."""
    global _motor_voz_cacheado, _motivo_voz
    with _bloqueo_motores:
        if _motor_voz_cacheado is not None or _motivo_voz is not None:
            return _motor_voz_cacheado
    try:
        from .biometria_voz import MotorVozEcapa  # perezoso: toca torch

        motor = MotorVozEcapa(Path(directorio_datos) / CARPETA_MODELOS)
    except Exception as e:  # ImportError, o cualquier cosa al importar torch
        with _bloqueo_motores:
            _motivo_voz = f"Voz desactivada ({type(e).__name__}: {e}). Instala requirements-biometria.txt"
            return None
    with _bloqueo_motores:
        _motor_voz_cacheado = motor
        return motor


def obtener_motor_cara(directorio_datos: Path) -> MotorCara | None:
    """El motor de caras. Construirlo NO baja modelos: eso pasa al primer uso."""
    global _motor_cara_cacheado, _motivo_cara
    with _bloqueo_motores:
        if _motor_cara_cacheado is not None or _motivo_cara is not None:
            return _motor_cara_cacheado
    try:
        from .biometria_cara import MotorCaraOpencv  # perezoso: toca cv2

        motor = MotorCaraOpencv(Path(directorio_datos) / CARPETA_MODELOS)
    except Exception as e:
        with _bloqueo_motores:
            _motivo_cara = f"Cara desactivada ({type(e).__name__}: {e}). Instala opencv-python"
            return None
    with _bloqueo_motores:
        _motor_cara_cacheado = motor
        return motor


def disponibilidad(directorio_datos: Path | None = None) -> dict[str, Any]:
    """Qué se puede hacer hoy y qué falta. Nunca lanza: es para pintar estado.

    Construir los motores comprueba las dependencias pero **no** baja modelos:
    eso queda para el primer uso real (`biometria_voz` y `biometria_cara` lo
    hacen perezoso). Abrir Ajustes no debe costar una descarga de 80 MB.
    """
    raiz = Path(directorio_datos or Path("."))
    voz = obtener_motor_voz(raiz)
    cara = obtener_motor_cara(raiz)
    return {
        "voz": voz is not None,
        "motivo_voz": None if voz is not None else (_motivo_voz or "Sin probar"),
        "cara": cara is not None,
        "motivo_cara": None if cara is not None else (_motivo_cara or "Sin probar"),
        "umbrales": {
            "voz": UMBRAL_VOZ,
            "gris": UMBRAL_GRIS,
            "cara": UMBRAL_CARA,
            "segundos_aprender": SEGUNDOS_VOZ_APRENDER,
        },
    }


def _con_red_de_seguridad(operacion):
    """Ejecuta una llamada al motor convirtiendo cualquier reviento en error.

    Los motores cargan modelos de red la primera vez, tocan torch y cv2, y
    pueden fallar por mil motivos que no son culpa de quien habla (sin red,
    NumPy desalineado, GPU ocupada). Ninguno de esos fallos debe salir de aquí
    como excepción: una ruta del núcleo que devuelve 500 a cada trozo de voz es
    peor que un reconocimiento apagado con su motivo.
    """
    try:
        return operacion()
    except Exception as e:  # noqa: BLE001 — exactamente lo que se busca
        logger.warning("El motor biométrico falló: %s: %s", type(e).__name__, e)
        return {"error": f"El motor falló ({type(e).__name__}). Mira el registro del núcleo."}


# --------------------------------------------------------------------------- #
# Perfiles en disco y su registro: `biometria_perfiles.py`
# --------------------------------------------------------------------------- #

cargar = disco.cargar
guardar = disco.guardar
ruta_perfiles = disco.ruta_perfiles
resumen = disco.resumen


# --------------------------------------------------------------------------- #
# La sesión vive en `biometria_sesion.py`: racimos, turno y lo que se ve.
# --------------------------------------------------------------------------- #


def reiniciar_sesion() -> None:
    """Vacía lo que vive en memoria. Los perfiles fijados no se tocan."""
    with sesion.bloqueo:
        sesion.reiniciar()


# --------------------------------------------------------------------------- #
# Identificar voz
# --------------------------------------------------------------------------- #


def identificar_voz(directorio_datos: Path, audio_base64: str) -> dict[str, Any]:
    """¿De quién es esta ventana de micrófono?

    Devuelve una de estas cosas:

      · `{"nombre": ..., "confianza": ...}` si es alguien con nombre;
      · `{"nombre": "Desconocido N", "provisional": true}` si no es nadie
        conocido: la etiqueta es de la sesión y no se guarda en ninguna parte
        hasta que alguien diga su nombre;
      · `{"nombre": None, "dudoso": ...}` si se parece a alguien pero no lo
        bastante, o a dos personas a la vez;
      · `{"nombre": None, "voz": segundos}` si en la ventana casi no había voz;
      · `{"error": motivo}` con el motor ausente: la llamada decide cómo decirlo.
    """
    motor = obtener_motor_voz(directorio_datos)
    if motor is None:
        return {"error": _motivo_voz or "Motor de voz no disponible"}

    try:
        pcm = base64.b64decode(audio_base64)
    except (ValueError, TypeError):
        return {"error": "Audio ilegible"}
    voz = senal.recortar_voz(senal.pcm16_a_flotantes(pcm))
    duracion = senal.segundos(voz)
    if duracion < SEGUNDOS_VOZ_MINIMOS:
        return {"nombre": None, "voz": round(duracion, 2)}

    vector = _con_red_de_seguridad(lambda: motor.incrustar(voz))
    if isinstance(vector, dict):
        return vector

    with sesion.bloqueo:
        perfiles = cargar(directorio_datos)
        ahora = sesion.reloj()
        notas = {
            nombre: galeria.puntuar(galeria.voces(perfil), vector)
            for nombre, perfil in perfiles.items()
            if galeria.voces(perfil)
        }

        sesion.seguir_o_cortar(perfiles, notas, vector, ahora, UMBRAL_GRIS, UMBRAL_VOZ)
        sesion.sumar_al_turno(notas, duracion, ahora)

        orden = sesion.lideres(perfiles)
        mejor, media = orden[0] if orden else (None, -1.0)
        segunda = orden[1][1] if len(orden) > 1 else -1.0
        if mejor is not None and media >= UMBRAL_VOZ and media - segunda >= MARGEN_VOZ:
            sesion.turno["racimo"] = None
            _mejorar_voz(directorio_datos, perfiles, mejor, notas, vector, duracion)
            return {"nombre": mejor, "confianza": round(media, 3)}

        if mejor is not None and media >= UMBRAL_GRIS:
            return {"nombre": None, "dudoso": mejor, "confianza": round(media, 3)}

        return _desconocido_voz(perfiles, vector, duracion, ahora)


def _mejorar_voz(
    directorio_datos: Path,
    perfiles: dict[str, dict[str, Any]],
    nombre: str,
    notas: dict[str, float],
    vector: list[float],
    duracion: float,
) -> None:
    """Guarda esta muestra si es de las buenas: así mejora con el uso.

    Buena es clara (`UMBRAL_REFUERZO_VOZ`), larga (`SEGUNDOS_REFUERZO`) y de
    nadie más: si otra persona conocida pasa siquiera del umbral de duda, la
    muestra no entra. Una muestra que vale para dos es la que contamina un
    perfil, y lo que se contamina ya no se ve.
    """
    otras = [nota for otro, nota in notas.items() if otro != nombre]
    if notas.get(nombre, -1.0) < UMBRAL_REFUERZO_VOZ or duracion < SEGUNDOS_REFUERZO:
        return
    if max(otras, default=-1.0) >= UMBRAL_GRIS:
        return
    perfil = perfiles[nombre]
    if disco.sumar_voces(perfil, [vector], MAX_VOCES):
        guardar(directorio_datos, perfiles)
        disco.anotar(
            directorio_datos,
            "muestra",
            nombre,
            f"voz ({notas[nombre]:.2f}, {duracion:.1f} s); {len(perfil['voces'])} en total",
        )


def _etiqueta_para_voz_nueva(perfiles: dict[str, dict[str, Any]], ahora: float) -> str:
    """La etiqueta del racimo nuevo: la de la cara desconocida a la vista, si la hay.

    Solo se hereda de un desconocido —racimo de cara de esta sesión—, nunca de
    alguien con nombre: si la voz del dueño no casara un día y él estuviera
    fuera de plano con una visita delante, se le estaría enseñando al perfil de
    la visita la voz del dueño.
    """
    cara = sesion.cara_unica(ahora)
    if cara and cara not in sesion.clusters_voz and cara in sesion.clusters_cara:
        return cara
    return sesion.nueva_etiqueta(perfiles)


def _desconocido_voz(
    perfiles: dict[str, dict[str, Any]], vector: list[float], duracion: float, ahora: float
) -> dict[str, Any]:
    """Alguien que no es nadie conocido: se le sigue, pero no se guarda nada."""
    etiqueta: str | None = None
    # Dentro del turno, el desconocido que ya hablaba sigue siendo él mientras
    # no deje de parecerse (eso ya lo decidió `sesion.seguir_o_cortar`).
    if sesion.turno["racimo"] in sesion.clusters_voz:
        etiqueta = sesion.turno["racimo"]
    if etiqueta is None:
        etiqueta = sesion.racimo_parecido(sesion.clusters_voz, vector, UMBRAL_VOZ)
    if etiqueta is None:
        etiqueta = _etiqueta_para_voz_nueva(perfiles, ahora)
        sesion.clusters_voz[etiqueta] = galeria.racimo_nuevo(vector, duracion)
    else:
        galeria.acumular(sesion.clusters_voz[etiqueta], vector, duracion)
    sesion.turno["racimo"] = etiqueta
    similitud = galeria.puntuar_racimo(sesion.clusters_voz[etiqueta], vector)
    return {"nombre": etiqueta, "confianza": round(similitud, 3), "provisional": True}


# --------------------------------------------------------------------------- #
# Identificar caras
# --------------------------------------------------------------------------- #


def calidad_cara(deteccion: dict[str, Any]) -> str | None:
    """`None` si la cara vale para enseñar; si no, el motivo en palabras.

    El motivo es para la persona que se está dando de alta: «más cerca»,
    «quieto», «de frente» se arreglan; «no vale» no se arregla.
    """
    _, _, ancho, alto = deteccion["caja"]
    if min(ancho, alto) < CARA_MINIMA_PX:
        return "La cara sale muy pequeña: acércate a la cámara"
    if deteccion.get("puntuacion", 1.0) < PUNTUACION_UTIL:
        return "La cara no se distingue bien: más luz, o quita lo que la tape"
    if deteccion.get("nitidez", NITIDEZ_MINIMA) < NITIDEZ_MINIMA:
        return "La imagen sale movida: quédate quieto un momento"
    if deteccion.get("giro", 0.0) > GIRO_MAXIMO:
        return "La cara sale muy de lado: mira más de frente"
    return None


def _etiqueta_para_cara_nueva(
    perfiles: dict[str, dict[str, Any]], caras_en_cuadro: int, ahora: float
) -> str:
    """Lo mismo que la voz, al revés: si habla un desconocido y solo se ve una cara."""
    habla = sesion.turno["racimo"]
    reciente = sesion.turno_reciente(ahora)
    if caras_en_cuadro == 1 and reciente and habla and habla not in sesion.clusters_cara:
        if habla in sesion.clusters_voz:
            return habla
    return sesion.nueva_etiqueta(perfiles)


def identificar_cara(directorio_datos: Path, imagen_base64: str) -> dict[str, Any]:
    """Quién sale en este JPEG: lista de caras con nombre, caja y confianza.

    Las caras con nombre, nítidas y sin nadie parecido cerca, refuerzan su
    galería con el ángulo de hoy si aporta algo. Las desconocidas llevan una
    etiqueta provisional de la sesión y **no se guardan** hasta que alguien
    diga su nombre, igual que las voces.
    """
    motor = obtener_motor_cara(directorio_datos)
    if motor is None:
        return {"error": _motivo_cara or "Motor de caras no disponible"}

    try:
        jpeg = base64.b64decode(imagen_base64)
    except (ValueError, TypeError):
        return {"error": "Imagen ilegible"}
    if not jpeg:
        return {"caras": []}

    detecciones = _con_red_de_seguridad(lambda: motor.detectar(jpeg))
    if isinstance(detecciones, dict):
        return detecciones

    salida: list[dict[str, Any]] = []

    with sesion.bloqueo:
        perfiles = cargar(directorio_datos)
        ahora = sesion.reloj()
        cambio = False
        usadas: set[int] = set()

        for deteccion in detecciones:
            caja = deteccion["caja"]
            vector = deteccion["vector"]
            defecto = calidad_cara(deteccion)
            pista = sesion.pista_para(caja, usadas, ahora)

            # La mejor nota de cada persona, y la de la segunda: un parecido
            # de familia no es la misma persona.
            por_persona = sorted(
                (
                    (max(galeria.coseno(vector, g) for g in perfil["caras"]), nombre)
                    for nombre, perfil in perfiles.items()
                    if perfil.get("caras")
                ),
                reverse=True,
            )
            mejor_similitud, mejor_nombre = por_persona[0] if por_persona else (-1.0, None)
            segunda = por_persona[1][0] if len(por_persona) > 1 else -1.0
            clara = mejor_similitud - segunda >= MARGEN_CARA

            sostenido = sesion.nombre_sostenido(pista)
            if mejor_nombre is not None and mejor_similitud >= UMBRAL_CARA and clara:
                refuerza = mejor_similitud >= UMBRAL_CARA + MARGEN_REFUERZO_CARA
                if defecto is None and refuerza and disco.sumar_caras(
                    perfiles[mejor_nombre], [vector], MAX_CARAS
                ):
                    cambio = True
                    disco.anotar(
                        directorio_datos,
                        "muestra",
                        mejor_nombre,
                        f"cara ({mejor_similitud:.2f}); {len(perfiles[mejor_nombre]['caras'])} en total",
                    )
                entrada = {"nombre": mejor_nombre, "caja": caja, "confianza": round(mejor_similitud, 3)}
            elif mejor_nombre is not None and mejor_similitud >= UMBRAL_CARA:
                entrada = {"nombre": None, "caja": caja, "confianza": round(mejor_similitud, 3), "dudoso": mejor_nombre}
            elif sostenido in perfiles:
                # Alguien conocido en un mal fotograma: se le sigue llamando por
                # su nombre y no se aprende nada de esta imagen.
                entrada = {"nombre": sostenido, "caja": caja, "confianza": round(max(0.0, mejor_similitud), 3)}
            elif defecto is None:
                entrada = _desconocida_cara(perfiles, vector, caja, len(detecciones), ahora)
            else:
                entrada = {"nombre": sostenido, "caja": caja, "confianza": 0.0, "calidad": defecto}

            pista["historial"].append(entrada["nombre"])
            salida.append(entrada)

        sesion.anotar_vistas(ahora, [e["nombre"] for e in salida])
        if cambio:
            guardar(directorio_datos, perfiles)

    return {"caras": salida}


def _desconocida_cara(
    perfiles: dict[str, dict[str, Any]],
    vector: list[float],
    caja: list[int],
    caras_en_cuadro: int,
    ahora: float,
) -> dict[str, Any]:
    """Una cara que no es nadie conocido: se la sigue en la sesión, y ya."""
    etiqueta = sesion.racimo_parecido(sesion.clusters_cara, vector, UMBRAL_CARA)
    if etiqueta is None:
        etiqueta = _etiqueta_para_cara_nueva(perfiles, caras_en_cuadro, ahora)
        sesion.clusters_cara[etiqueta] = galeria.racimo_nuevo(vector, 1.0)
    else:
        galeria.acumular(sesion.clusters_cara[etiqueta], vector, 1.0)
    racimo = sesion.clusters_cara[etiqueta]
    return {
        "nombre": etiqueta,
        "caja": caja,
        "confianza": round(max(0.0, galeria.puntuar_racimo(racimo, vector)), 3),
        "provisional": True,
    }


# --------------------------------------------------------------------------- #
# Gestión manual
# --------------------------------------------------------------------------- #


def _limpiar(nombre: str) -> str:
    return nombre.strip()


def enrolar(
    directorio_datos: Path,
    nombre: str,
    audio: str | None = None,
    imagen: str | None = None,
) -> dict[str, Any]:
    """Crea o refuerza un perfil con una muestra traída a propósito.

    Es el camino corto: en vez de esperar a que los racimos decidan, el señor
    Persus graba seis segundos de su voz desde Ajustes y el perfil nace hecho.
    Varias tomas van sumando a la galería. Una toma que no sirve —casi sin voz,
    la cara movida o de lado— se rechaza **diciendo por qué**, que es lo único
    que permite repetirla bien.
    """
    nombre = _limpiar(nombre)
    if not nombre:
        return {"error": "Falta el nombre del perfil"}

    vector_voz: list[float] | None = None
    segundos_voz = 0.0
    if audio:
        motor = obtener_motor_voz(directorio_datos)
        if motor is None:
            return {"error": _motivo_voz or "Motor de voz no disponible"}
        try:
            voz = senal.recortar_voz(senal.pcm16_a_flotantes(base64.b64decode(audio)))
        except (ValueError, TypeError):
            return {"error": "Audio ilegible"}
        segundos_voz = senal.segundos(voz)
        if segundos_voz < SEGUNDOS_ENROLAR:
            return {
                "error": (
                    f"Se oye muy poca voz ({segundos_voz:.1f} s). "
                    "Habla seguido, cerca del micrófono, durante toda la toma."
                ),
                "segundos_voz": round(segundos_voz, 1),
            }
        vector_voz = _con_red_de_seguridad(lambda: motor.incrustar(voz))
        if isinstance(vector_voz, dict):
            return vector_voz

    vector_cara: list[float] | None = None
    if imagen:
        motor_cara = obtener_motor_cara(directorio_datos)
        if motor_cara is None:
            return {"error": _motivo_cara or "Motor de caras no disponible"}
        try:
            jpeg = base64.b64decode(imagen)
        except (ValueError, TypeError):
            return {"error": "Imagen ilegible"}
        detecciones = _con_red_de_seguridad(lambda: motor_cara.detectar(jpeg))
        if isinstance(detecciones, dict):
            return detecciones
        if not detecciones:
            return {"error": "No se ve ninguna cara en la imagen"}
        # La muestra es de él, no de un grupo: la cara más grande.
        principal = max(detecciones, key=lambda d: d["caja"][2] * d["caja"][3])
        defecto = calidad_cara(principal)
        if defecto is not None:
            return {"error": defecto}
        vector_cara = principal["vector"]

    if sesion.es_desconocido(nombre):
        return {"error": "«Desconocido N» no es un nombre: di cómo se llama de verdad"}

    with sesion.bloqueo:
        perfiles = cargar(directorio_datos)
        nuevo_perfil = nombre not in perfiles
        perfil = disco.perfil_de(perfiles, nombre)
        añadido: list[str] = []
        salida: dict[str, Any] = {"ok": True, "nombre": nombre}

        if vector_voz is not None:
            previas = galeria.voces(perfil)
            if previas:
                # Cuánto se parece a las tomas de antes: una toma que no se
                # parece nada suele ser otra persona o un micrófono distinto, y
                # quien la graba tiene que saberlo.
                salida["parecido_voz"] = round(galeria.puntuar(previas, vector_voz), 3)
            # Una toma de alta entra siempre, aunque se parezca a otra: la ha
            # hecho él a propósito y cuenta como muestra.
            if not disco.sumar_voces(perfil, [vector_voz], MAX_VOCES):
                perfil["muestras"] = perfil.get("muestras", 0) + 1
            salida["segundos_voz"] = round(segundos_voz, 1)
            añadido.append("voz")

        if vector_cara is not None:
            disco.sumar_caras(perfil, [vector_cara], MAX_CARAS)
            añadido.append("cara")

        guardar(directorio_datos, perfiles)
        detalle = []
        if vector_voz is not None:
            detalle.append(f"voz ({segundos_voz:.1f} s)")
        if vector_cara is not None:
            detalle.append("cara")
        disco.anotar(
            directorio_datos,
            "alta" if nuevo_perfil else "toma",
            nombre,
            " y ".join(detalle) + " desde Ajustes",
        )
        salida["añadido"] = añadido
        salida["voces"] = len(galeria.voces(perfil))
        salida["caras"] = len(perfil.get("caras") or [])

        # Nacer con perfil borra los racimos a medio hacer: si ya tiene nombre,
        # el aprendizaje automático sobra y podría chocar con él. El contador no
        # se toca: los «Desconocido N» ya fijados siguen en disco.
        sesion.clusters_voz.clear()
        sesion.clusters_cara.clear()
        sesion.cortar_turno()

    return salida


def renombrar(directorio_datos: Path, antes: str, nuevo: str) -> dict[str, Any]:
    """Le pone el nombre real a un «Desconocido N». Todo lo demás se conserva.

    **También vale para quien aún se está aprendiendo.** Antes solo aceptaba
    perfiles ya fijados, y eso dejaba fuera el caso que más importa: alguien
    entra en la llamada, Perseo le pregunta cómo se llama y lo dice a los diez
    segundos — cuando su racimo todavía no ha llegado a los doce segundos de
    voz ni a los ocho fotogramas de cara que hacen falta para fijarse solo. El
    2026-08-25 el padre del señor Persus se quedó en «Desconocido» toda la
    llamada por esto. Ahora el nombre CIERRA el aprendizaje: el racimo se fija
    en ese instante con el nombre real, que es exactamente la información que
    faltaba y que ningún vector podía dar.

    Y como voz y cara pueden compartir etiqueta, se cierran las dos cosas a la
    vez: el perfil fijado por la voz y el racimo de cara a medias, o al revés.
    """
    antes, nuevo = _limpiar(antes), _limpiar(nuevo)
    if not nuevo:
        return {"error": "Falta el nombre nuevo"}
    if sesion.es_desconocido(nuevo):
        return {"error": "«Desconocido N» no es un nombre: di cómo se llama de verdad"}
    with sesion.bloqueo:
        perfiles = cargar(directorio_datos)
        racimo_voz = sesion.clusters_voz.pop(antes, None)
        racimo_cara = sesion.clusters_cara.pop(antes, None)

        if antes in perfiles:
            if nuevo in perfiles and nuevo != antes:
                # Nada cambia: los racimos vuelven a su sitio.
                if racimo_voz is not None:
                    sesion.clusters_voz[antes] = racimo_voz
                if racimo_cara is not None:
                    sesion.clusters_cara[antes] = racimo_cara
                return {"error": f"Ya existe un perfil llamado '{nuevo}'"}
            perfiles[nuevo] = perfiles.pop(antes)
            aprendido = False
        elif racimo_voz is None and racimo_cara is None:
            # Ni perfil fijado ni racimo a medias: aquí no hay nadie con ese nombre.
            return {"error": f"No hay ningún perfil llamado '{antes}'"}
        else:
            aprendido = True

        # Si ya existe un perfil con el nombre nuevo, lo aprendido se le suma:
        # es la misma persona vista por otro canal, no una segunda ficha.
        perfil = disco.perfil_de(perfiles, nuevo)
        detalle = []
        if racimo_voz is not None:
            disco.sumar_voces(perfil, racimo_voz["vectores"], MAX_VOCES)
            detalle.append(f"voz ({racimo_voz['peso']:.0f} s oídos)")
        if racimo_cara is not None:
            disco.sumar_caras(perfil, racimo_cara["vectores"], MAX_CARAS)
            detalle.append(f"cara ({racimo_cara['peso']:.0f} fotogramas)")

        guardar(directorio_datos, perfiles)
        if aprendido:
            disco.anotar(directorio_datos, "nombrado", nuevo, f"era «{antes}»; " + " y ".join(detalle))
        else:
            disco.anotar(directorio_datos, "renombrado", nuevo, f"antes «{antes}»")
        # Lo que el turno acumuló iba a nombre de la etiqueta vieja.
        sesion.cortar_turno()

    salida: dict[str, Any] = {"ok": True, "nombre": nuevo}
    if aprendido:
        salida["aprendido"] = True
    return salida


def borrar(directorio_datos: Path, nombre: str) -> dict[str, Any]:
    """Borra el perfil y sus vectores. No hay copia: eso es lo pedido."""
    nombre = _limpiar(nombre)
    with sesion.bloqueo:
        perfiles = cargar(directorio_datos)
        if nombre not in perfiles:
            return {"error": f"No hay ningún perfil llamado '{nombre}'"}
        del perfiles[nombre]
        guardar(directorio_datos, perfiles)
        sesion.cortar_turno()
    disco.anotar(directorio_datos, "borrado", nombre, "con todas sus muestras")
    return {"ok": True}


def estado_completo(directorio_datos: Path) -> dict[str, Any]:
    """Todo lo que pinta la pantalla de ajustes, en una respuesta."""
    with sesion.bloqueo:
        perfiles = cargar(directorio_datos)
        return {
            "perfiles": resumen(perfiles),
            "aprendiendo": {
                "voz": sesion.progreso(sesion.clusters_voz, SEGUNDOS_VOZ_APRENDER),
                "cara": sesion.progreso(sesion.clusters_cara, FRAMES_CARA_APRENDER),
            },
            "disponibilidad": disponibilidad(directorio_datos),
            "registro": disco.ultimas(directorio_datos, 10),
        }
