"""Quién habla y quién sale por la cámara: perfiles biométricos locales.

La idea que esto cumple llega del señor Persus: en una llamada, Perseo debería
saber **quién** está delante —ponerle nombre a quien habla, etiquetar la cara que
ve— y, si no conoce a alguien, **aprenderlo con la propia llamada**: acumular voz
suficiente de ese desconocido y reconocerlo la próxima vez sin que nadie entrene
nada.

**Dónde vive la inteligencia.** En el núcleo, como manda la regla del plan: las
caras no piensan. La app de voz solo transporta trozos —PCM del micrófono y JPEG
de la cámara, los mismos que ya le manda a Gemini— y pinta la etiqueta que le
devuelve esta pieza. Aquí dentro ocurre todo lo demás: extraer el vector de la
voz (ECAPA-TDNN) o de la cara (YuNet + SFace), compararlo contra los perfiles
guardados, y decidir si es alguien conocido, alguien que se está aprendiendo o
ruido.

**Los modelos son opcionales, como Ollama.** `torch`+`speechbrain` pesan cientos
de megas y `opencv-python` otros tantos: obligar a todo el mundo a instalarlos
para arrancar el núcleo sería romper la Raspberry Pi del plan por un capricho.
Sin ellos el núcleo arranca igual y las rutas de biometría contestan con un
motivo claro (`disponibilidad()`), igual que `estado.telemetria()` devuelve
`disponible: false` sin psutil. Lo que hace falta va anotado en
`requirements-biometria.txt`.

**Privacidad, que aquí no es decorado.** Lo que se guarda son números, nunca
audio ni imágenes: por cada persona, un vector de 192 números (voz) y otro de
128 (cara). Viven en `<datos>/perfiles.json`, que ya está fuera de git porque
todo `perseo_core/datos/` lo está. Nada sale de la máquina: ni los vectores ni
las muestras viajan a ningún servicio; el único tercero que ve algo es Gemini,
que ya veía el mismo micrófono y la misma cámara antes de que existiera este
módulo. Y borrar un perfil borra sus números de verdad: no hay copia en ninguna
otra parte.

**Cómo se aprende a un desconocido.** Cada trozo de voz que no casa con ningún
perfil se compara primero contra los *racimos* de la sesión en curso (en
memoria): si parece el mismo desconocido de antes, engorda su racimo; si no,
abre uno nuevo («Desconocido 1», «Desconocido 2»…). Cuando un racimo acumula
`SEGUNDOS_VOZ_APRENDER` de voz útil —unos doce segundos hablando normal— se
fija como perfil permanente con esa etiqueta, que luego puede cambiarse por el
nombre real (`renombrar`). Con las caras igual, contando detecciones buenas en
vez de segundos. Al reiniciar el núcleo los racimos a medio aprender se pierden:
aprender a medias es exactamente eso, y los perfiles fijados sí sobreviven.

Ver bitacora/05_PLAN_PERSEO_V2.md §2 (el núcleo decide) y §7 (lo observado no
es instrucción: un rostro conocido no autoriza nada).
"""

from __future__ import annotations

import base64
import json
import logging
import math
import struct
import threading
import time
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constantes de decisión
# --------------------------------------------------------------------------- #

#: Fichero de perfiles dentro de `<datos>/`. Fuera de git como todo `datos/`.
NOMBRE_FICHERO = "perfiles.json"

#: Carpeta de modelos descargados, dentro de `<datos>/`.
CARPETA_MODELOS = "modelos"

#: Similitud coseno mínima para decir «esta voz es la de ese perfil». Con
#: ECAPA-TDNN el mismo hablante suele rondar 0.65-0.90 y otra persona 0.20-0.45;
#: el umbral se queda en medio, conservador a propósito: mejor un «Desconocido»
#: de más que un nombre equivocado puesto con seguridad.
UMBRAL_VOZ = 0.55

#: Umbral coseno de SFace para «es la misma cara». Es el que recomienda OpenCV
#: en su zoo de modelos (0.363), medido sobre LFW; no se afina a mano.
UMBRAL_CARA = 0.363

#: Segundos de voz útil —con energía, no silencio— que acumula un racimo de
#: desconocido antes de fijarse como perfil. Menos de ~10 s el vector sale
#: inestable entre sesiones; más retrasa aprender a quien ya está delante.
SEGUNDOS_VOZ_APRENDER = 12.0

#: Detecciones buenas de la misma cara desconocida antes de fijar perfil.
FRAMES_CARA_APRENDER = 8

#: Cuántos vectores de cara guarda un perfil, como mucho. Distintos ángulos
#: ayudan a reconocer de perfil o con gafas; a partir de aquí ya no aprende.
MAX_VECTORES_CARA = 5

#: Muestras mínimas para molestar al motor: menos de ~0,4 s de audio produce
#: vectores basura y gasta CPU en nada.
MUESTRAS_MINIMAS = 6400

FRECUENCIA = 16000


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
    """Lo que se le pide a un detector+reconocedor de caras."""

    def detectar(self, jpeg: bytes) -> list[dict[str, Any]]:
        """Una entrada por cara: {"caja": [x, y, w, h], "vector": [floats]}."""
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
            "cara": UMBRAL_CARA,
            "segundos_aprender": SEGUNDOS_VOZ_APRENDER,
        },
    }


# --------------------------------------------------------------------------- #
# Álgebra mínima, sin numpy a propósito: un producto escalar de 192 números no
# necesita tirar de una librería científica, y así este fichero entero corre en
# cualquier Python 3.11 pelado.
# --------------------------------------------------------------------------- #


def _coseno(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return -1.0
    punto = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return -1.0
    return punto / (na * nb)


def _media(a: list[float], b: list[float]) -> list[float]:
    """Punto medio, renormalizado. Así el racimo converge sin crecer siempre."""
    media = [(x + y) / 2.0 for x, y in zip(a, b)]
    norma = math.sqrt(sum(x * x for x in media)) or 1.0
    return [x / norma for x in media]


def _pcm16_a_flotantes(pcm: bytes) -> list[float]:
    """PCM little-endian int16 mono → flotantes en [-1, 1]. Lo que manda la app."""
    utiles = pcm[: (len(pcm) // 2) * 2]
    enteros = struct.unpack(f"<{len(utiles) // 2}h", utiles)
    return [e / 32768.0 for e in enteros]


def _reforzar(perfil: dict[str, Any], vector: list[float]) -> None:
    """Acerca un poco el vector guardado hacia el de hoy.

    La voz de alguien cambia con el día, el micro y el resfriado; un perfil
    congelado en su primera grabación envejece mal. Cada acierto tira poco
    (media simple) y el contador deja de subir en `TOPE_REFUERZOS`, para que
    cien llamadas seguidas no diluyan la primera impresión buena.
    """
    TOPE_REFUERZOS = 30
    if perfil.get("muestras", 0) < TOPE_REFUERZOS:
        perfil["voz"] = _media(perfil["voz"], vector)
        perfil["muestras"] = perfil.get("muestras", 1) + 1


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
# Perfiles en disco
#
# Formato: {"nombre": {"voz": [...]|null, "caras": [[...]], "muestras": int,
#                      "creado": iso}}
# Escritura atómica (tmp + replace): un corte de luz a mitad de escritura no
# debe quedarse sin perfiles ni dejar un JSON a medias que `cargar` tendría que
# tragarse.
# --------------------------------------------------------------------------- #


def ruta_perfiles(directorio_datos: Path) -> Path:
    return Path(directorio_datos) / NOMBRE_FICHERO


def cargar(directorio_datos: Path) -> dict[str, dict[str, Any]]:
    ruta = ruta_perfiles(directorio_datos)
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError):
        # Un fichero roto no tumba el reconocimiento: se empieza de cero y la
        # siguiente escritura atómica lo repone. Perder unos perfiles es malo;
        # que el núcleo no arranque por ello, peor.
        return {}
    if not isinstance(datos, dict):
        return {}
    return datos


def guardar(directorio_datos: Path, perfiles: dict[str, dict[str, Any]]) -> None:
    ruta = ruta_perfiles(directorio_datos)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(".tmp")
    temporal.write_text(json.dumps(perfiles, ensure_ascii=False, indent=1), encoding="utf-8")
    temporal.replace(ruta)


def resumen(perfiles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Lo que enseña la pantalla: nombres y tamaños, nunca los vectores."""
    salida = []
    for nombre, perfil in sorted(perfiles.items()):
        salida.append(
            {
                "nombre": nombre,
                "voz": bool(perfil.get("voz")),
                "caras": len(perfil.get("caras") or []),
                "muestras": perfil.get("muestras", 0),
                "creado": perfil.get("creado"),
            }
        )
    return salida


# --------------------------------------------------------------------------- #
# Racimos de la sesión: los desconocidos a medio aprender. Memoria, no disco:
# si el núcleo se reinicia a mitad, se pierden, y no pasa nada.
# --------------------------------------------------------------------------- #

_bloqueo_sesion = threading.Lock()
_clusters_voz: dict[str, dict[str, Any]] = {}
_clusters_cara: dict[str, dict[str, Any]] = {}
_contador_desconocidos = 0


def reiniciar_sesion() -> None:
    """Vacía los racimos a medio aprender. Los perfiles fijados no se tocan."""
    global _contador_desconocidos
    with _bloqueo_sesion:
        _clusters_voz.clear()
        _clusters_cara.clear()
        _contador_desconocidos = 0


def _progreso_racimo(clusters: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """El racimo más avanzado, para que la pantalla enseñe cuánto falta."""
    if not clusters:
        return None
    etiqueta, cluster = max(clusters.items(), key=lambda par: par[1]["peso"])
    objetivo = SEGUNDOS_VOZ_APRENDER if clusters is _clusters_voz else FRAMES_CARA_APRENDER
    return {"etiqueta": etiqueta, "peso": round(cluster["peso"], 1), "objetivo": objetivo}


def _racimo_parecido(
    clusters: dict[str, dict[str, Any]], vector: list[float], umbral: float
) -> tuple[str | None, float]:
    mejor_etiqueta: str | None = None
    mejor_similitud = -1.0
    for etiqueta, cluster in clusters.items():
        similitud = _coseno(vector, cluster["vector"])
        if similitud > mejor_similitud:
            mejor_etiqueta, mejor_similitud = etiqueta, similitud
    if mejor_similitud < umbral:
        return None, mejor_similitud
    return mejor_etiqueta, mejor_similitud


# --------------------------------------------------------------------------- #
# Identificar voz
# --------------------------------------------------------------------------- #


def identificar_voz(directorio_datos: Path, audio_base64: str) -> dict[str, Any]:
    """¿De quién es este trozo de PCM?

    Devuelve una de tres cosas: `{"nombre": ..., "confianza": ...}` si es
    alguien con perfil; `{"nombre": "Desconocido N", "aprendiendo": {...}}`
    mientras engorda el racimo de alguien nuevo; o `{"nombre": etiqueta,
    "aprendido": true}` justo el instante en que el racimo se convierte en
    perfil. Con el motor ausente, `{"error": motivo}`: la llamada decide cómo
    decirlo.
    """
    motor = obtener_motor_voz(directorio_datos)
    if motor is None:
        return {"error": _motivo_voz or "Motor de voz no disponible"}

    try:
        pcm = base64.b64decode(audio_base64)
    except (ValueError, TypeError):
        return {"error": "Audio ilegible"}
    muestras = _pcm16_a_flotantes(pcm)
    if len(muestras) < MUESTRAS_MINIMAS:
        return {"nombre": None}

    vector = _con_red_de_seguridad(lambda: motor.incrustar(muestras))
    if isinstance(vector, dict):
        return vector
    duracion = len(muestras) / FRECUENCIA

    with _bloqueo_sesion:
        perfiles = cargar(directorio_datos)

        mejor_nombre: str | None = None
        mejor_similitud = -1.0
        for nombre, perfil in perfiles.items():
            if not perfil.get("voz"):
                continue
            similitud = _coseno(vector, perfil["voz"])
            if similitud > mejor_similitud:
                mejor_nombre, mejor_similitud = nombre, similitud

        if mejor_nombre is not None and mejor_similitud >= UMBRAL_VOZ:
            _reforzar(perfiles[mejor_nombre], vector)
            guardar(directorio_datos, perfiles)
            return {"nombre": mejor_nombre, "confianza": round(mejor_similitud, 3)}

        # Desconocido: a engordar racimo, o a estrenarlo.
        global _contador_desconocidos
        etiqueta, similitud_racimo = _racimo_parecido(_clusters_voz, vector, UMBRAL_VOZ)
        if etiqueta is None:
            _contador_desconocidos += 1
            etiqueta = f"Desconocido {_contador_desconocidos}"
            _clusters_voz[etiqueta] = {"vector": vector, "peso": 0.0}
            similitud_racimo = 1.0
        racimo = _clusters_voz[etiqueta]
        racimo["vector"] = _media(racimo["vector"], vector)
        racimo["peso"] += duracion

        if racimo["peso"] >= SEGUNDOS_VOZ_APRENDER:
            del _clusters_voz[etiqueta]
            ahora = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
            perfiles[etiqueta] = {
                "voz": racimo["vector"],
                "caras": [],
                "muestras": 1,
                "creado": ahora,
            }
            guardar(directorio_datos, perfiles)
            return {
                "nombre": etiqueta,
                "confianza": round(similitud_racimo, 3),
                "aprendido": True,
            }

        # Mientras aprende, el desconocido YA lleva su etiqueta provisional:
        # la pantalla puede decir «Desconocido 1» desde la primera frase, y
        # cuando el racimo madure será siempre esa misma persona.
        return {
            "nombre": etiqueta,
            "confianza": round(similitud_racimo, 3),
            "aprendiendo": _progreso_racimo(_clusters_voz),
        }


# --------------------------------------------------------------------------- #
# Identificar caras
# --------------------------------------------------------------------------- #


def identificar_cara(directorio_datos: Path, imagen_base64: str) -> dict[str, Any]:
    """Quién sale en este JPEG: lista de caras con nombre, caja y confianza.

    Las caras conocidas refuerzan su perfil con el ángulo de hoy si aún le
    quedan huecos (`MAX_VECTORES_CARA`). Las desconocidas van a racimos como la
    voz, contando detecciones en vez de segundos.
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

    global _contador_desconocidos
    salida: list[dict[str, Any]] = []

    with _bloqueo_sesion:
        perfiles = cargar(directorio_datos)
        cambio = False

        for deteccion in detecciones:
            caja = deteccion["caja"]
            vector = deteccion["vector"]

            mejor_nombre: str | None = None
            mejor_similitud = -1.0
            for nombre, perfil in perfiles.items():
                for guardado in perfil.get("caras") or []:
                    similitud = _coseno(vector, guardado)
                    if similitud > mejor_similitud:
                        mejor_nombre, mejor_similitud = nombre, similitud

            if mejor_nombre is not None and mejor_similitud >= UMBRAL_CARA:
                perfil = perfiles[mejor_nombre]
                caras = perfil.setdefault("caras", [])
                # Un ángulo muy parecido al que ya hay no aporta nada; uno
                # distinto sí, si queda sitio. Sin esto el perfil solo vería
                # bien la pose de la primera foto.
                if len(caras) < MAX_VECTORES_CARA and mejor_similitud < 0.95:
                    caras.append(vector)
                    cambio = True
                salida.append(
                    {"nombre": mejor_nombre, "caja": caja, "confianza": round(mejor_similitud, 3)}
                )
                continue

            etiqueta, _ = _racimo_parecido(_clusters_cara, vector, UMBRAL_CARA)
            if etiqueta is None:
                _contador_desconocidos += 1
                etiqueta = f"Desconocido {_contador_desconocidos}"
                _clusters_cara[etiqueta] = {"vector": vector, "peso": 0.0}
            racimo = _clusters_cara[etiqueta]
            racimo["vector"] = _media(racimo["vector"], vector)
            racimo["peso"] += 1

            if racimo["peso"] >= FRAMES_CARA_APRENDER:
                del _clusters_cara[etiqueta]
                ahora = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
                perfiles[etiqueta] = {
                    "voz": None,
                    "caras": [racimo["vector"]],
                    "muestras": 0,
                    "creado": ahora,
                }
                cambio = True
                salida.append(
                    {"nombre": etiqueta, "caja": caja, "confianza": 0.5, "aprendido": True}
                )
            else:
                salida.append(
                    {
                        "nombre": etiqueta,
                        "caja": caja,
                        "confianza": round(max(0.0, _coseno(vector, racimo["vector"])), 3),
                        "aprendiendo": {
                            "etiqueta": etiqueta,
                            "peso": racimo["peso"],
                            "objetivo": FRAMES_CARA_APRENDER,
                        },
                    }
                )

        if cambio:
            guardar(directorio_datos, perfiles)

    return {"caras": salida}


# --------------------------------------------------------------------------- #
# Gestión manual de perfiles
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
    Si el perfil ya existe, la muestra se añade (voz: media; cara: hueco libre).
    """
    nombre = _limpiar(nombre)
    if not nombre:
        return {"error": "Falta el nombre del perfil"}

    with _bloqueo_sesion:
        perfiles = cargar(directorio_datos)
        perfil = perfiles.get(nombre)
        if perfil is None:
            perfil = {
                "voz": None,
                "caras": [],
                "muestras": 0,
                "creado": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            }
            perfiles[nombre] = perfil

        añadido: list[str] = []

        if audio:
            motor = obtener_motor_voz(directorio_datos)
            if motor is None:
                return {"error": _motivo_voz or "Motor de voz no disponible"}
            try:
                muestras = _pcm16_a_flotantes(base64.b64decode(audio))
            except (ValueError, TypeError):
                return {"error": "Audio ilegible"}
            if len(muestras) < MUESTRAS_MINIMAS:
                return {"error": "La muestra de voz es demasiado corta"}
            vector = _con_red_de_seguridad(lambda: motor.incrustar(muestras))
            if isinstance(vector, dict):
                return vector
            perfil["voz"] = _media(perfil["voz"], vector) if perfil.get("voz") else vector
            perfil["muestras"] = perfil.get("muestras", 0) + 1
            añadido.append("voz")

        if imagen:
            motor = obtener_motor_cara(directorio_datos)
            if motor is None:
                return {"error": _motivo_cara or "Motor de caras no disponible"}
            try:
                jpeg = base64.b64decode(imagen)
            except (ValueError, TypeError):
                return {"error": "Imagen ilegible"}
            detecciones = _con_red_de_seguridad(lambda: motor.detectar(jpeg))
            if isinstance(detecciones, dict):
                return detecciones
            if not detecciones:
                return {"error": "No se ve ninguna cara en la imagen"}
            caras = perfil.setdefault("caras", [])
            for deteccion in detecciones[:1]:  # la muestra es de él, no de un grupo
                if len(caras) < MAX_VECTORES_CARA:
                    caras.append(deteccion["vector"])
            añadido.append("cara")

        guardar(directorio_datos, perfiles)

    # Nacer con perfil borra los racimos a medio hacer: si ya tiene nombre, el
    # aprendizaje automático sobra y podría chocar con él.
    reiniciar_sesion()

    return {"ok": True, "nombre": nombre, "añadido": añadido}


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
    """
    antes, nuevo = _limpiar(antes), _limpiar(nuevo)
    if not nuevo:
        return {"error": "Falta el nombre nuevo"}
    with _bloqueo_sesion:
        perfiles = cargar(directorio_datos)
        if antes in perfiles:
            if nuevo in perfiles and nuevo != antes:
                return {"error": f"Ya existe un perfil llamado '{nuevo}'"}
            perfiles[nuevo] = perfiles.pop(antes)
            guardar(directorio_datos, perfiles)
            return {"ok": True, "nombre": nuevo}

        # Ni perfil fijado ni racimo a medias: aquí no hay nadie con ese nombre.
        racimo_voz = _clusters_voz.pop(antes, None)
        racimo_cara = _clusters_cara.pop(antes, None)
        if racimo_voz is None and racimo_cara is None:
            return {"error": f"No hay ningún perfil llamado '{antes}'"}

        # Si ya existe un perfil con el nombre nuevo, lo aprendido se le suma:
        # es la misma persona vista por otro canal, no una segunda ficha.
        perfil = perfiles.get(nuevo)
        if perfil is None:
            perfil = {
                "voz": None,
                "caras": [],
                "muestras": 0,
                "creado": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            }
            perfiles[nuevo] = perfil

        if racimo_voz is not None:
            perfil["voz"] = (
                _media(perfil["voz"], racimo_voz["vector"])
                if perfil.get("voz")
                else racimo_voz["vector"]
            )
            perfil["muestras"] = perfil.get("muestras", 0) + 1
        if racimo_cara is not None:
            caras = perfil.setdefault("caras", [])
            if len(caras) < MAX_VECTORES_CARA:
                caras.append(racimo_cara["vector"])

        guardar(directorio_datos, perfiles)
    return {"ok": True, "nombre": nuevo, "aprendido": True}


def borrar(directorio_datos: Path, nombre: str) -> dict[str, Any]:
    """Borra el perfil y sus vectores. No hay copia: eso es lo pedido."""
    nombre = _limpiar(nombre)
    with _bloqueo_sesion:
        perfiles = cargar(directorio_datos)
        if nombre not in perfiles:
            return {"error": f"No hay ningún perfil llamado '{nombre}'"}
        del perfiles[nombre]
        guardar(directorio_datos, perfiles)
    return {"ok": True}


def estado_completo(directorio_datos: Path) -> dict[str, Any]:
    """Todo lo que pinta la pantalla de ajustes, en una respuesta."""
    with _bloqueo_sesion:
        perfiles = cargar(directorio_datos)
        return {
            "perfiles": resumen(perfiles),
            "aprendiendo": {
                "voz": _progreso_racimo(_clusters_voz),
                "cara": _progreso_racimo(_clusters_cara),
            },
            "disponibilidad": disponibilidad(directorio_datos),
        }
