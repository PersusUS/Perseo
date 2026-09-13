"""Política de confirmación por niveles.

R15 del plan pedía "poca fricción de confirmación", y eso no puede significar
"ninguna": el sistema lee correo y pantalla, y eso es contenido no confiable que
llega a un agente con manos. La salida de §7 es por niveles, no por interruptor:

| Nivel | Qué es | Qué pasa |
|---|---|---|
| **libre** | Leer correo, leer el vault, buscar, abrir una app | Se ejecuta |
| **reversible** | Escribir en el vault, editar código, mover un evento | Se ejecuta, y queda registrado |
| **irreversible** | Mandar un correo, borrar, `git push`, teclear a ciegas | Se para y pide un sí |

Hasta ahora cada agente llevaba su política escrita a mano: `pc` no preguntaba
nada, `dev` tenía listas blancas y negras, `memoria` simplemente no borraba. Eso
funciona hasta que se escribe el agente número siete y se olvida una de las tres.
Aquí está en un sitio, en forma de tabla, y **se aplica en el trabajador**, antes
de que el agente llegue a ejecutarse.

**Y desde el 2026-09-12, una dimensión más: quién lo pide.** Cada trabajo puede
traer el perfil de la persona que habló —lo pone el reconocimiento de voz de la
llamada— y una orden de alguien que no es el dueño se para aunque el modo
confianza esté encendido. La confianza dice «hay alguien delante», no
«cualquiera que hable manda»: sin esta regla, tener una visita en la sala
convertía la llamada en un mando a distancia para cualquiera.

Dos decisiones que conviene entender:

1. **Lo que no está en la tabla es irreversible.** Un agente nuevo, o una acción
   nueva de uno viejo, pide confirmación hasta que alguien decida en qué nivel
   está. El fallo por exceso cuesta un clic; el fallo por defecto cuesta un
   correo enviado que no querías.
2. **El modo confianza baja el tercer nivel al segundo, y caduca solo.** Es para
   cuando estás delante del PC: dictar por voz sin que cada frase pida permiso.
   Que caduque no es un detalle — un interruptor que se queda encendido para
   siempre es exactamente lo que esta política quiere evitar. Durante una
   llamada, la cara lo renueva con la voz del dueño en ventanas cortas, así que
   se apaga solo cuando el que habla deja de ser él.
3. **Un sí vale para la repetición exacta de lo mismo durante unos minutos.**
   Ver `MINUTOS_REPETICION`. No es una rendija: la petición tiene que ser
   idéntica hasta el último parámetro, lo crítico nunca entra, y apagar la
   confianza borra los síes guardados.

**Y desde el 2026-09-12, nada de esto llega a aplicarse.** Todo lo de arriba
sigue escrito, probado y vivo, pero hay un interruptor por encima —
`CONFIRMACIONES`, aquí abajo— que el dueño puso en apagado, y con él nada se
para. La tabla de esta cabecera describe lo que la política **decide**;
mientras el interruptor esté apagado, la columna de la derecha es «se
ejecuta» en las cuatro filas. La razón y lo que cuesta están escritos en el
propio interruptor, que es donde alguien los buscará el día que quiera
volver a encenderlo.


"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from . import identidad
from ..dominio.niveles import CRITICO, IRREVERSIBLE, LIBRE, NIVELES, REVERSIBLE

logger = logging.getLogger(__name__)

#: Qué nivel tiene cada cosa. La clave es el agente, o `agente.accion` cuando el
#: agente hace cosas de niveles distintos. Lo que no esté aquí es irreversible.
TABLA: dict[str, str] = {
    # Leer nunca cambia nada.
    "correo": LIBRE,
    "correo.triar": LIBRE,
    "agenda": LIBRE,
    "agenda.avisar": LIBRE,
    "memoria.buscar": LIBRE,
    "memoria.leer": LIBRE,
    "web.leer": LIBRE,
    "web.buscar": LIBRE,
    "eco": LIBRE,
    # `simulacro` se queda libre a propósito: lleva su propia
    # `NecesitaConfirmacion` dentro y es lo que prueba el circuito. Si además lo
    # parase la política, preguntaría dos veces.
    "simulacro": LIBRE,
    # Escribir en el vault y editar código son reversibles: están en el vault y
    # en git, y la memoria además nunca sobrescribe.
    "memoria.anotar": REVERSIBLE,
    "memoria.conversacion": REVERSIBLE,
    "dev": REVERSIBLE,
    # Un borrador de correo es reversible **porque no se puede enviar**: el
    # testigo pide `gmail.compose`, que escribe borradores y no incluye `send`.
    # Lo que queda es un texto en la carpeta de borradores, visible y borrable,
    # y darle a enviar sigue siendo un gesto de una persona.
    #
    # Esta línea tiene que estar. `correo` entero está como LIBRE por el triaje,
    # así que sin una entrada propia, redactar heredaría "libre" y escribiría en
    # la cuenta sin que constara en ninguna parte.
    "correo.redactar": REVERSIBLE,
    # Del PC, lo que solo abre cosas es libre; teclear y los atajos van a ciegas
    # sobre la ventana que tenga el foco, y eso puede ser cualquier cosa.
    "pc.abrir_app": LIBRE,
    "pc.buscar_youtube": LIBRE,
    "pc.volumen": LIBRE,
    "pc.mover_raton": LIBRE,
    "pc.click_raton": LIBRE,
    "pc.escribir_teclado": IRREVERSIBLE,
    "pc.atajo_teclado": IRREVERSIBLE,
    # El catálogo de servidores MCP es solo lectura; llamar a una herramienta
    # de un servidor no está aquí a propósito: su nivel lo decide el servidor
    # en mcp.json (ver el gancho de abajo).
    "mcp.servidores": LIBRE,
    # El chat escrito es conversación, no manos: lo irreversible que pueda hacer
    # pasa por los agentes de abajo (pc.escribir_teclado, dev…), que ya tienen
    # aquí su nivel y piden su sí por el camino de siempre. Parar el turno del
    # chat entero sería preguntar dos veces por lo mismo.
    "chat": LIBRE,
}

#: **Si el sistema para algo alguna vez, o no para nunca.**
#:
#: El 2026-09-12 el señor Persus lo apagó entero, y la razón se escribe aquí
#: para que volver a encenderlo no necesite arqueología. Las confirmaciones que
#: le llegaban no venían de un peligro: venían de un camino roto. La ventana de
#: confianza se caía a los diez minutos porque el reconocimiento de voz no le
#: nombraba —`UMBRAL_VOZ` está puesto alto a propósito, «mejor un Desconocido»—,
#: así que «no sé quién habla» acababa tratándose igual que «no es él». Y
#: preguntar por algo que acababa de ordenar en voz alta no protegía de nada.
#: Su decisión, con sus palabras: «cualquier usuario puede decir una tarea y no
#: requerir confirmación no es tan preocupante como para tener que usarse».
#:
#: **Lo que NO se ha hecho: borrar la política.** La tabla de niveles, el modo
#: confianza, las repeticiones y `pide_confirmacion` siguen enteros y probados,
#: y sus pruebas siguen corriendo. Este interruptor es lo único que se
#: interpone. Rearmar el sistema es ponerlo en `True`, o arrancar con
#: `PERSEO_CONFIRMACIONES=1` sin tocar el código.
#:
#: **Lo que se pierde mientras esté apagado, dicho sin adornos:** `CRITICO`
#: también. Borrar ficheros, tocar el registro, matar procesos y `PowerShell`
#: salen sin preguntar. Esa es exactamente la puerta por la que el 2026-08-27
#: salió un `Remove-Item` que nadie autorizó, y que es la razón de que el nivel
#: exista. Ver la cabecera de `CRITICO` en `dominio/niveles.py`.
CONFIRMACIONES = os.environ.get("PERSEO_CONFIRMACIONES", "").strip().lower() in (
    "1",
    "si",
    "sí",
    "true",
)


#: Cuánto dura el modo confianza si no se dice otra cosa. Una sesión de trabajo,
#: no un día: la idea es que se acabe sola antes de que te olvides de ella.
MINUTOS_CONFIANZA = 60

#: Tope duro. Aunque se pidan mil minutos, no se conceden más que estos.
MAX_MINUTOS_CONFIANZA = 480

#: Cuánto vale un sí para las repeticiones EXACTAS de lo mismo. Dictar una
#: dirección letra a letra son seis `escribir_teclado` idénticos en dos minutos,
#: y preguntar seis veces por lo mismo no protege de nada: enseña a decir que sí
#: sin leer, que es el peor sitio al que puede llegar una confirmación.
#:
#: Tres cosas la mantienen estrecha: solo cuenta si la petición es **idéntica**
#: —mismo agente, misma acción, mismos parámetros—, nunca se aplica a lo
#: crítico, y vive en memoria, así que reiniciar el núcleo la borra.
MINUTOS_REPETICION = 10

#: Huella de cada sí reciente y hasta cuándo vale. En memoria a propósito: ver
#: arriba.
_repeticiones: dict[str, datetime] = {}

_fichero: Path | None = None

#: Quien tenga niveles propios —hoy, los servidores MCP— registra aquí una
#: función que responde por su gente. Devuelve un nivel de `NIVELES` o `None`
#: si la petición no es suya, y entonces manda la tabla y el camino de siempre:
#: lo desconocido es irreversible.
_extra_niveles = None


def registrar_niveles(fn) -> None:
    global _extra_niveles
    _extra_niveles = fn


def iniciar(directorio_datos: Path | str) -> None:
    global _fichero
    _fichero = Path(directorio_datos) / "confianza.txt"
    if not CONFIRMACIONES:
        logger.warning(
            "Las confirmaciones están APAGADAS (politica.CONFIRMACIONES): nada se parará a pedir un sí, tampoco lo crítico."
        )


def _ahora() -> datetime:
    return datetime.now(timezone.utc)


def nivel(agente: str, peticion: dict[str, Any] | None = None) -> str:
    """En qué nivel cae esta petición.

    Se mira primero `agente.accion` y luego el agente a secas, para que un agente
    pueda tener acciones de niveles distintos sin partirlo en dos.
    """
    accion = str((peticion or {}).get("accion", "")).strip().lower()
    if accion and f"{agente}.{accion}" in TABLA:
        return TABLA[f"{agente}.{accion}"]
    if agente in TABLA:
        return TABLA[agente]
    if _extra_niveles is not None:
        extra = _extra_niveles(agente, peticion)
        if extra in NIVELES:
            return extra
        if extra is not None:
            logger.warning("Un nivel raro (%r) llegó desde fuera; irreversible.", extra)
    # Lo desconocido pregunta. Ver la decisión 1 de la cabecera.
    return IRREVERSIBLE


# --------------------------------------------------------------------------- #
# Modo confianza
# --------------------------------------------------------------------------- #


def activar_confianza(minutos: float = MINUTOS_CONFIANZA) -> datetime:
    """Enciende el modo confianza y devuelve hasta cuándo dura."""
    if _fichero is None:
        raise RuntimeError("La política no está iniciada; falta llamar a politica.iniciar().")

    minutos = max(1.0, min(float(minutos), MAX_MINUTOS_CONFIANZA))
    hasta = _ahora() + timedelta(minutes=minutos)
    _fichero.write_text(hasta.isoformat(), encoding="utf-8")
    logger.warning(
        "Modo confianza activado hasta las %s UTC: lo irreversible dejará de pedir un sí.",
        hasta.strftime("%H:%M"),
    )
    return hasta


def desactivar_confianza() -> None:
    # Apagar la confianza es decir «vuelve a preguntármelo todo», así que los
    # síes recientes se van con ella. Si no, quedaría una ventana de diez
    # minutos en la que lo irreversible seguiría pasando solo.
    olvidar_repeticiones()
    if _fichero is not None and _fichero.exists():
        _fichero.unlink()
        logger.info("Modo confianza apagado.")


def confianza_hasta() -> datetime | None:
    """Hasta cuándo dura el modo confianza, o `None` si no está activo.

    Un fichero caducado se borra al leerlo: así el estado en disco no miente
    después de que el plazo pase.
    """
    if _fichero is None or not _fichero.exists():
        return None
    try:
        hasta = datetime.fromisoformat(_fichero.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        # Un fichero ilegible se trata como "no hay confianza", que es el lado
        # seguro, y se quita de en medio.
        desactivar_confianza()
        return None

    if hasta.tzinfo is None:
        hasta = hasta.replace(tzinfo=timezone.utc)
    if hasta <= _ahora():
        desactivar_confianza()
        return None
    return hasta


def hay_confianza() -> bool:
    return confianza_hasta() is not None


# --------------------------------------------------------------------------- #
# «Igual que el anterior»
# --------------------------------------------------------------------------- #


def _huella(agente: str, peticion: dict[str, Any] | None) -> str:
    """La misma orden dicha dos veces da la misma huella, y una parecida no.

    `sort_keys` es lo que hace que dos diccionarios con las claves en otro orden
    cuenten como lo mismo; cualquier diferencia en un parámetro —una letra del
    texto que se teclea— da una huella distinta y vuelve a preguntar.
    """
    return agente + "|" + json.dumps(peticion or {}, sort_keys=True, ensure_ascii=False)


def recordar_aprobacion(
    agente: str, peticion: dict[str, Any] | None = None, quien: str | None = None
) -> None:
    """Apunta que esto acaba de aprobarse, para no preguntarlo otra vez enseguida.

    Se llama desde el trabajador, en el mismo sitio donde se aplica la política:
    un sí que no llegó a ejecutarse no vale, y tener las dos cosas en una sola
    función es lo que evita que el día de mañana alguien añada un camino de
    aprobación y se olvide de este.
    """
    if nivel(agente, peticion) != IRREVERSIBLE:
        # Lo crítico no se acumula: cada vez es cada vez. Y lo libre o
        # reversible no pregunta, así que no hay nada que recordar.
        return
    if quien is not None and not identidad.es_el_dueno(quien):
        return
    _repeticiones[_huella(agente, peticion)] = _ahora() + timedelta(minutes=MINUTOS_REPETICION)


def hay_repeticion(agente: str, peticion: dict[str, Any] | None = None) -> bool:
    """Si esto mismo se aprobó hace poco. Limpia lo caducado al pasar."""
    huella = _huella(agente, peticion)
    hasta = _repeticiones.get(huella)
    if hasta is None:
        return False
    if hasta <= _ahora():
        _repeticiones.pop(huella, None)
        return False
    return True


def olvidar_repeticiones() -> None:
    """Borra los síes recientes. Lo usa el apagado del modo confianza: quien lo
    apaga está diciendo «vuelve a preguntármelo todo»."""
    _repeticiones.clear()


# --------------------------------------------------------------------------- #
# La decisión
# --------------------------------------------------------------------------- #


def pide_confirmacion(
    agente: str, peticion: dict[str, Any] | None = None, quien: str | None = None
) -> bool:
    """Si esta petición hay que parar y preguntar.

    `quien` es el perfil de la persona que la pidió, tal y como lo etiquetó el
    reconocimiento de voz («Persus», «Javi», «Desconocido 3»), o `None` cuando
    no se sabe. Es lo que separa una orden del dueño de una de una visita, y
    llega hasta aquí desde la llamada por `almacen.encolar`.
    """
    en_juego = nivel(agente, peticion)
    if en_juego == LIBRE:
        # Leer no cambia nada, tampoco pedido por una visita. Que se cuente o
        # no delante de ella es harina de otro costal, y esa decisión es del
        # modelo, con las reglas de trato que ya lleva en sus instrucciones.
        return False
    if quien is not None and not identidad.es_el_dueno(quien):
        # Una visita no mueve las manos de esta casa. Ni con el modo confianza
        # encendido: la confianza dice «hay alguien delante», no «cualquiera
        # que hable manda». Es la otra mitad de la lección de N-3, que en la
        # cabecera de `CRITICO` se cuenta desde el otro lado.
        return True
    if en_juego == CRITICO:
        # Lo crítico pregunta siempre, con confianza o sin ella. Es la única
        # puerta que no se queda abierta durante una llamada.
        return True
    if en_juego != IRREVERSIBLE:
        return False
    if hay_repeticion(agente, peticion):
        # Esto mismo, palabra por palabra, se aprobó hace menos de
        # MINUTOS_REPETICION. Ver la constante.
        return False
    # El modo confianza baja lo irreversible a reversible mientras dura.
    return not hay_confianza()


def hay_que_parar(
    agente: str, peticion: dict[str, Any] | None = None, quien: str | None = None
) -> bool:
    """Si este trabajo se para de verdad, aquí y ahora.

    Es lo único que mira el trabajador, y existe para que el interruptor viva
    en un sitio y no repartido por los caminos de ejecución. `pide_confirmacion`
    dice lo que la política **querría**; esta dice lo que **pasa**. Separarlas
    es lo que permite apagar las preguntas sin dejar la política sin probar:
    todo lo de arriba se sigue comprobando aunque hoy no llegue a aplicarse.
    """
    if not CONFIRMACIONES:
        return False
    return pide_confirmacion(agente, peticion, quien)


def resumir(
    agente: str, peticion: dict[str, Any] | None = None, quien: str | None = None
) -> str:
    """La pregunta que se enseña. Corta: sale por Telegram, donde solo va el titular."""
    peticion = peticion or {}
    accion = str(peticion.get("accion", "")).strip()
    que = f"{agente} · {accion}" if accion else agente
    if quien is not None and not identidad.es_el_dueno(quien):
        # Quién lo pidió es LO que hay que decidir aquí, así que va en el
        # titular y no en el detalle: una visita pidiendo teclear no es la
        # misma pregunta que la de siempre.
        return f"Lo pide {quien}, que no eres tú. ¿Lo autorizas? ({que})"
    if nivel(agente, peticion) == CRITICO:
        return f"¿Confirmas algo que no se puede deshacer? ({que})"
    return f"¿Confirmas una acción irreversible? ({que})"
