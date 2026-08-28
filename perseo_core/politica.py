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

Dos decisiones que conviene entender:

1. **Lo que no está en la tabla es irreversible.** Un agente nuevo, o una acción
   nueva de uno viejo, pide confirmación hasta que alguien decida en qué nivel
   está. El fallo por exceso cuesta un clic; el fallo por defecto cuesta un
   correo enviado que no querías.
2. **El modo confianza baja el tercer nivel al segundo, y caduca solo.** Es para
   cuando estás delante del PC: dictar por voz sin que cada frase pida permiso.
   Que caduque no es un detalle — un interruptor que se queda encendido para
   siempre es exactamente lo que esta política quiere evitar.

Ver bitacora/05_PLAN_PERSEO_V2.md §7 (R15).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

LIBRE = "libre"
REVERSIBLE = "reversible"
IRREVERSIBLE = "irreversible"

#: Como irreversible, pero **el modo confianza no lo tapa**. Es para lo que no
#: se puede deshacer con nada: borrar ficheros, tocar el registro, matar
#: procesos. Nació el 2026-08-27 de un caso concreto: la llamada de voz enciende
#: la confianza al conectar (N-3), así que durante una llamada NADA preguntaba;
#: Perseo dijo de su cosecha «¿confirma que ejecuto el comando?», nadie
#: contestó, y el comando salió igual porque el sistema nunca llegó a
#: preguntarlo. Tener delante a alguien hablando no es su sí a esta orden.
CRITICO = "critico"

NIVELES = (LIBRE, REVERSIBLE, IRREVERSIBLE, CRITICO)

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

#: Cuánto dura el modo confianza si no se dice otra cosa. Una sesión de trabajo,
#: no un día: la idea es que se acabe sola antes de que te olvides de ella.
MINUTOS_CONFIANZA = 60

#: Tope duro. Aunque se pidan mil minutos, no se conceden más que estos.
MAX_MINUTOS_CONFIANZA = 480

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
# La decisión
# --------------------------------------------------------------------------- #


def pide_confirmacion(agente: str, peticion: dict[str, Any] | None = None) -> bool:
    """Si esta petición hay que parar y preguntar."""
    en_juego = nivel(agente, peticion)
    if en_juego == CRITICO:
        # Lo crítico pregunta siempre, con confianza o sin ella. Es la única
        # puerta que no se queda abierta durante una llamada.
        return True
    if en_juego != IRREVERSIBLE:
        return False
    # El modo confianza baja lo irreversible a reversible mientras dura.
    return not hay_confianza()


def resumir(agente: str, peticion: dict[str, Any] | None = None) -> str:
    """La pregunta que se enseña. Corta: sale por Telegram, donde solo va el titular."""
    peticion = peticion or {}
    accion = str(peticion.get("accion", "")).strip()
    que = f"{agente} · {accion}" if accion else agente
    if nivel(agente, peticion) == CRITICO:
        return f"¿Confirmas algo que no se puede deshacer? ({que})"
    return f"¿Confirmas una acción irreversible? ({que})"
