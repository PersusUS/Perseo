"""Agente `seguimiento`: lo que se quedó sin contestar, dicho antes de que se olvide.

Instinct «retoma los hilos que se caen sin que se lo pidan dos veces». Aquí el
hilo que se cae es el de siempre: un correo que el triaje marcó como «requiere
acción» y que dos días después sigue sin respuesta.

**Sin respuesta de verdad, no sin marcar.** Casi nadie marca los correos en el
panel: se contestan desde Gmail. Si el seguimiento se fiara de la marca,
llamaría por correos ya contestados, y un aviso que se equivoca dos veces se
desactiva a la tercera. Por eso el agente mira el hilo en Gmail —`format=minimal`,
solo etiquetas—: si el último mensaje lleva `SENT`, ya contestó, y el correo se
marca como atendido sin molestar. Solo lo que sigue callado hace sonar el timbre.

Cómo se reparten el trabajo, que es la regla de `infra/disparadores.py`: el
disparador solo **encuentra** candidatos, con lo que ya hay en la base; mirar
Gmail, marcar y avisar lo hace el agente, en la cola, a la vista.

**Una vez por correo, y a horas de persona.** Cada correo se avisa una sola vez
(`<datos>/seguimiento_avisados.json`), y el disparador solo encola entre las
nueve y las nueve: un «llevas dos días sin contestar» a las tres de la mañana no
es seguimiento, es ruido.
"""

from __future__ import annotations

import asyncio
import email.utils
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..infra import almacen, disparadores
from ..infra.configuracion import Configuracion
from ..infra.router import registrar
from ..servicios import correo_lectura, llamada_saliente, recordatorios
from . import correo

logger = logging.getLogger(__name__)

#: Desde cuándo un correo que pide algo cuenta como «sin contestar».
DIAS_SIN_RESPUESTA = 2

#: Hasta cuándo se avisa. Lo que lleva más de dos semanas ya no es un hilo que
#: se cae: es uno que se decidió no seguir, y avisar de él el día que se
#: enciende esto sería una avalancha de correos viejos.
DIAS_TOPE = 14

HORA_DESDE = 9
HORA_HASTA = 21

#: Cuántos se nombran en la llamada. El resto se cuenta.
TOPE_NOMBRADOS = 4

_cfg: Configuracion | None = None
_avisados: disparadores.Vistos | None = None


def iniciar(cfg: Configuracion) -> None:
    global _cfg
    _cfg = cfg


def edad(fecha: str, ahora: datetime) -> timedelta | None:
    """Cuánto hace de un correo, por su cabecera `Date` (o ISO, en el buzón falso)."""
    if not fecha:
        return None
    try:
        cuando = email.utils.parsedate_to_datetime(fecha)
    except (TypeError, ValueError):
        try:
            cuando = datetime.fromisoformat(fecha)
        except ValueError:
            return None
    if cuando.tzinfo is None:
        cuando = cuando.replace(tzinfo=ahora.tzinfo)
    return ahora - cuando


def candidatos(correos: list[dict[str, Any]], ahora: datetime, avisados: set[str]) -> list[dict[str, Any]]:
    """Los que piden algo, siguen sin marcar y llevan entre 2 y 14 días."""
    salida = []
    for c in correos:
        if c.get("clase") != "requiere_accion" or c.get("hecho") or c.get("id") in avisados:
            continue
        cuanto = edad(str(c.get("fecha") or ""), ahora)
        if cuanto is None or not timedelta(days=DIAS_SIN_RESPUESTA) <= cuanto <= timedelta(days=DIAS_TOPE):
            continue
        salida.append({**c, "dias": cuanto.days})
    return salida


def _motivo(sin_respuesta: list[dict[str, Any]]) -> str:
    nombrados = [
        f"{c.get('remitente') or 'alguien'} («{c.get('asunto') or 'sin asunto'}», {c.get('dias')} días)"
        for c in sin_respuesta[:TOPE_NOMBRADOS]
    ]
    resto = len(sin_respuesta) - len(nombrados)
    cola = f" y {resto} más" if resto > 0 else ""
    return f"Siguen sin respuesta correos que pedían algo: {'; '.join(nombrados)}{cola}."


@registrar("seguimiento")
async def _seguimiento(trabajo: dict[str, Any]) -> dict[str, Any]:
    correos = [c for c in (trabajo.get("peticion") or {}).get("correos") or [] if isinstance(c, dict)]
    buzon = correo.buzon()
    mirar = getattr(buzon, "respondido", None)

    contestados: list[dict[str, Any]] = []
    sin_respuesta: list[dict[str, Any]] = []
    for c in correos:
        respondido = False
        if mirar is not None and c.get("hilo"):
            try:
                respondido = await mirar(str(c["hilo"]))
            except Exception as e:  # noqa: BLE001 — sin saberlo, se avisa: es el lado que no calla
                logger.warning("Seguimiento: no se pudo mirar el hilo %s: %s", c.get("hilo"), e)
        (contestados if respondido else sin_respuesta).append(c)

    for c in contestados:
        await asyncio.to_thread(almacen.marcar_correo, str(c["id"]), almacen.ATENDIDO)

    if not sin_respuesta:
        return {
            "texto": f"{len(contestados)} correo(s) ya tenían respuesta suya; marcados como atendidos.",
            "titular": None,
            "callado": True,
        }
    motivo = _motivo(sin_respuesta)
    await asyncio.to_thread(llamada_saliente.llamar, motivo)
    return {
        "texto": motivo
        + (f" ({len(contestados)} más ya tenían respuesta y quedan atendidos.)" if contestados else ""),
        # Por Telegram, solo cuántos: remitentes y asuntos son contenido suyo.
        "titular": f"{len(sin_respuesta)} correo(s) que pedían algo siguen sin respuesta",
    }


@disparadores.registrar("seguimiento", intervalo=1800)
async def _vigilar_seguimiento(ctx: disparadores.Contexto) -> None:
    global _avisados
    ahora = recordatorios.ahora_local()
    if not HORA_DESDE <= ahora.hour < HORA_HASTA:
        return
    if _avisados is None:
        _avisados = disparadores.Vistos(ruta=Path(ctx.cfg.directorio_datos) / "seguimiento_avisados.json").cargar()
    try:
        correos = await asyncio.to_thread(correo_lectura.cargar_correos, Path(ctx.cfg.ruta_db))
    except Exception as e:  # noqa: BLE001 — una base sin correos todavía no es un fallo
        logger.debug("Seguimiento: sin correos que leer (%s).", e)
        return
    pendientes = candidatos(correos, ahora, _avisados.ids)
    if not pendientes:
        return
    # Se anotan al encolar, como en el correo: si el trabajo falla, repetirlo
    # cada media hora no lo arreglaría, y queda visible en la cola.
    _avisados.anotar([str(c["id"]) for c in pendientes])
    campos = ("id", "hilo", "remitente", "asunto", "dias")
    trabajo = await ctx.encolar(
        "seguimiento", {"correos": [{k: c.get(k) for k in campos} for c in pendientes]}
    )
    logger.info("Seguimiento: %d correo(s) sin contestar, trabajo %s.", len(pendientes), trabajo["id"])
