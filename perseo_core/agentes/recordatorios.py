"""Agente `recordatorios`: apuntar, listar, quitar y avisar.

Todo lo que sabe de fechas y de dónde se guardan vive en
`servicios/recordatorios.py`; aquí está lo que se le contesta a quien lo pidió
—el modelo, que lo va a decir en voz alta— y el disparador que avisa.

Las respuestas repiten **la hora ya resuelta** («mañana a las 09:00») y no la
que dijo el modelo. Es lo que permite que el señor Persus oiga el error si el
modelo calculó mal «el jueves que viene», en vez de descubrirlo el jueves.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from ..infra import disparadores
from ..infra.configuracion import Configuracion
from ..infra.router import registrar
from ..servicios import llamada_saliente, recordatorios

logger = logging.getLogger(__name__)

_cfg: Configuracion | None = None


def iniciar(cfg: Configuracion) -> None:
    global _cfg
    _cfg = cfg


def _directorio() -> str:
    if _cfg is None:
        raise RuntimeError("El agente recordatorios no está iniciado; falta iniciar(cfg).")
    return str(_cfg.directorio_datos)


@registrar("recordatorios")
async def _recordatorios(trabajo: dict[str, Any]) -> dict[str, Any]:
    peticion = trabajo.get("peticion") or {}
    accion = str(peticion.get("accion", "")).strip().lower()
    ahora = recordatorios.ahora_local()

    if accion == "crear":
        try:
            cuando = recordatorios.calcular_cuando(
                ahora,
                en_minutos=peticion.get("en_minutos"),
                fecha_hora=peticion.get("fecha_hora"),
                hora=peticion.get("hora"),
                dias=peticion.get("dias"),
                dia_semana=peticion.get("dia_semana"),
            )
            creado = await asyncio.to_thread(
                recordatorios.crear,
                _directorio(),
                str(peticion.get("texto", "")),
                cuando,
                str(peticion.get("repetir") or "nunca"),
                trabajo.get("quien"),
            )
        except ValueError as e:
            return {"texto": f"No se ha apuntado: {e}."}
        repite = "" if creado["repetir"] == "nunca" else f", y luego {_repeticion(creado['repetir'])}"
        return {
            "texto": (
                f"Apuntado: te lo recuerdo {recordatorios.describir(cuando, ahora)}{repite}: "
                f"«{creado['texto']}»."
            ),
            "id": creado["id"],
        }

    if accion == "listar":
        lista = await asyncio.to_thread(recordatorios.pendientes, _directorio())
        if not lista:
            return {"texto": "No hay ningún recordatorio pendiente."}
        lineas = [
            f"- {recordatorios.describir(datetime.fromisoformat(r['cuando']), ahora)}: {r['texto']}"
            + ("" if r.get("repetir", "nunca") == "nunca" else f" ({_repeticion(r['repetir'])})")
            for r in lista[: recordatorios.TOPE_LISTA]
        ]
        resto = len(lista) - recordatorios.TOPE_LISTA
        if resto > 0:
            lineas.append(f"- y {resto} más")
        return {"texto": f"{len(lista)} recordatorio(s) pendiente(s):\n" + "\n".join(lineas)}

    if accion == "cancelar":
        try:
            quitado = await asyncio.to_thread(
                recordatorios.cancelar, _directorio(), str(peticion.get("texto") or peticion.get("id") or "")
            )
        except ValueError as e:
            return {"texto": f"No se ha quitado nada: {e}."}
        return {"texto": f"Quitado: «{quitado['texto']}»."}

    if accion == "avisar":
        vencidos = [r for r in peticion.get("recordatorios") or [] if isinstance(r, dict)]
        if not vencidos:
            return {"texto": "Nada que avisar."}
        # La llamada: el mismo timbre que los encargos. Si ya está hablando con
        # Perseo, se lo dice ahí mismo; si no contesta, queda en pendientes.
        for recordatorio in vencidos:
            await asyncio.to_thread(llamada_saliente.llamar, _aviso(recordatorio))
        return {
            "recordatorios": vencidos,
            "texto": "\n".join(_aviso(r) for r in vencidos),
            "titular": titular(vencidos),
        }

    raise ValueError(f"Acción desconocida para recordatorios: {accion!r}")


def _repeticion(repetir: str) -> str:
    return {
        "diario": "cada día",
        "laborables": "cada día laborable",
        "semanal": "cada semana",
    }.get(repetir, repetir)


def _aviso(recordatorio: dict[str, Any]) -> str:
    retraso = int(recordatorio.get("retraso_min") or 0)
    tarde = f" (llega {retraso} min tarde: el núcleo no estaba a la hora)" if retraso >= 2 else ""
    return f"Recordatorio: {recordatorio.get('texto')}{tarde}"


def titular(vencidos: list[dict[str, Any]]) -> str:
    """Lo que sale por Telegram: que hay aviso y de qué hora. **El texto no sale.**

    Es la regla de `caras/telegram.py` —titular por Telegram, detalle por
    Tailscale—: lo que el señor Persus dictó es contenido suyo y viaja por un
    tercero. El texto entero está en la cola, a un toque del enlace.
    """
    horas = sorted({datetime.fromisoformat(str(r["cuando"])).strftime("%H:%M") for r in vencidos})
    if len(vencidos) == 1:
        return f"Recordatorio de las {horas[0]}"
    return f"{len(vencidos)} recordatorios ({', '.join(horas)})"


@disparadores.registrar("recordatorios", intervalo=30)
async def _vigilar_recordatorios(ctx: disparadores.Contexto) -> None:
    """Encola el aviso de los que hayan vencido. Treinta segundos de margen como mucho."""
    ahora = recordatorios.ahora_local()
    vencidos = await asyncio.to_thread(recordatorios.vencidos, ctx.cfg.directorio_datos, ahora)
    if not vencidos:
        return
    trabajo = await ctx.encolar("recordatorios", {"accion": "avisar", "recordatorios": vencidos})
    logger.info("Encolado el trabajo %s con %d recordatorio(s).", trabajo["id"], len(vencidos))
