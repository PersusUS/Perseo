"""Agente `vigilancias`: apuntar, listar y quitar lo que Perseo vigila en la web.

Lo que de verdad mira la web es el agente `recado`, en modo `vigilar`: este solo
lleva la lista (`servicios/vigilancias.py`) y el disparador que, cuando toca,
encola la comprobación. Así una vigilancia es exactamente un recado —el mismo
navegador, la misma bóveda, la misma parada antes de pagar— y no un segundo
sistema con sus propias reglas.

**Una sola herramienta con `que`, y no tres.** La cara de voz traduce las
herramientas en Rust con una tabla que fija la `accion` del trabajo
(`DIRECTAS` en `nucleo.rs`), y ese fichero vive pegado a su techo de 900
líneas. `que` —crear, listar o cancelar— viaja dentro de la petición y lo
decide este agente, que es donde se decide.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from ..infra import disparadores
from ..infra.configuracion import Configuracion
from ..infra.router import registrar
from ..servicios import llamada_saliente, recordatorios, vigilancias

logger = logging.getLogger(__name__)

_cfg: Configuracion | None = None


def iniciar(cfg: Configuracion) -> None:
    global _cfg
    _cfg = cfg


def _directorio() -> str:
    if _cfg is None:
        raise RuntimeError("El agente vigilancias no está iniciado; falta iniciar(cfg).")
    return str(_cfg.directorio_datos)


def _cuando(iso: str, ahora: datetime) -> str:
    try:
        return recordatorios.describir(datetime.fromisoformat(iso), ahora)
    except ValueError:
        return iso


def _linea(v: dict[str, Any], ahora: datetime) -> str:
    ultima = v.get("ultima") or {}
    visto = f" Última vez: {ultima.get('detalle')}." if ultima.get("detalle") else ""
    return (
        f"- «{v['objetivo']}» — avisa cuando: {v['condicion']} "
        f"({'lo hace' if v.get('al_cumplirse') == 'hacer' else 'avisa'}; cada {v['cada_horas']:g} h, "
        f"hasta {_cuando(v['hasta'], ahora)}; próxima {_cuando(v['proxima'], ahora)}).{visto}"
    )


@registrar("vigilancias")
async def _vigilancias(trabajo: dict[str, Any]) -> dict[str, Any]:
    peticion = trabajo.get("peticion") or {}
    accion = str(peticion.get("accion", "")).strip().lower()
    ahora = recordatorios.ahora_local()

    if accion == "caducadas":
        lista = [v for v in peticion.get("vigilancias") or [] if isinstance(v, dict)]
        for v in lista:
            await asyncio.to_thread(
                llamada_saliente.llamar,
                f"La vigilancia «{v.get('objetivo')}» ha caducado sin que pasara: {v.get('condicion')}.",
            )
        return {
            "texto": "\n".join(f"Caducada sin cumplirse: «{v.get('objetivo')}»" for v in lista) or "Nada.",
            # Por Telegram, solo cuántas: lo vigilado es contenido suyo.
            "titular": f"{len(lista)} vigilancia(s) caducada(s) sin cumplirse" if lista else None,
        }

    que = str(peticion.get("que") or accion or "").strip().lower()
    if que == "crear":
        try:
            v = await asyncio.to_thread(
                vigilancias.crear,
                _directorio(),
                str(peticion.get("objetivo") or peticion.get("texto") or ""),
                str(peticion.get("condicion") or ""),
                ahora,
                peticion.get("cada_horas"),
                peticion.get("dias"),
                str(peticion.get("al_cumplirse") or "avisar"),
                trabajo.get("quien"),
            )
        except ValueError as e:
            return {"texto": f"No se ha apuntado: {e}."}
        hace = "lo haré, y lo que se pague o se envíe esperará tu sí" if v["al_cumplirse"] == "hacer" else "te aviso"
        return {
            "texto": (
                f"Vigilando «{v['objetivo']}»: miro ahora y luego cada {v['cada_horas']:g} h hasta "
                f"{_cuando(v['hasta'], ahora)}. Cuando {v['condicion']}, {hace}."
            ),
            "id": v["id"],
        }

    if que == "listar":
        lista = await asyncio.to_thread(vigilancias.activas, _directorio())
        if not lista:
            return {"texto": "No hay ninguna vigilancia activa."}
        return {"texto": f"{len(lista)} vigilancia(s) activa(s):\n" + "\n".join(_linea(v, ahora) for v in lista)}

    if que == "cancelar":
        try:
            v = await asyncio.to_thread(
                vigilancias.cancelar, _directorio(), str(peticion.get("objetivo") or peticion.get("texto") or "")
            )
        except ValueError as e:
            return {"texto": f"No se ha quitado nada: {e}."}
        return {"texto": f"Dejo de vigilar «{v['objetivo']}»."}

    raise ValueError(f"Qué hacer con las vigilancias: crear, listar o cancelar (llegó {que!r}).")


@disparadores.registrar("vigilancias", intervalo=60)
async def _vigilar(ctx: disparadores.Contexto) -> None:
    """Encola la comprobación de las que tocan, y el aviso de las que caducan."""
    tocan, caducadas = await asyncio.to_thread(
        vigilancias.para_comprobar, ctx.cfg.directorio_datos, recordatorios.ahora_local()
    )
    for v in tocan:
        trabajo = await ctx.encolar(
            "recado", {"texto": v["objetivo"], "accion": "vigilar", "vigilancia": v["id"]}
        )
        logger.info("Vigilancia %s: comprobación encolada (trabajo %s).", v["id"], trabajo["id"])
    if caducadas:
        await ctx.encolar("vigilancias", {"accion": "caducadas", "vigilancias": caducadas})
