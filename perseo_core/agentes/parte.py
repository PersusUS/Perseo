"""Agente `parte`: el parte del día, en una sola respuesta.

Lo que el señor Persus tenía que preguntar por piezas —¿qué tengo hoy?, ¿algún
correo importante?, ¿qué me toca?— junto, y leído de donde de verdad está: el
calendario, los correos ya triados, los recordatorios, y los espejos del
tablero y de los hábitos.

**Qué dice cuando algo no está.** Si el permiso de Google caducó, el parte lo
dice en la línea de la agenda en vez de callarse la agenda: un parte que omite
una sección sin avisar parece un día vacío, que es justo lo que no es.

**Por Telegram, solo recuentos.** Es la regla de `caras/telegram.py`: el
titular dice cuántas citas y cuántos correos, y el detalle —con quién, de qué—
está a un toque del enlace, por el tailnet.

**El parte de la mañana es opcional y viene apagado.** Con `PERSEO_PARTE_HORA`
(`08:30`, por ejemplo) un disparador lo encola una vez al día a esa hora, y si
el núcleo arrancó después, en las tres horas siguientes. Sin la variable, el
disparador se retira y el parte solo sale cuando se pide.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any

from ..infra import disparadores
from ..infra.configuracion import Configuracion
from ..infra.router import registrar
from ..servicios import correo_lectura, google_api, habitos, recordatorios, tareas
from . import agenda

logger = logging.getLogger(__name__)

#: Cuántas líneas de cada cosa. Un parte se escucha de pie, con el café.
TOPE_CITAS = 8
TOPE_CORREOS = 5
TOPE_LINEAS_ESPEJO = 12

#: Hasta cuándo, después de la hora, se da el parte de la mañana si el núcleo
#: no estaba encendido a la hora. Más tarde ya no es «de la mañana».
VENTANA_PARTE = timedelta(hours=3)

_cfg: Configuracion | None = None


def iniciar(cfg: Configuracion) -> None:
    global _cfg
    _cfg = cfg


async def _agenda_de_hoy(ahora: datetime) -> tuple[str, int | None]:
    try:
        calendario = agenda.calendario()
    except RuntimeError:
        calendario = None
    if calendario is None:
        return "Agenda: no hay calendario configurado.", None
    fin_del_dia = datetime.combine(ahora.date(), time(23, 59), tzinfo=ahora.tzinfo)
    try:
        eventos = await calendario.proximos(max(fin_del_dia - ahora, timedelta(minutes=1)))
    except google_api.SinCredenciales as e:
        return f"Agenda: no se puede mirar — {e}", None
    except Exception as e:  # noqa: BLE001 — el parte sale igual, con la sección dicha
        logger.warning("El parte no pudo leer la agenda: %s", e)
        return "Agenda: Google no ha contestado; no se sabe qué hay hoy.", None
    if not eventos:
        return "Agenda: nada más hoy.", 0
    lineas = [f"- {_hora_local(e.inicio, ahora)} {e.titulo}" for e in eventos[:TOPE_CITAS]]
    return f"Agenda de hoy ({len(eventos)}):\n" + "\n".join(lineas), len(eventos)


def _hora_local(inicio: str, ahora: datetime) -> str:
    """La hora de un evento en la hora de aquí: Google puede darla en UTC («…Z»)."""
    try:
        return datetime.fromisoformat(str(inicio).replace("Z", "+00:00")).astimezone(ahora.tzinfo).strftime("%H:%M")
    except ValueError:
        return str(inicio)[11:16]


def _correos(ruta_db: Path) -> tuple[str, int]:
    try:
        correos = correo_lectura.cargar_correos(ruta_db)
    except Exception as e:  # noqa: BLE001 — una base sin correos aún no es un fallo
        logger.debug("El parte no pudo leer los correos: %s", e)
        return "Correo: no hay nada triado.", 0
    piden = [c for c in correos if c.get("clase") == "requiere_accion" and not c.get("hecho")]
    if not piden:
        return "Correo: ninguno pide nada.", 0
    lineas = [f"- {correo_lectura.linea(c)}" for c in piden[:TOPE_CORREOS]]
    return f"Correos que piden algo ({len(piden)}):\n" + "\n".join(lineas), len(piden)


def _recordatorios_de_hoy(directorio: Path, ahora: datetime) -> tuple[str, int]:
    hoy = [
        r
        for r in recordatorios.pendientes(directorio)
        if datetime.fromisoformat(r["cuando"]).astimezone(ahora.tzinfo).date() == ahora.date()
    ]
    if not hoy:
        return "Recordatorios: ninguno más hoy.", 0
    lineas = [f"- {datetime.fromisoformat(r['cuando']).astimezone(ahora.tzinfo):%H:%M} {r['texto']}" for r in hoy]
    return f"Recordatorios de hoy ({len(hoy)}):\n" + "\n".join(lineas), len(hoy)


def _espejo(titulo: str, texto: str) -> str:
    """Las primeras líneas de lo que la app ya redactó: el tablero o los hábitos."""
    lineas = [linea for linea in texto.splitlines() if linea.strip()]
    recorte = lineas[:TOPE_LINEAS_ESPEJO]
    if len(lineas) > TOPE_LINEAS_ESPEJO:
        recorte.append("…")
    return f"{titulo}:\n" + "\n".join(recorte)


async def componer(cfg: Configuracion, ahora: datetime) -> dict[str, Any]:
    directorio = Path(cfg.directorio_datos)
    agenda_texto, citas = await _agenda_de_hoy(ahora)
    correo_texto, correos = await asyncio.to_thread(_correos, Path(cfg.ruta_db))
    avisos_texto, avisos = await asyncio.to_thread(_recordatorios_de_hoy, directorio, ahora)
    tablero = await asyncio.to_thread(tareas.resumen, directorio)
    seguimiento = await asyncio.to_thread(habitos.resumen, directorio)

    secciones = [
        agenda_texto,
        correo_texto,
        avisos_texto,
        _espejo("Tareas", tablero),
        _espejo("Hábitos", seguimiento),
    ]
    return {
        "texto": f"Parte del {ahora:%d/%m}, a las {ahora:%H:%M}.\n\n" + "\n\n".join(secciones),
        "titular": titular(citas, correos, avisos),
    }


def titular(citas: int | None, correos: int, avisos: int) -> str:
    """Solo recuentos: el detalle va por el tailnet."""
    trozos = ["agenda sin mirar" if citas is None else f"{citas} cita(s)"]
    trozos.append(f"{correos} correo(s) que piden algo")
    if avisos:
        trozos.append(f"{avisos} recordatorio(s)")
    return "Parte de hoy: " + " · ".join(trozos)


@registrar("parte")
async def _parte(trabajo: dict[str, Any]) -> dict[str, Any]:
    if _cfg is None:
        raise RuntimeError("El agente parte no está iniciado; falta iniciar(cfg).")
    return await componer(_cfg, recordatorios.ahora_local())


# --------------------------------------------------------------------------- #
# El parte de la mañana
# --------------------------------------------------------------------------- #

_dados: disparadores.Vistos | None = None


def toca(ahora: datetime, hora: str) -> bool:
    """Si ahora cae entre la hora del parte y las tres horas siguientes."""
    horas, minutos = (int(x) for x in hora.split(":")[:2])
    desde = ahora.replace(hour=horas, minute=minutos, second=0, microsecond=0)
    return desde <= ahora < desde + VENTANA_PARTE


@disparadores.registrar("parte", intervalo=60)
async def _vigilar_parte(ctx: disparadores.Contexto) -> None:
    global _dados
    hora = (ctx.cfg.parte_hora or "").strip()
    if not hora:
        raise disparadores.Retirarse("sin PERSEO_PARTE_HORA, el parte solo sale cuando se pide")
    try:
        toca(recordatorios.ahora_local(), hora)
    except ValueError:
        raise disparadores.Retirarse(f"PERSEO_PARTE_HORA={hora!r} no es una hora (HH:MM)") from None

    if _dados is None:
        _dados = disparadores.Vistos(ruta=Path(ctx.cfg.directorio_datos) / "parte_dados.json").cargar()
    ahora = recordatorios.ahora_local()
    marca = f"{ahora:%Y-%m-%d}"
    if not toca(ahora, hora) or not _dados.sin_ver([marca]):
        return
    _dados.anotar([marca])
    trabajo = await ctx.encolar("parte", {"accion": "dar"})
    logger.info("Encolado el parte del día (trabajo %s).", trabajo["id"])
