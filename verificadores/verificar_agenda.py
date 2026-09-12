"""Verificación del agente `agenda`, sin Google Calendar.

El calendario es un fichero JSON al que este script escribe eventos con horas
relativas a ahora. Lo que se comprueba es lo que no depende del proveedor: que se
avisa de lo que viene y no de lo que ya pasó, que se avisa **una sola vez**, y
que por el canal sale la hora pero no el título.

    python verificadores/verificar_agenda.py
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perseo_core.agentes import agenda  # noqa: E402
from perseo_core.infra import disparadores  # noqa: E402
from verificadores.arnes_pruebas import Nucleo, comprobar, resumir  # noqa: E402
from perseo_core.infra.bus import Bus  # noqa: E402
from perseo_core.infra.configuracion import cargar_configuracion  # noqa: E402

INTERVALO = "2"

#: El titulo es la parte que **no** puede salir por Telegram.
TITULO_SECRETO = "Revision medica con el doctor Ramirez"


def dentro_de(minutos: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=minutos)).isoformat()


def escribir(ruta: Path, eventos: list[dict[str, Any]]) -> None:
    ruta.write_text(json.dumps(eventos, ensure_ascii=False, indent=2), encoding="utf-8")


def trabajos_de_agenda(nucleo: Nucleo) -> list[dict[str, Any]]:
    _, lista = nucleo.pedir("/trabajos?limite=100", nucleo.token)
    return [t for t in (lista.get("trabajos") or []) if t.get("agente") == "agenda"]


def comprobar_en_proceso(ruta: Path) -> None:
    print("--- sobre un calendario de fichero ---\n")

    escribir(
        ruta,
        [
            {"id": "pasado", "titulo": "Lo que ya empezo", "inicio": dentro_de(-30)},
            {"id": "pronto", "titulo": TITULO_SECRETO, "inicio": dentro_de(20)},
            {"id": "luego", "titulo": "Cena", "inicio": dentro_de(50)},
            {"id": "lejos", "titulo": "La semana que viene", "inicio": dentro_de(60 * 24 * 7)},
            {"id": "roto", "titulo": "Con la fecha mal", "inicio": "el martes"},
        ],
    )
    calendario = agenda.CalendarioFalso(ruta)
    proximos = asyncio.run(calendario.proximos(timedelta(minutes=60)))
    ids = [e.id for e in proximos]

    comprobar("Solo entra lo que cae dentro del horizonte", ids == ["pronto", "luego"], str(ids))
    comprobar("Lo que ya empezo no se avisa", "pasado" not in ids)
    comprobar("Una fecha ilegible no rompe nada", "roto" not in ids)

    # La regla del canal, otra vez: por Telegram, cuantos y a que hora.
    texto = agenda.titular(proximos) or ""
    comprobar("El titular dice cuantos son", texto.startswith("2 eventos"), texto)
    comprobar("Y a que hora empieza el primero", ":" in texto, texto)
    comprobar("Pero NO el titulo del evento", TITULO_SECRETO not in texto, texto)
    comprobar("Sin eventos no hay titular", agenda.titular([]) is None)

    # Sin calendario configurado, el disparador se retira solo.
    entorno_datos = tempfile.mkdtemp(prefix="perseo_agenda_")
    import os

    previo = dict(os.environ)
    os.environ["PERSEO_CORE_DATOS"] = entorno_datos
    os.environ.pop("PERSEO_AGENDA", None)
    cfg = cargar_configuracion()
    os.environ.clear()
    os.environ.update(previo)

    async def sin_calendario() -> str:
        ctx = disparadores.Contexto(cfg=cfg, bus=Bus())
        try:
            await agenda._vigilar_calendario(ctx)
        except disparadores.Retirarse as motivo:
            return str(motivo)
        return ""

    comprobar("Sin calendario configurado el disparador se retira", "calendario" in asyncio.run(sin_calendario()))
    shutil.rmtree(entorno_datos, ignore_errors=True)
    print()


def comprobar_de_punta_a_punta(ruta: Path) -> None:
    escribir(ruta, [{"id": "cita-1", "titulo": TITULO_SECRETO, "inicio": dentro_de(20)}])

    nucleo = Nucleo(
        {
            "PERSEO_AGENDA": "falso",
            "PERSEO_AGENDA_FALSA": str(ruta),
            "PERSEO_AGENDA_INTERVALO": INTERVALO,
            "PERSEO_AGENDA_ANTELACION": "60",
            "PERSEO_DISPARADORES": "agenda",
        }
    )
    nucleo.arrancar()

    limite = time.monotonic() + 20
    trabajo: dict[str, Any] | None = None
    while time.monotonic() < limite and trabajo is None:
        encontrados = trabajos_de_agenda(nucleo)
        trabajo = encontrados[0] if encontrados else None
        if trabajo is None:
            time.sleep(0.5)

    comprobar("Un evento cercano encola un aviso", trabajo is not None)
    if trabajo is None:
        nucleo.volcar()
        nucleo.limpiar()
        return

    comprobar("Con origen de disparador", trabajo.get("origen") == "disparador", str(trabajo.get("origen")))
    hecho = nucleo.esperar_estado(int(trabajo["id"]), ("hecho", "fallido"), intentos=60)
    comprobar(
        "El aviso se prepara bien",
        hecho.get("estado") == "hecho",
        f"estado={hecho.get('estado')} error={hecho.get('error')}",
    )
    resultado = hecho.get("resultado") or {}
    comprobar("Trae titular para el canal", bool(resultado.get("titular")), str(resultado.get("titular")))
    comprobar(
        "Y el titular no lleva el titulo del evento",
        TITULO_SECRETO not in str(resultado.get("titular")),
        str(resultado.get("titular")),
    )
    comprobar(
        "El detalle si queda en la cola",
        any(e.get("titulo") == TITULO_SECRETO for e in resultado.get("eventos") or []),
    )

    # No repetirse: el mismo evento no vuelve a avisar en la vuelta siguiente.
    time.sleep(float(INTERVALO) * 3)
    comprobar(
        "No se avisa dos veces del mismo evento",
        len(trabajos_de_agenda(nucleo)) == 1,
        f"{len(trabajos_de_agenda(nucleo))} trabajo(s)",
    )

    # Y sobrevive a un reinicio: la marca de agua esta en disco.
    nucleo.reiniciar()
    time.sleep(float(INTERVALO) * 3)
    comprobar(
        "Reiniciar tampoco repite el aviso",
        len(trabajos_de_agenda(nucleo)) == 1,
        f"{len(trabajos_de_agenda(nucleo))} trabajo(s)",
    )

    nucleo.limpiar()


def main() -> None:
    carpeta = Path(tempfile.mkdtemp(prefix="perseo_cal_"))
    try:
        comprobar_en_proceso(carpeta / "agenda.json")
        comprobar_de_punta_a_punta(carpeta / "agenda.json")
    finally:
        shutil.rmtree(carpeta, ignore_errors=True)
    resumir()


if __name__ == "__main__":
    main()
