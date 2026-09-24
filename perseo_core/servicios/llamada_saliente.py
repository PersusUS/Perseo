"""Que Perseo llame: el mismo timbre para recordatorios y encargos.

**La tecnología ya existía.** Los subagentes (`commands/subagentes_mcp.py`)
dejaban un marcador, `.perseo-autollamada`, con el motivo dentro; Rust lo vigila
cada medio segundo (`autollamada.rs`), saca la ventana, y la app hace sonar el
timbre: si el señor Persus contesta, la llamada empieza con ese motivo; si no,
queda en llamadas pendientes. Aquí el núcleo usa exactamente eso, en vez de
inventar un segundo camino.

**Dos cosas iban mal, y se arreglan aquí o en la app:**

  · El marcador se **sobrescribía**: dos avisos casi a la vez y el primero se
    perdía. Ahora cada motivo es una línea que se añade, y Rust se lleva el
    fichero entero de una vez (lo renombra antes de leerlo).
  · Un encargo del agente `dev` que terminaba **sin nadie esperando** —pedido
    por escrito, que se deja de esperar a los 25 s, o desde el móvil— acababa
    en silencio. Ahora, si nadie ha preguntado por él en `SIN_ESPERA`, llama.

**Cómo se sabe si alguien espera.** Quien espera un trabajo pregunta por él una
y otra vez —Rust cada 250 ms, el chat escrito cada 400 ms—: cada pregunta se
apunta con `marcar_espera`. Si al terminar nadie ha preguntado en unos
segundos, es que nadie está mirando, y entonces se llama. Si alguien espera,
el resultado se lo lleva él: llamar además sería contarlo dos veces.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from ..infra.bus import Bus

logger = logging.getLogger(__name__)

#: El marcador que vigila la app. Vive en la raíz del repositorio, que es donde
#: lo busca `rutas_de_marcador` en `commands.rs` y donde lo dejan los subagentes.
NOMBRE_MARCADOR = ".perseo-autollamada"
RAIZ_REPOSITORIO = Path(__file__).resolve().parents[2]


def _raiz() -> Path:
    """Dónde se deja el marcador. `PERSEO_MARCADOR_DIR` lo aparta en las
    verificaciones: un núcleo de prueba no debe hacer sonar la app de verdad."""
    apartado = os.environ.get("PERSEO_MARCADOR_DIR", "").strip()
    return Path(apartado) if apartado else RAIZ_REPOSITORIO


#: Si nadie ha preguntado por un trabajo en estos segundos, nadie lo espera.
SIN_ESPERA = 10.0

#: Los agentes cuyo final merece una llamada cuando nadie espera. Los que
#: contestan en un segundo no: su resultado ya se lo llevó quien preguntó.
#: Un recado sí: se pide y se deja de mirar, que es la gracia de pedirlo.
AGENTES_QUE_AVISAN = ("dev", "recado")

_bloqueo = threading.Lock()
_ultima_pregunta: dict[int, float] = {}


def llamar(motivo: str, raiz: Path | None = None) -> None:
    """Deja el motivo en el marcador: la app timbra, o lo cuenta si ya habla."""
    motivo = " ".join(str(motivo or "").split())
    if not motivo:
        return
    ruta = Path(raiz or _raiz()) / NOMBRE_MARCADOR
    try:
        with _bloqueo, ruta.open("a", encoding="utf-8") as fichero:
            fichero.write(motivo + "\n")
    except OSError as e:
        logger.warning("No se pudo dejar el aviso de llamada (%s): %s", e, motivo)
        return
    logger.info("Llamada pedida: %s", motivo)


def marcar_espera(id_trabajo: int) -> None:
    """Alguien acaba de preguntar por este trabajo: lo está esperando."""
    with _bloqueo:
        _ultima_pregunta[int(id_trabajo)] = time.monotonic()
        # Sin crecer para siempre: lo de hace una hora ya no espera nadie.
        if len(_ultima_pregunta) > 500:
            corte = time.monotonic() - 3600
            for clave in [k for k, v in _ultima_pregunta.items() if v < corte]:
                del _ultima_pregunta[clave]


def alguien_espera(id_trabajo: int, ahora: float | None = None) -> bool:
    with _bloqueo:
        ultima = _ultima_pregunta.get(int(id_trabajo))
    if ultima is None:
        return False
    return (ahora if ahora is not None else time.monotonic()) - ultima < SIN_ESPERA


def motivo_de_encargo(trabajo: dict[str, Any], estado: str) -> str:
    """«El encargo #12 (arreglar el login) ha terminado: …». Una línea, sin volcar nada."""
    peticion = trabajo.get("peticion") or {}
    pedido = " ".join(str(peticion.get("texto") or "").split())[:80]
    resultado = trabajo.get("resultado")
    if estado == "fallido":
        detalle = str(trabajo.get("error") or "sin detalle")
    elif isinstance(resultado, dict):
        detalle = str(
            resultado.get("aviso") or resultado.get("titular") or resultado.get("texto") or resultado.get("resumen") or ""
        )
    else:
        detalle = str(resultado or "")
    detalle = (detalle.strip().splitlines() or [""])[0][:160]
    que = "ha fallado" if estado == "fallido" else "ha terminado"
    return f"El encargo #{trabajo.get('id')} ({pedido or 'sin descripción'}) {que}" + (
        f": {detalle}" if detalle else "."
    )


class Avisador:
    """Escucha el bus y llama cuando un encargo termina sin nadie mirando."""

    def __init__(self, bus: Bus, raiz: Path | None = None) -> None:
        self._bus = bus
        self._raiz = raiz
        #: Lo que tarda el que espera en hacer su última pregunta tras el final.
        #: Rust pregunta cada 250 ms; con dos segundos sobra.
        self.gracia = 2.0
        #: Las esperas en curso. Una tarea de asyncio sin nadie que la
        #: sostenga puede recogerla el recolector a mitad de camino.
        self._en_curso: set[asyncio.Task] = set()

    async def ejecutar(self) -> None:
        async with self._bus.suscribir() as eventos:
            async for evento in eventos:
                if evento.tipo not in ("trabajo.hecho", "trabajo.fallido"):
                    continue
                trabajo = evento.datos.get("trabajo") or {}
                if trabajo.get("agente") not in AGENTES_QUE_AVISAN or trabajo.get("id") is None:
                    continue
                resultado = trabajo.get("resultado")
                if isinstance(resultado, dict) and resultado.get("callado"):
                    # Lo pide el propio agente: una vigilancia que comprobó y
                    # no encontró nada no merece un timbre cada tres horas.
                    continue
                tarea = asyncio.create_task(self._quizas_llamar(trabajo, evento.tipo.split(".")[1]))
                self._en_curso.add(tarea)
                tarea.add_done_callback(self._en_curso.discard)

    async def _quizas_llamar(self, trabajo: dict[str, Any], estado: str) -> None:
        await asyncio.sleep(self.gracia)
        if alguien_espera(int(trabajo["id"])):
            return
        motivo = motivo_de_encargo(trabajo, estado)
        await asyncio.to_thread(llamar, motivo, self._raiz)
        # Y al móvil, por si no está delante: solo el titular, como siempre.
        self._bus.publicar("aviso.encargo", trabajo=trabajo, estado=estado)
