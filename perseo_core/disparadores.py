"""Disparadores: lo que hace que Perseo empiece algo sin que se lo pidas.

Hasta aquí todo trabajo nacía de una petición —la web, la voz, la API—. Un
disparador es lo contrario: mira algo cada tanto y, si hay novedad, encola. El
bus ya estaba preparado para esto desde la Fase A; lo que faltaba era quien
publicara.

Tres decisiones que conviene no deshacer:

1. **Un disparador que falla no tumba el núcleo.** Cada vuelta va dentro de su
   `try`. Que Gmail devuelva un 500 o que se caiga el wifi tiene que costar una
   línea en el registro y la siguiente vuelta, no el proceso entero.
2. **Un disparador sin configurar se retira solo**, como hace Telegram sin token.
   No es un error: es una capacidad que hoy no está.
3. **Encolan, no ejecutan.** El disparador no clasifica ni contesta: mete un
   trabajo en la cola y se va. Así lo que descubre sobrevive a un reinicio, se ve
   en la web y pasa por el mismo camino —con sus confirmaciones— que lo que pides
   tú. Un disparador que hiciera el trabajo por su cuenta sería un segundo
   sistema con sus propias reglas.


"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from . import almacen
from .bus import Bus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Contexto:
    """Lo que recibe un disparador en cada vuelta.

    Es un objeto y no dos argumentos sueltos para que añadir algo más adelante
    —el triaje, un cliente compartido— no obligue a tocar todos los disparadores.
    """

    cfg: almacen.Configuracion
    bus: Bus

    async def encolar(self, agente: str, peticion: dict[str, Any]) -> dict[str, Any]:
        """Mete un trabajo en la cola con origen `disparador` y lo anuncia.

        El origen importa: es lo que permite distinguir en la web lo que has
        pedido tú de lo que ha salido solo.
        """
        trabajo = await asyncio.to_thread(almacen.encolar, agente, peticion, "disparador")
        self.bus.publicar("trabajo.encolado", trabajo=trabajo)
        return trabajo


#: Qué hace un disparador cuando le toca. No devuelve nada: lo que produce lo
#: deja en la cola o en el bus.
Revision = Callable[[Contexto], Awaitable[None]]


@dataclass(frozen=True)
class Disparador:
    nombre: str
    intervalo: float
    revisar: Revision


REGISTRO: dict[str, Disparador] = {}


def registrar(nombre: str, intervalo: float) -> Callable[[Revision], Revision]:
    """Decorador que da de alta un disparador. `intervalo` en segundos."""

    def decorador(funcion: Revision) -> Revision:
        if nombre in REGISTRO:
            raise ValueError(f"El disparador {nombre!r} ya está registrado.")
        REGISTRO[nombre] = Disparador(nombre=nombre, intervalo=intervalo, revisar=funcion)
        return funcion

    return decorador


@dataclass
class Vistos:
    """Qué ha atendido ya un disparador, para no repetirse.

    Vive en un fichero del directorio de datos y no en memoria: si viviera en
    memoria, reiniciar el núcleo volvería a triar el buzón entero y a avisar de
    eventos que ya te había avisado. Es, para los disparadores, lo que la cola en
    SQLite es para los trabajos.
    """

    ruta: Path
    ids: set[str] = field(default_factory=set)

    #: Cuántos identificadores se recuerdan. Se guardan los últimos: un buzón
    #: activo genera miles al año y el fichero no debe crecer sin fin.
    tope: int = 2000

    #: Verdadero mientras no exista el fichero. La primera vuelta se comporta
    #: distinto por eso, y quien la usa necesita saberlo.
    estrenando: bool = True

    def cargar(self) -> "Vistos":
        if self.ruta.exists():
            self.estrenando = False
            try:
                crudo = json.loads(self.ruta.read_text(encoding="utf-8") or "[]")
                self.ids = {str(i) for i in crudo}
            except (OSError, json.JSONDecodeError) as e:
                # Perder la marca de agua avisa dos veces; tumbar el núcleo por un
                # fichero corrupto es peor.
                logger.warning("No se pudo leer %s (%s); se empieza de cero.", self.ruta, e)
        return self

    def sin_ver(self, ids: list[str]) -> list[str]:
        return [i for i in ids if i and i not in self.ids]

    def anotar(self, ids: list[str]) -> None:
        self.ids.update(i for i in ids if i)
        self.estrenando = False
        if len(self.ids) > self.tope:
            self.ids = set(sorted(self.ids)[-self.tope :])
        try:
            self.ruta.write_text(
                json.dumps(sorted(self.ids), ensure_ascii=False), encoding="utf-8"
            )
        except OSError as e:
            logger.warning("No se pudo guardar %s (%s).", self.ruta, e)


class Retirarse(Exception):
    """La lanza un disparador cuando no tiene nada que vigilar.

    No es un fallo: es "esto no está configurado". El bucle se para sin ruido en
    vez de reintentar cada minuto contra algo que no existe.
    """


class Planificador:
    """Mantiene en marcha los disparadores activos."""

    def __init__(self, cfg: almacen.Configuracion, bus: Bus) -> None:
        self._contexto = Contexto(cfg=cfg, bus=bus)
        self._activos = tuple(n for n in cfg.disparadores if n in REGISTRO)
        self._desconocidos = tuple(n for n in cfg.disparadores if n not in REGISTRO)
        self._parar = asyncio.Event()

    def detener(self) -> None:
        self._parar.set()

    async def ejecutar(self) -> None:
        for nombre in self._desconocidos:
            logger.warning("Disparador desconocido en la configuración: %r.", nombre)
        if not self._activos:
            logger.info("Sin disparadores activos.")
            return

        logger.info("Disparadores en marcha: %s.", ", ".join(self._activos))
        await asyncio.gather(*(self._bucle(REGISTRO[n]) for n in self._activos))

    def _intervalo(self, disparador: Disparador) -> float:
        """Cada cuánto le toca. La configuración manda sobre el valor del registro."""
        return float(self._contexto.cfg.intervalos.get(disparador.nombre, disparador.intervalo))

    async def _bucle(self, disparador: Disparador) -> None:
        espera = self._intervalo(disparador)
        while not self._parar.is_set():
            try:
                await disparador.revisar(self._contexto)
            except Retirarse as motivo:
                logger.info("Disparador %s retirado: %s", disparador.nombre, motivo)
                return
            except asyncio.CancelledError:
                raise
            except Exception:
                # Una vuelta mala no vale una caída: se anota y se vuelve a
                # intentar en el siguiente turno.
                logger.exception("El disparador %s falló en esta vuelta.", disparador.nombre)

            try:
                await asyncio.wait_for(self._parar.wait(), timeout=espera)
            except asyncio.TimeoutError:
                pass
        logger.debug("Disparador %s detenido.", disparador.nombre)
