"""El hilo principal: una sola conversación para todas las caras.

Instinct es un único hilo: se le escribe desde donde sea y se acuerda de todo.
Perseo tenía conversaciones sueltas —una por sesión del panel— y la voz, que no
dejaba nada en ninguna: lo hablado se guardaba en el vault y el chat escrito no
lo veía. Aquí vive el hilo que comparten todas:

- el panel y el móvil lo ven como una conversación más de su lista, «Hilo
  principal», que sube arriba porque es la que más se mueve;
- Telegram, WhatsApp y el teléfono escriben y leen en él;
- al colgar una llamada, lo hablado entra en él (`anotar_voz`), así que por
  escrito se puede seguir donde se dejó hablando;
- y la voz lo lee al empezar (`reciente`, por la herramienta `hilo_reciente`).

**Un turno cada vez, y quien llega espera.** La conversación tiene su semáforo
(`almacen.marcar_turno_chat`): el panel contesta 409 si otra pantalla está a
mitad de turno. Un canal de fuera no puede enseñar un 409 a nadie, así que
espera a que se libere, con plazo.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from ..infra import almacen

logger = logging.getLogger(__name__)

NOMBRE_FICHERO = "hilo_principal.json"
TITULO = "Hilo principal"

#: Cuánto de una llamada entra en el hilo. Una llamada larga son cientos de
#: frases; el chat lee los últimos veinticuatro turnos, así que más no se vería.
TOPE_TURNOS_VOZ = 24

#: Cuánto se espera a que el turno de otra cara acabe antes de rendirse.
ESPERA_TURNO = 90.0


def _ruta(directorio_datos: Path | str) -> Path:
    return Path(directorio_datos) / NOMBRE_FICHERO


def principal(directorio_datos: Path | str) -> int:
    """El id de la conversación principal, creándola si no existe o se borró."""
    ruta = _ruta(directorio_datos)
    try:
        guardado = int(json.loads(ruta.read_text(encoding="utf-8")).get("sesion"))
    except (FileNotFoundError, ValueError, TypeError, json.JSONDecodeError, AttributeError):
        guardado = 0
    if guardado and almacen.obtener_sesion_chat(guardado) is not None:
        return guardado
    creada = almacen.crear_sesion_chat(TITULO)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps({"sesion": creada["id"]}), encoding="utf-8")
    logger.info("Hilo principal creado: conversación %s.", creada["id"])
    return int(creada["id"])


async def hablar(directorio_datos: Path | str, texto: str, canal: str, espera: float = 240.0) -> str:
    """Un turno en el hilo principal desde un canal de fuera; devuelve la respuesta.

    `canal` («telegram», «whatsapp», «teléfono») va delante del texto entre
    corchetes: el modelo sabe así por dónde le hablan —y que por teléfono no
    puede enseñar una lista larga—, y el panel lo enseña igual.
    """
    sesion = await asyncio.to_thread(principal, directorio_datos)
    limite = asyncio.get_running_loop().time() + ESPERA_TURNO
    while True:
        try:
            await asyncio.to_thread(almacen.marcar_turno_chat, sesion, "ocupado")
            break
        except ValueError:
            if asyncio.get_running_loop().time() >= limite:
                return "Estoy terminando otra cosa en este mismo hilo; dímelo otra vez en un momento."
            await asyncio.sleep(1.0)
    texto = f"[{canal}] {texto.strip()}"
    try:
        await asyncio.to_thread(almacen.anadir_mensaje_chat, sesion, "usuario", texto)
        id_perseo = await asyncio.to_thread(almacen.anadir_mensaje_chat, sesion, "perseo", "", "escribiendo")
        await asyncio.to_thread(
            almacen.encolar, "chat", {"sesion": sesion, "mensaje": id_perseo, "texto": texto}, "texto"
        )
    except Exception:
        await asyncio.to_thread(almacen.marcar_turno_chat, sesion, "libre")
        raise
    return await esperar_respuesta(sesion, id_perseo, espera)


async def esperar_respuesta(sesion: int, id_mensaje: int, espera: float) -> str:
    limite = asyncio.get_running_loop().time() + espera
    while asyncio.get_running_loop().time() < limite:
        mensajes = await asyncio.to_thread(almacen.mensajes_chat, sesion, 10)
        mio = next((m for m in mensajes if m["id"] == id_mensaje), None)
        if mio is not None and mio["estado"] in ("hecho", "fallido"):
            return mio["texto"].strip() or "Hecho."
        await asyncio.sleep(0.5)
    return "Sigo en ello; te lo cuento en cuanto acabe."


def anotar_voz(directorio_datos: Path | str, mensajes: list[dict[str, Any]]) -> int:
    """Mete en el hilo lo hablado en una llamada. Devuelve cuántas frases entraron.

    Entran como mensajes normales —«usuario» lo que dijo él, «perseo» lo que
    contestó—, marcados `[voz]`, para que el chat los lea como parte de la misma
    conversación y no como una nota aparte.
    """
    utiles = [
        m for m in mensajes
        if isinstance(m, dict) and str(m.get("tipo")) in ("user", "ai") and str(m.get("texto", "")).strip()
    ][-TOPE_TURNOS_VOZ:]
    if not utiles:
        return 0
    sesion = principal(directorio_datos)
    for m in utiles:
        rol = "usuario" if m["tipo"] == "user" else "perseo"
        almacen.anadir_mensaje_chat(sesion, rol, f"[voz] {str(m['texto']).strip()}")
    return len(utiles)


def reciente(directorio_datos: Path | str, tope: int = 12) -> str:
    """Lo último del hilo, en texto, para que la voz sepa de qué se habló por escrito."""
    sesion = principal(directorio_datos)
    mensajes = [m for m in almacen.mensajes_chat(sesion, tope) if m["texto"].strip() and m["estado"] != "fallido"]
    if not mensajes:
        return "El hilo principal está vacío: no habéis hablado de nada todavía."
    lineas = [
        f"- {m['momento'][:16].replace('T', ' ')} {'Él' if m['rol'] == 'usuario' else 'Perseo'}: "
        f"{' '.join(m['texto'].split())[:300]}"
        for m in mensajes
    ]
    return "Lo último del hilo principal (de lo más viejo a lo más nuevo):\n" + "\n".join(lineas)
