"""Verificación de la pantalla de estado: `GET /estado`.

Lo que se comprueba no es que el panel sea bonito, es que **clasifique bien**:

- lo que no está configurado sale como `apagado` y no como fallo,
- una imitación —el buzón falso, el motor de `dev` simulado— no pasa por verde,
- lo que está roto trae al lado qué hacer,
- y ningún sondeo puede tumbar la respuesta.

Es la diferencia entre un panel que se mira todos los días y uno que está
siempre en rojo y se deja de leer el segundo día.

Corre sin Ollama, sin Obsidian y sin credenciales: el núcleo se levanta sobre un
directorio de datos temporal y las piezas que sondean fuera pueden salir en rojo
sin que eso sea un fallo de esta verificación. Ejecutar desde la raíz:

    python verificadores/verificar_estado.py
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verificadores.arnes_pruebas import Nucleo, comprobar, resumir  # noqa: E402

ESTADOS = {"ok", "aviso", "malo", "apagado"}


def pieza(panel: dict[str, Any], id_pieza: str) -> dict[str, Any]:
    for p in panel.get("piezas", []):
        if p.get("id") == id_pieza:
            return p
    return {}


def sin_token(nucleo: Nucleo, ruta: str) -> int:
    """El código que devuelve una ruta pedida como la pide el sistema operativo."""
    try:
        with urllib.request.urlopen(nucleo.base + ruta, timeout=10) as respuesta:
            return respuesta.status
    except urllib.error.HTTPError as e:
        return e.code


#: Lo que se vacía para que la verificación mida lo que cree que mide. El hijo
#: hereda el entorno de quien lanza esto, y en la máquina donde Perseo funciona
#: de verdad estas variables están puestas: sin esto, "sin configurar" sería
#: "configurado como lo tenga hoy quien ejecuta el script".
PELADO = {
    "PERSEO_CORREO": "",
    "PERSEO_AGENDA": "",
    "PERSEO_TELEGRAM_TOKEN": "",
    "PERSEO_TELEGRAM_CHAT": "",
    "PERSEO_MODELO_SUPLENTE": "",
    "GEMINI_API_KEY": "",
    "PERSEO_VAULT": "",
    "PERSEO_DEV_MOTOR": "",
    "PERSEO_WEB": "",
}


def comprobar_sin_configurar() -> None:
    """Un núcleo pelado: casi todo tiene que salir apagado, no roto."""
    nucleo = Nucleo(entorno_extra=dict(PELADO))
    nucleo.arrancar()
    try:
        codigo, _ = nucleo.pedir("/estado")
        comprobar("Sin token, /estado devuelve 401", codigo == 401, f"HTTP {codigo}")

        codigo, panel = nucleo.pedir("/estado", nucleo.token)
        comprobar("Con token, /estado responde", codigo == 200, f"HTTP {codigo}")

        faltan = [
            c
            for c in ("piezas", "trabajos", "agentes", "disparadores", "cuota", "encendido_segundos")
            if c not in panel
        ]
        comprobar("El panel trae todo lo que pinta la pantalla", not faltan, f"faltan: {faltan}")

        raros = [p for p in panel["piezas"] if p.get("estado") not in ESTADOS]
        comprobar("Ninguna pieza trae un estado inventado", not raros, str(raros))

        incompletas = [
            p["id"] for p in panel["piezas"] if not p.get("nombre") or not p.get("detalle")
        ]
        comprobar("Toda pieza se explica", not incompletas, str(incompletas))

        # Lo que de verdad hace útil el panel: distinguir "no está puesto" de
        # "está roto". Sin esto, un núcleo recién arrancado sale entero en rojo.
        for id_pieza in ("google", "telegram", "correo", "agenda", "suplente"):
            actual = pieza(panel, id_pieza)
            comprobar(
                f"Sin configurar, {id_pieza} sale apagado y no roto",
                actual.get("estado") == "apagado",
                f"{actual.get('estado')}: {actual.get('detalle')}",
            )
            comprobar(
                f"{id_pieza} dice cómo encenderlo",
                bool(actual.get("arreglo")),
                actual.get("arreglo", ""),
            )

        comprobar(
            "Las confirmaciones vienen puestas",
            pieza(panel, "confianza").get("estado") == "ok",
            str(pieza(panel, "confianza").get("detalle")),
        )

        # La ola y el avatar los pide el navegador en la pantalla que pide el
        # token, o sea antes de que haya cookie. Con 401 se verían dos huecos.
        for ruta in ("/hokusai-bg.png", "/perseo-avatar.jpg"):
            comprobar(f"{ruta} se sirve sin token", sin_token(nucleo, ruta) == 200)

        # Y el panel refleja la cola de verdad, no una copia suya.
        nucleo.pedir("/trabajos", nucleo.token, "POST", {"agente": "eco", "peticion": {"texto": "x"}})
        _, panel = nucleo.pedir("/estado", nucleo.token)
        comprobar(
            "El panel cuenta los trabajos de la cola",
            sum(panel["trabajos"].values()) >= 1,
            str(panel["trabajos"]),
        )
    finally:
        nucleo.limpiar()


def comprobar_con_imitaciones() -> None:
    """Verificar con imitaciones está bien; creérselas, no.

    Un buzón de mentira y un motor de `dev` simulado tienen que salir en ámbar:
    el sistema funciona, pero no está haciendo lo que parece que hace.
    """
    nucleo = Nucleo(
        entorno_extra={
            **PELADO,
            "PERSEO_CORREO": "falso",
            "PERSEO_AGENDA": "falso",
            "PERSEO_DEV_MOTOR": "falso",
            "PERSEO_WEB": "falso",
            "PERSEO_MODELO_SUPLENTE": "gemma-4-31b-it",
            "PERSEO_TELEGRAM_TOKEN": "de-mentira",
            "PERSEO_TELEGRAM_CHAT": "1",
            "PERSEO_TELEGRAM_API": "http://127.0.0.1:1",
            "PERSEO_DISPARADORES": "",
        }
    )
    nucleo.arrancar()
    try:
        _, panel = nucleo.pedir("/estado", nucleo.token)

        for id_pieza in ("correo", "agenda", "dev", "web"):
            actual = pieza(panel, id_pieza)
            comprobar(
                f"Una imitación de {id_pieza} no pasa por verde",
                actual.get("estado") == "aviso",
                f"{actual.get('estado')}: {actual.get('detalle')}",
            )

        # Suplente pedido pero sin clave: avisa de que no contestaría.
        comprobar(
            "El suplente sin clave avisa",
            pieza(panel, "suplente").get("estado") == "aviso",
            str(pieza(panel, "suplente").get("detalle")),
        )

        # El enlace de Telegram apunta al bucle local porque no hay tailnet: es
        # el fallo que en el móvil abre una página en blanco.
        comprobar(
            "Telegram avisa de que su enlace no sale de la máquina",
            pieza(panel, "telegram").get("estado") == "aviso",
            str(pieza(panel, "telegram").get("detalle")),
        )

        comprobar(
            "Los disparadores apagados se ven apagados",
            all(not d["activo"] for d in panel["disparadores"]),
            str(panel["disparadores"]),
        )

        servicios = {s["modelo"]: s for s in panel["cuota"]["servicios"]}
        comprobar(
            "La cuota enseña el suplente aunque no se haya gastado nada",
            servicios.get("gemma-4-31b-it", {}).get("usadas") == 0,
            str(panel["cuota"]["servicios"]),
        )
        comprobar(
            "Y trae el tope de la familia",
            servicios.get("gemma-4-31b-it", {}).get("tope") == 14400,
            str(servicios.get("gemma-4-31b-it")),
        )
        comprobar(
            "La cuota avisa de que cuenta por debajo",
            "app de voz" in panel["cuota"]["nota"],
            panel["cuota"]["nota"],
        )
    finally:
        nucleo.limpiar()


def main() -> None:
    comprobar_sin_configurar()
    print()
    comprobar_con_imitaciones()
    resumir()


if __name__ == "__main__":
    main()
