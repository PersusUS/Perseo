"""Verificación de los recordatorios contra el núcleo real, de punta a punta.

Lo que se comprueba, con un núcleo de verdad en un directorio temporal:

- apuntar un recordatorio por la cola devuelve la hora ya resuelta,
- se lista y se quita por su texto,
- el que vence **suena el timbre**: deja su motivo en el marcador de llamada
  que vigila la app (apartado aquí, para no hacer sonar la de verdad),
- y deja además su aviso en la cola, que es de donde lo lleva Telegram;
- un encargo de código que termina **sin nadie esperándolo** suena el mismo
  timbre; uno que alguien espera, no, porque ya se lo lleva quien pregunta.

El disparador mira cada segundo en vez de cada treinta, para no esperar.

    python verificadores/verificar_recordatorios.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verificadores.arnes_pruebas import Nucleo, comprobar, resumir  # noqa: E402

MARCADOR = ".perseo-autollamada"


def _encolar_y_esperar(nucleo: Nucleo, peticion: dict, espera: float = 15) -> dict:
    codigo, trabajo = nucleo.pedir(
        "/trabajos", token=nucleo.token, metodo="POST",
        cuerpo={"agente": "recordatorios", "peticion": peticion},
    )
    if codigo not in (200, 201):
        return {"estado": f"HTTP {codigo}", "resultado": trabajo}
    limite = time.time() + espera
    while time.time() < limite:
        _, actual = nucleo.pedir(f"/trabajos/{trabajo['id']}", token=nucleo.token)
        if actual.get("estado") in ("hecho", "fallido", "rechazado", "cancelado"):
            return actual
        time.sleep(0.3)
    return {"estado": "sin terminar"}


def comprobar_todo() -> None:
    nucleo = Nucleo({"PERSEO_DISPARADORES": "recordatorios", "PERSEO_RECORDATORIOS_INTERVALO": "1"})
    nucleo.arrancar()
    try:
        apuntado = _encolar_y_esperar(nucleo, {"accion": "crear", "texto": "Regar las plantas", "hora": "23:59"})
        texto = (apuntado.get("resultado") or {}).get("texto", "")
        comprobar("Apuntar devuelve la hora ya resuelta", "a las 23:59" in texto, texto)

        lista = _encolar_y_esperar(nucleo, {"accion": "listar"})
        texto = (lista.get("resultado") or {}).get("texto", "")
        comprobar("Se lista con su texto", "Regar las plantas" in texto, texto)

        quitado = _encolar_y_esperar(nucleo, {"accion": "cancelar", "texto": "regar"})
        texto = (quitado.get("resultado") or {}).get("texto", "")
        comprobar("Se quita por el principio del texto", texto == "Quitado: «Regar las plantas».", texto)

        # Uno que vence en un segundo.
        _encolar_y_esperar(nucleo, {"accion": "crear", "texto": "Sacar el pan del horno", "en_minutos": 0.02})
        marcador = nucleo.datos / MARCADOR
        limite = time.time() + 20
        while time.time() < limite and not marcador.exists():
            time.sleep(0.3)
        contenido = marcador.read_text(encoding="utf-8") if marcador.exists() else ""
        comprobar(
            "Al vencer suena el timbre de llamada",
            "Recordatorio: Sacar el pan del horno" in contenido,
            contenido.strip() or "no apareció el marcador",
        )

        _, trabajos = nucleo.pedir("/trabajos?limite=10", token=nucleo.token)
        avisos = [
            t for t in (trabajos.get("trabajos") if isinstance(trabajos, dict) else trabajos) or []
            if t.get("agente") == "recordatorios" and (t.get("peticion") or {}).get("accion") == "avisar"
        ]
        comprobar(
            "Y deja su aviso en la cola, que es lo que lleva Telegram",
            bool(avisos) and avisos[0].get("origen") == "disparador",
            str([(t.get("id"), t.get("estado"), t.get("origen")) for t in avisos]),
        )
    except Exception:
        print("===== VOLCADO DEL HIJO =====")
        nucleo.volcar()
        raise
    finally:
        nucleo.parar()


def comprobar_encargos() -> None:
    nucleo = Nucleo(
        {"PERSEO_DISPARADORES": "", "PERSEO_DEV_MOTOR": "falso", "PERSEO_DEV_TARDANZA": "1"}
    )
    nucleo.arrancar()
    try:
        marcador = nucleo.datos / MARCADOR
        # Uno que nadie espera: se encola y no se pregunta por él.
        codigo, sin_mirar = nucleo.pedir(
            "/trabajos", token=nucleo.token, metodo="POST",
            cuerpo={"agente": "dev", "peticion": {"texto": "encargo que nadie mira"}},
        )
        comprobar("El encargo entra en la cola", codigo in (200, 201), str(codigo))
        limite = time.time() + 30
        while time.time() < limite and not marcador.exists():
            time.sleep(0.5)
        contenido = marcador.read_text(encoding="utf-8") if marcador.exists() else ""
        comprobar(
            "Un encargo que termina sin nadie esperándolo llama",
            f"El encargo #{sin_mirar.get('id')}" in contenido and "ha terminado" in contenido,
            contenido.strip() or "no apareció el marcador",
        )

        # Uno que se espera preguntando, como hace la voz: no se cuenta dos veces.
        marcador.unlink(missing_ok=True)
        _, esperado = nucleo.pedir(
            "/trabajos", token=nucleo.token, metodo="POST",
            cuerpo={"agente": "dev", "peticion": {"texto": "encargo que se espera"}},
        )
        hecho = nucleo.esperar_estado(esperado["id"], ("hecho", "fallido"))
        time.sleep(4)
        comprobar(
            "Uno que alguien espera no suena el timbre",
            hecho.get("estado") == "hecho" and not marcador.exists(),
            f"{hecho.get('estado')}, marcador: {marcador.exists()}",
        )
    except Exception:
        print("===== VOLCADO DEL HIJO =====")
        nucleo.volcar()
        raise
    finally:
        nucleo.parar()


if __name__ == "__main__":
    print("== Recordatorios ==")
    comprobar_todo()
    print("\n== Encargos que terminan sin nadie mirando ==")
    comprobar_encargos()
    resumir()
