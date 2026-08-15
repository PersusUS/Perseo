"""Verificación del criterio de aceptación de la Fase A.

Criterio (bitacora/05_PLAN_PERSEO_V2.md §9):
  "encolas un trabajo por HTTP desde el móvil, se ejecuta, y sigue en la cola
   tras reiniciar el núcleo."

Arranca el núcleo como proceso hijo sobre un directorio de datos temporal, así
que no toca el estado real ni el puerto por defecto. Ejecutar desde la raíz:

    python perseo_core/verificar_fase_a.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perseo_core.arnes_pruebas import Escucha, Nucleo, comprobar, resumir  # noqa: E402


def main() -> None:
    nucleo = Nucleo()
    print(f"Directorio de datos aislado: {nucleo.datos}")
    print(f"Puerto: {nucleo.puerto}\n")

    nucleo.arrancar()
    token = nucleo.token
    comprobar(
        "Se genera un token en el primer arranque", len(token) > 20, f"{len(token)} caracteres"
    )

    # 1. Sin token, la API no contesta.
    estado, _ = nucleo.pedir("/trabajos")
    comprobar("Sin token devuelve 401", estado == 401, f"HTTP {estado}")

    # 2. /salud es publica a proposito.
    estado, salud = nucleo.pedir("/salud")
    comprobar("/salud responde sin token", estado == 200, f"agentes={salud.get('agentes')}")

    # 3. Encolar por HTTP.
    estado, trabajo = nucleo.pedir(
        "/trabajos",
        token,
        "POST",
        {"agente": "eco", "peticion": {"texto": "hola desde el movil"}, "origen": "texto"},
    )
    comprobar("Encolar por HTTP devuelve 201", estado == 201, f"HTTP {estado}")
    id_trabajo = trabajo.get("id")
    comprobar(
        "El trabajo nace pendiente",
        trabajo.get("estado") == "pendiente",
        str(trabajo.get("estado")),
    )

    # 4. El trabajador lo ejecuta.
    final = nucleo.esperar_estado(id_trabajo, ("hecho", "fallido"))
    comprobar("El trabajador lo completa", final.get("estado") == "hecho", str(final.get("estado")))
    comprobar(
        "El resultado llega intacto",
        (final.get("resultado") or {}).get("texto") == "hola desde el movil",
        json.dumps(final.get("resultado"), ensure_ascii=False),
    )

    # 5. Agente inexistente se rechaza en la puerta.
    estado, _ = nucleo.pedir("/trabajos", token, "POST", {"agente": "inventado", "peticion": {}})
    comprobar("Agente desconocido devuelve 400", estado == 400, f"HTTP {estado}")

    # 6. Encolar uno mas y matar el proceso con el a medias.
    _, huerfano = nucleo.pedir(
        "/trabajos", token, "POST", {"agente": "eco", "peticion": {"texto": "sobrevive al reinicio"}}
    )
    id_huerfano = huerfano["id"]
    # Se sondea en vez de dormir un rato fijo: el trabajador tarda hasta medio
    # segundo en reclamarlo y el agente eco solo pasa un segundo dentro. Con un
    # `sleep` fijo, la ventana se falla de vez en cuando y parece un fallo del
    # núcleo cuando es del reloj.
    en_vuelo = nucleo.esperar_estado(id_huerfano, ("en_curso",), intentos=20)
    comprobar(
        "El segundo trabajo esta en curso al matar",
        en_vuelo.get("estado") == "en_curso",
        str(en_vuelo.get("estado")),
    )

    nucleo.reiniciar()

    # 7. Lo importante: la cola sobrevivio al reinicio.
    _, tras_reinicio = nucleo.pedir(f"/trabajos/{id_trabajo}", token)
    comprobar(
        "El trabajo completado sigue en la cola tras reiniciar",
        tras_reinicio.get("estado") == "hecho",
        f"id={id_trabajo} estado={tras_reinicio.get('estado')}",
    )

    # 8. Y el que quedo a medias vuelve a ejecutarse en vez de quedarse clavado.
    recuperado = nucleo.esperar_estado(id_huerfano, ("hecho",))
    comprobar(
        "El trabajo huerfano se recupera y termina",
        recuperado.get("estado") == "hecho",
        f"intentos={recuperado.get('intentos')}",
    )

    # 9. El flujo SSE entrega eventos.
    escucha = Escucha(nucleo, cuantos=2)
    escucha.empezar()
    nucleo.pedir("/trabajos", token, "POST", {"agente": "eco", "peticion": {"texto": "por el bus"}})
    tipos = escucha.terminar()
    comprobar("El flujo SSE entrega eventos", len(tipos) >= 1, f"{len(tipos)} evento(s)")
    if tipos:
        comprobar("Los eventos llevan tipo reconocible", all(tipos), ", ".join(tipos))

    nucleo.limpiar()
    resumir()


if __name__ == "__main__":
    main()
