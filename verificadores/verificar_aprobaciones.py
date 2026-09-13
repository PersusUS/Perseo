"""Verificación del camino de confirmación (Fase B, paso 6).

Lo que se quiere probar no es que salga un botón, sino que **la pregunta
sobrevive**: un trabajo puede pararse a mitad, quedar esperando un sí, aguantar
un reinicio del núcleo, y continuar cuando alguien contesta — desde la web hoy y
desde Telegram mañana.

    python verificadores/verificar_aprobaciones.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verificadores.arnes_pruebas import Escucha, Nucleo, comprobar, resumir  # noqa: E402


def encolar_simulacro(nucleo: Nucleo, accion: str, detalle: str = "") -> int:
    _, trabajo = nucleo.pedir(
        "/trabajos",
        nucleo.token,
        "POST",
        {"agente": "simulacro", "peticion": {"accion": accion, "detalle": detalle}},
    )
    return int(trabajo["id"])


def main() -> None:
    nucleo = Nucleo()
    print(f"Directorio de datos aislado: {nucleo.datos}")
    print(f"Puerto: {nucleo.puerto}\n")

    nucleo.arrancar()
    token = nucleo.token

    estado, salud = nucleo.pedir("/salud")
    comprobar(
        "El agente de simulacro esta registrado",
        "simulacro" in (salud.get("agentes") or []),
        str(salud.get("agentes")),
    )

    # 1. El agente para solo y deja la pregunta guardada.
    escucha = Escucha(nucleo, cuantos=3)
    escucha.empezar()
    id_uno = encolar_simulacro(nucleo, "borrar la carpeta de descargas", "37 ficheros")
    esperando = nucleo.esperar_estado(id_uno, ("esperando", "hecho", "fallido"))
    comprobar(
        "El trabajo se para a esperar confirmacion",
        esperando.get("estado") == "esperando",
        str(esperando.get("estado")),
    )
    confirmacion = esperando.get("confirmacion") or {}
    comprobar(
        "La pregunta queda guardada con el trabajo",
        "borrar la carpeta de descargas" in str(confirmacion.get("resumen")),
        str(confirmacion.get("resumen")),
    )
    comprobar(
        "El detalle tambien",
        confirmacion.get("detalle") == "37 ficheros",
        str(confirmacion.get("detalle")),
    )
    comprobar("Todavia no hay decision", confirmacion.get("decision") is None)

    tipos = escucha.terminar()
    comprobar(
        "El bus avisa de que hay algo que confirmar",
        "trabajo.espera_confirmacion" in tipos,
        ", ".join(tipos),
    )

    # 2. Nadie lo ejecuta mientras espera. Es lo que separa 'esperando' de
    #    'pendiente': el trabajador no debe tocarlo.
    time.sleep(2)
    _, quieto = nucleo.pedir(f"/trabajos/{id_uno}", token)
    comprobar(
        "Sigue esperando y nadie lo ha ejecutado",
        quieto.get("estado") == "esperando",
        f"estado={quieto.get('estado')} intentos={quieto.get('intentos')}",
    )

    # 3. Y sobrevive a reiniciar el nucleo sin convertirse en huerfano.
    nucleo.reiniciar()
    _, tras_reinicio = nucleo.pedir(f"/trabajos/{id_uno}", token)
    comprobar(
        "La pregunta sobrevive al reinicio",
        tras_reinicio.get("estado") == "esperando",
        str(tras_reinicio.get("estado")),
    )
    comprobar(
        "Y no la barre la recuperacion de huerfanos",
        (tras_reinicio.get("confirmacion") or {}).get("resumen") is not None,
    )

    # 4. Aprobar lo devuelve a la cola y esta vez el agente sigue adelante.
    estado, aprobado = nucleo.pedir(f"/trabajos/{id_uno}/aprobar", token, "POST")
    comprobar("Aprobar devuelve 200", estado == 200, f"HTTP {estado}")
    comprobar(
        "Al aprobar vuelve a la cola",
        aprobado.get("estado") == "pendiente",
        str(aprobado.get("estado")),
    )
    comprobar(
        "La decision queda registrada",
        (aprobado.get("confirmacion") or {}).get("decision") == "aprobado",
        str((aprobado.get("confirmacion") or {}).get("decision")),
    )

    hecho = nucleo.esperar_estado(id_uno, ("hecho", "fallido"))
    comprobar("Y termina", hecho.get("estado") == "hecho", str(hecho.get("estado")))
    comprobar(
        "El agente hizo la accion que se aprobo",
        "borrar la carpeta de descargas" in str((hecho.get("resultado") or {}).get("texto")),
        str((hecho.get("resultado") or {}).get("texto")),
    )

    # 5. Contestar dos veces no ejecuta la accion dos veces. Es el caso real de
    #    tener la web abierta y el movil en la mano.
    estado, _ = nucleo.pedir(f"/trabajos/{id_uno}/aprobar", token, "POST")
    comprobar("Aprobar por segunda vez devuelve 409", estado == 409, f"HTTP {estado}")

    # 6. Rechazar cierra el trabajo sin ejecutarlo.
    id_dos = encolar_simulacro(nucleo, "vaciar la papelera")
    nucleo.esperar_estado(id_dos, ("esperando",))
    estado, rechazado = nucleo.pedir(f"/trabajos/{id_dos}/rechazar", token, "POST")
    comprobar("Rechazar devuelve 200", estado == 200, f"HTTP {estado}")
    comprobar(
        "El trabajo rechazado queda cerrado",
        rechazado.get("estado") == "rechazado",
        str(rechazado.get("estado")),
    )
    time.sleep(1.5)
    _, sigue_rechazado = nucleo.pedir(f"/trabajos/{id_dos}", token)
    comprobar(
        "Y nadie lo ejecuta despues",
        sigue_rechazado.get("estado") == "rechazado" and sigue_rechazado.get("resultado") is None,
        f"estado={sigue_rechazado.get('estado')} resultado={sigue_rechazado.get('resultado')}",
    )

    # 7. Un trabajo esperando tambien se puede cancelar sin contestar.
    id_tres = encolar_simulacro(nucleo, "reiniciar el equipo")
    nucleo.esperar_estado(id_tres, ("esperando",))
    estado, cancelado = nucleo.pedir(f"/trabajos/{id_tres}/cancelar", token, "POST")
    comprobar(
        "Se puede cancelar un trabajo que espera",
        estado == 200 and cancelado.get("estado") == "cancelado",
        f"HTTP {estado} estado={cancelado.get('estado')}",
    )

    # 8. Aprobar algo que no espera no cuela.
    estado, _ = nucleo.pedir(f"/trabajos/{id_tres}/aprobar", token, "POST")
    comprobar("Aprobar un trabajo cancelado devuelve 409", estado == 409, f"HTTP {estado}")
    estado, _ = nucleo.pedir("/trabajos/99999/aprobar", token, "POST")
    comprobar("Aprobar un trabajo inexistente devuelve 404", estado == 404, f"HTTP {estado}")
    estado, _ = nucleo.pedir(f"/trabajos/{id_uno}/aprobar", None, "POST")
    comprobar("Sin token, aprobar devuelve 401", estado == 401, f"HTTP {estado}")

    nucleo.limpiar()
    resumir()


if __name__ == "__main__":
    main()
