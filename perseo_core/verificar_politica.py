"""Verificación de la política de confirmación por niveles (§7 del plan).

Lo que hay que comprobar aquí no es que la tabla tenga las entradas que tiene
—eso se lee—, sino las tres cosas que se romperían sin que nadie se enterase:

1. Que **lo desconocido pregunta**. Un agente nuevo tiene que pedir un sí hasta
   que alguien lo clasifique, no colarse por el hueco.
2. Que la parada ocurre **antes** de ejecutar. Un trabajo irreversible que se
   quede esperando no puede haber hecho ya la mitad.
3. Que el **modo confianza caduca**. Un interruptor que se queda encendido para
   siempre es exactamente lo que esta política existe para evitar.

    python perseo_core/verificar_politica.py
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perseo_core import politica  # noqa: E402
from perseo_core.arnes_pruebas import Nucleo, comprobar, resumir  # noqa: E402


def comprobar_la_tabla() -> None:
    print("--- la tabla ---\n")

    comprobar(
        "Leer el vault es libre",
        politica.nivel("memoria", {"accion": "buscar"}) == politica.LIBRE,
    )
    comprobar(
        "Escribir en el vault es reversible",
        politica.nivel("memoria", {"accion": "anotar"}) == politica.REVERSIBLE,
    )
    comprobar(
        "Editar codigo es reversible",
        politica.nivel("dev", {"texto": "arregla esto"}) == politica.REVERSIBLE,
    )
    comprobar(
        "Abrir una app es libre",
        politica.nivel("pc", {"accion": "abrir_app"}) == politica.LIBRE,
    )
    comprobar(
        "Teclear a ciegas es irreversible",
        politica.nivel("pc", {"accion": "escribir_teclado"}) == politica.IRREVERSIBLE,
    )
    comprobar(
        "Leer una pagina web es libre",
        politica.nivel("web", {"accion": "leer"}) == politica.LIBRE,
    )

    # Lo importante de todo el modulo.
    comprobar(
        "Un agente que no esta en la tabla pregunta",
        politica.nivel("agente_del_futuro", {}) == politica.IRREVERSIBLE,
    )
    comprobar(
        "Y una accion nueva de un agente conocido, tambien",
        politica.nivel("memoria", {"accion": "borrar"}) == politica.IRREVERSIBLE,
    )
    comprobar(
        "El simulacro se queda fuera, que ya pregunta el solo",
        politica.nivel("simulacro", {"accion": "lo que sea"}) == politica.LIBRE,
    )
    print()


def comprobar_la_confianza() -> None:
    print("--- el modo confianza ---\n")

    with tempfile.TemporaryDirectory(prefix="perseo_pol_") as tmp:
        politica.iniciar(tmp)
        comprobar("De entrada no hay confianza", not politica.hay_confianza())
        comprobar(
            "Y lo irreversible pide un si",
            politica.pide_confirmacion("pc", {"accion": "escribir_teclado"}),
        )

        politica.activar_confianza(30)
        comprobar("Encendida, hay confianza", politica.hay_confianza())
        comprobar(
            "Y lo irreversible deja de preguntar",
            not politica.pide_confirmacion("pc", {"accion": "escribir_teclado"}),
        )
        comprobar(
            "Pero lo libre sigue sin preguntar (no cambia nada)",
            not politica.pide_confirmacion("memoria", {"accion": "buscar"}),
        )

        # Caducidad: se escribe una fecha ya pasada, que es lo que habria dentro
        # del fichero una hora despues.
        caducada = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        Path(tmp, "confianza.txt").write_text(caducada, encoding="utf-8")
        comprobar("Una confianza caducada no vale", not politica.hay_confianza())
        comprobar(
            "Y vuelve a preguntar",
            politica.pide_confirmacion("pc", {"accion": "escribir_teclado"}),
        )
        comprobar(
            "El fichero caducado se borra al leerlo",
            not Path(tmp, "confianza.txt").exists(),
        )

        # Topes y basura.
        hasta = politica.activar_confianza(99999)
        margen = hasta - datetime.now(timezone.utc)
        comprobar(
            "Pedir mil minutos no da mas que el tope",
            margen <= timedelta(minutes=politica.MAX_MINUTOS_CONFIANZA),
            f"{margen.total_seconds() / 60:.0f} minutos",
        )
        Path(tmp, "confianza.txt").write_text("esto no es una fecha", encoding="utf-8")
        comprobar("Un fichero ilegible se trata como sin confianza", not politica.hay_confianza())

        politica.activar_confianza(30)
        politica.desactivar_confianza()
        comprobar("Y se puede apagar a mano", not politica.hay_confianza())
    print()


def comprobar_de_punta_a_punta() -> None:
    nucleo = Nucleo()
    nucleo.arrancar()
    token = nucleo.token

    # 1. Un trabajo irreversible se para **antes** de ejecutarse.
    _, trabajo = nucleo.pedir(
        "/trabajos",
        token,
        "POST",
        {"agente": "pc", "peticion": {"accion": "escribir_teclado", "parametro": "hola"}},
    )
    id_trabajo = int(trabajo["id"])
    esperando = nucleo.esperar_estado(id_trabajo, ("esperando", "hecho", "fallido"), intentos=60)
    comprobar(
        "Teclear se para y pide un si",
        esperando.get("estado") == "esperando",
        str(esperando.get("estado")),
    )
    comprobar(
        "Con su pregunta guardada",
        "irreversible" in str((esperando.get("confirmacion") or {}).get("resumen", "")),
        str((esperando.get("confirmacion") or {}).get("resumen")),
    )

    # 2. Rechazarlo lo cierra sin ejecutarlo. (Aprobarlo tecleara de verdad en la
    #    ventana que tenga el foco, asi que eso no se prueba aqui.)
    nucleo.pedir(f"/trabajos/{id_trabajo}/rechazar", token, "POST", {})
    rechazado = nucleo.esperar_estado(id_trabajo, ("rechazado", "hecho"), intentos=40)
    comprobar(
        "Y rechazarlo lo cierra sin ejecutar",
        rechazado.get("estado") == "rechazado" and rechazado.get("resultado") is None,
        str(rechazado.get("estado")),
    )

    # 3. Lo libre no pasa por ahi.
    _, libre = nucleo.pedir(
        "/trabajos", token, "POST", {"agente": "eco", "peticion": {"texto": "hola"}}
    )
    hecho = nucleo.esperar_estado(int(libre["id"]), ("hecho", "esperando", "fallido"), intentos=40)
    comprobar("Lo libre se ejecuta sin preguntar", hecho.get("estado") == "hecho", str(hecho.get("estado")))

    # 4. El interruptor, por la API.
    codigo, sin_token = nucleo.pedir("/confianza", None)
    comprobar("El estado de la confianza no se cuenta sin token", codigo == 401, f"HTTP {codigo}")

    _, estado = nucleo.pedir("/confianza", token)
    comprobar("De entrada esta apagada", estado.get("confianza") is False, str(estado))

    _, encendida = nucleo.pedir("/confianza", token, "POST", {"minutos": 10})
    comprobar("Se puede encender", encendida.get("confianza") is True, str(encendida))

    # 5. Y con ella encendida, lo irreversible pasa sin parar.
    _, con_confianza = nucleo.pedir(
        "/trabajos",
        token,
        "POST",
        {"agente": "pc", "peticion": {"accion": "atajo_teclado", "parametro": "ctrl,inventada"}},
    )
    resuelto = nucleo.esperar_estado(
        int(con_confianza["id"]), ("hecho", "fallido", "esperando"), intentos=60
    )
    comprobar(
        "Con confianza, lo irreversible ya no pide un si",
        resuelto.get("estado") != "esperando",
        f"estado={resuelto.get('estado')}",
    )

    _, apagada = nucleo.pedir("/confianza", token, "POST", {"activo": False})
    comprobar("Y se puede apagar", apagada.get("confianza") is False, str(apagada))

    # 6. Sobre todo: la confianza no sobrevive a un reinicio con el plazo pasado,
    #    y el estado se lee del disco, no de la memoria del proceso.
    nucleo.pedir("/confianza", token, "POST", {"minutos": 5})
    nucleo.reiniciar()
    _, tras_reinicio = nucleo.pedir("/confianza", nucleo.token)
    comprobar(
        "La confianza en curso sobrevive al reinicio",
        tras_reinicio.get("confianza") is True,
        str(tras_reinicio),
    )

    nucleo.limpiar()


def main() -> None:
    comprobar_la_tabla()
    comprobar_la_confianza()
    comprobar_de_punta_a_punta()
    resumir()


if __name__ == "__main__":
    main()
