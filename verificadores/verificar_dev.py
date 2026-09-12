"""Verificación del agente `dev`, sin gastar suscripción.

El motor de verdad es Claude Code, y hacerle encargos reales en cada verificación
costaría cuota y tardaría minutos. Se usa `MotorFalso`, que recorre el mismo
camino —cola, carril propio, resultado, titular— y además puede tardar a
propósito, que es como se comprueba lo que de verdad importa de esta pieza:

**que un encargo de código largo no deje al resto de la cola esperando.**

Lo que este script no puede comprobar es que Claude Code entienda el encargo.
Eso se prueba a mano, una vez, con `PERSEO_DEV_MOTOR` sin poner.

    python verificadores/verificar_dev.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perseo_core.agentes import dev, dev_motores  # noqa: E402
from perseo_core.infra import almacen  # noqa: E402
from verificadores.arnes_pruebas import Nucleo, comprobar, resumir  # noqa: E402

#: Lo que tarda el encargo simulado. Suficiente para que el trabajo corto que se
#: encola detrás tenga que adelantarlo si los carriles funcionan.
TARDANZA = "6"


def comprobar_en_proceso() -> None:
    print("--- sin levantar el nucleo ---\n")

    previo = dict(os.environ)
    with tempfile.TemporaryDirectory(prefix="perseo_dev_") as tmp:
        os.environ["PERSEO_CORE_DATOS"] = tmp
        os.environ["PERSEO_DEV_MOTOR"] = "falso"
        cfg = almacen.cargar_configuracion()
    os.environ.clear()
    os.environ.update(previo)

    motor = dev.iniciar(cfg)
    comprobar("Con motor falso hay motor", motor is not None)

    # 1. Nada sale de la raiz permitida. Es la misma regla que en `memoria`, y por
    #    el mismo motivo: un encargo puede venir de un correo.
    raiz = Path(cfg.dev_raiz).resolve()
    comprobar("Sin directorio, el encargo va a la raiz", dev.resolver_raiz("") == raiz, str(raiz))
    # El cerco creció el 2026-08-24: la raíz, el perfil del usuario y el
    # Escritorio. Subir un escalón desde el repositorio YA VALE —es el
    # perfil—, así que aquí se comprueba lo que sigue estando fuera.
    fuera_del_perfil = str(Path.home().resolve().parent)
    for intento in (fuera_del_perfil, "../../..", "C:\\Windows"):
        try:
            dev.resolver_raiz(intento)
            comprobar(f"Se niega a trabajar en {intento!r}", False, "no se nego")
        except ValueError:
            comprobar(f"Se niega a trabajar en {intento!r}", True)

    # 2. Las listas de herramientas son lo que hace aceptable que no pregunte.
    comprobar(
        "git push esta denegado",
        any("git push" in h for h in dev_motores.HERRAMIENTAS_DENEGADAS),
    )
    comprobar(
        "Y borrar tambien",
        any(h.startswith("Bash(rm ") for h in dev_motores.HERRAMIENTAS_DENEGADAS),
    )
    comprobar(
        "No hay un Bash abierto entre las permitidas",
        "Bash" not in dev_motores.HERRAMIENTAS_PERMITIDAS,
        ", ".join(h for h in dev_motores.HERRAMIENTAS_PERMITIDAS if h.startswith("Bash")),
    )
    comprobar("Hay tope de vueltas", dev_motores.MAX_VUELTAS > 0, str(dev_motores.MAX_VUELTAS))
    # La lista ancha es para lo que pide el señor Persus con el dedo; sin ella,
    # "abre la app de armario" acaba en verde sin abrir nada.
    comprobar(
        "La lista ancha puede arrancar procesos",
        "Bash" in dev_motores.HERRAMIENTAS_PERMITIDAS_AMPLIAS
        and "Task" in dev_motores.HERRAMIENTAS_PERMITIDAS_AMPLIAS,
    )
    comprobar(
        "Y lo denegado sigue denegado con ella",
        all(h in dev_motores.HERRAMIENTAS_DENEGADAS for h in ("Bash(git push*)", "Bash(rm *)")),
    )
    comprobar(
        "Una URL no se toma por directorio",
        not dev._parece_carpeta("http://127.0.0.1:8000") and dev._parece_carpeta("C:\\algo"),
    )

    # 2 bis. La bitacora: es lo que convierte "HECHO (16 vueltas)" en algo que
    #        se puede depurar sin abrir el registro del nucleo.
    dev._anotar(9001, dev_motores.Paso(tipo="herramienta", titulo="Leyendo api.py"))
    dev._anotar(9001, dev_motores.Paso(tipo="resultado", titulo="Error", agente="tu_1", ok=False))
    actividad = dev.actividad_de(9001)
    comprobar(
        "La bitacora apunta el paso a paso",
        len(actividad["pasos"]) == 2 and len(actividad["agentes"]) == 2,
        f"{len(actividad['pasos'])} pasos, {len(actividad['agentes'])} agentes",
    )
    comprobar(
        "Y separa lo de cada subagente",
        any(a["id"] == "tu_1" and a["fallos"] == 1 for a in actividad["agentes"]),
    )

    # 3. Un encargo vacio se rechaza antes de arrancar nada.
    async def vacio() -> bool:
        try:
            await dev._dev({"peticion": {"texto": "   "}})
        except ValueError:
            return True
        return False

    comprobar("Un encargo sin texto se rechaza", asyncio.run(vacio()))
    dev.detener()
    print()


def comprobar_de_punta_a_punta() -> None:
    nucleo = Nucleo(
        {
            "PERSEO_DEV_MOTOR": "falso",
            "PERSEO_DEV_TARDANZA": TARDANZA,
        }
    )
    nucleo.arrancar()
    token = nucleo.token

    # 4. Un encargo de codigo entra por la cola como cualquier otro.
    codigo, encargo = nucleo.pedir(
        "/trabajos",
        token,
        "POST",
        {"agente": "dev", "peticion": {"texto": "escribe una funcion que sume dos numeros"}},
    )
    comprobar("El agente dev existe para la API", codigo == 201, f"HTTP {codigo}")
    if codigo != 201:
        nucleo.limpiar()
        return

    # 5. Lo que de verdad importa: mientras el encargo largo corre, un trabajo
    #    corto tiene que poder adelantarlo. Con un solo trabajador, este `eco`
    #    esperaria los seis segundos del encargo.
    time.sleep(1.0)
    _, corto = nucleo.pedir(
        "/trabajos", token, "POST", {"agente": "eco", "peticion": {"texto": "hola"}}
    )
    hecho_corto = nucleo.esperar_estado(int(corto["id"]), ("hecho", "fallido"), intentos=16)
    _, mientras = nucleo.pedir(f"/trabajos/{encargo['id']}", token)

    comprobar(
        "Un trabajo corto termina mientras dev sigue",
        hecho_corto.get("estado") == "hecho" and mientras.get("estado") == "en_curso",
        f"corto={hecho_corto.get('estado')} dev={mientras.get('estado')}",
    )

    # 6. Y el encargo acaba bien, con su titular para el canal.
    terminado = nucleo.esperar_estado(int(encargo["id"]), ("hecho", "fallido"), intentos=80)
    comprobar(
        "El encargo termina bien",
        terminado.get("estado") == "hecho",
        f"estado={terminado.get('estado')} error={terminado.get('error')}",
    )
    resultado: dict[str, Any] = terminado.get("resultado") or {}
    comprobar(
        "Devuelve lo que contesto el motor",
        "escribe una funcion" in str(resultado.get("texto")),
        str(resultado.get("texto"))[:80],
    )
    comprobar("Trae titular para el canal", bool(resultado.get("titular")), str(resultado.get("titular")))

    # 6 bis. Y la bitacora se puede leer por la API DESPUES de terminar, que es
    # cuando uno se pregunta que hizo de verdad.
    codigo_bitacora, bitacora = nucleo.pedir(f"/trabajos/{encargo['id']}/actividad", token)
    comprobar(
        "La actividad del encargo se lee por la API",
        codigo_bitacora == 200 and len(bitacora.get("pasos") or []) > 0,
        f"HTTP {codigo_bitacora}, {len(bitacora.get('pasos') or [])} pasos",
    )
    comprobar(
        "Y la sesion, para poder continuar el encargo",
        bool(resultado.get("sesion")),
        str(resultado.get("sesion")),
    )

    # 7. Un encargo que apunta fuera de la raiz falla, y no a medias.
    _, fuera = nucleo.pedir(
        "/trabajos",
        token,
        "POST",
        {"agente": "dev", "peticion": {"texto": "toca algo", "directorio": "../.."}},
    )
    fallido = nucleo.esperar_estado(int(fuera["id"]), ("fallido", "hecho"), intentos=60)
    comprobar(
        "Un encargo fuera de la raiz falla",
        fallido.get("estado") == "fallido",
        str(fallido.get("error"))[:90],
    )

    nucleo.limpiar()


def main() -> None:
    comprobar_en_proceso()
    comprobar_de_punta_a_punta()
    resumir()


if __name__ == "__main__":
    main()
