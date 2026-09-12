"""Comprueba que el router local decide bien con Ollama levantado.

Es la pieza que la Fase A dejó sin verificar: la ruta de respaldo (Ollama caído →
encolar todo) sí estaba probada, pero no que el modelo con gramática devuelva el
esquema esperado.

No basta con que Ollama conteste. Un modelo puede **ignorar la gramática** y
devolver prosa: la petición sale con 200, el router marca `disponible = True`, y
sin embargo cae al respaldo en cada decisión. Por eso aquí se mira el contenido,
no solo que no haya excepción.

    python perseo_core/verificar_router.py [modelo]

Sin argumento usa `PERSEO_MODELO_ROUTER` (por defecto `qwen3:4b`).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if len(sys.argv) > 1:
    os.environ["PERSEO_MODELO_ROUTER"] = sys.argv[1]
# Directorio aparte: este script no debe tocar la cola real ni su token.
os.environ.setdefault(
    "PERSEO_CORE_DATOS", str(Path(tempfile.gettempdir()) / "perseo_verificar_router")
)

from perseo_core import almacen  # noqa: E402
from perseo_core.agentes import REGISTRO, Router  # noqa: E402
from perseo_core.arnes_pruebas import comprobar, resumir  # noqa: E402

#: Casos y el destino que se espera. `None` = cualquiera vale; lo que se
#: comprueba entonces es solo que la respuesta salga del esquema.
CASOS: list[tuple[str, str | None]] = [
    ("hola, buenas", "responder"),
    ("gracias, hasta luego", "responder"),
    ("resumeme los correos sin leer y dime cuales necesitan respuesta", None),
    ("busca los precios de la RTX 5060 y hazme una tabla comparativa", None),
]

DESTINOS = {"responder", "encolar", "no_seguro"}

async def main() -> None:
    # El router avisa por registro cuando Ollama no responde; sin esto el aviso
    # se pierde y un fallo de conexión parece un fallo de decisión.
    logging.basicConfig(level=logging.INFO, format="       %(levelname)s %(message)s")

    cfg = almacen.cargar_configuracion()
    print(f"Modelo: {cfg.modelo_router}   Ollama: {cfg.url_ollama}\n")

    router = Router(cfg)
    await router.abrir()
    try:
        for texto, esperado in CASOS:
            inicio = time.perf_counter()
            ruta = await router.decidir(texto)
            ms = (time.perf_counter() - inicio) * 1000
            print(f"  «{texto}»")
            print(f"    destino={ruta.destino} agente={ruta.agente} ({ms:.0f} ms)")
            if ruta.motivo:
                print(f"    motivo: {ruta.motivo}")
            if ruta.respuesta:
                print(f"    respuesta: {ruta.respuesta}")

            # El respaldo pone este motivo; si aparece, la decisión no es del
            # modelo y darla por buena sería engañarse.
            del_modelo = not ruta.motivo.startswith(
                ("Router local no disponible", "El router no devolvió", "Ollama respondió")
            )
            comprobar("    decide el modelo, no el respaldo", del_modelo)
            comprobar("    el destino es del esquema", ruta.destino in DESTINOS, ruta.destino)
            # Contra el registro de verdad, no contra una lista escrita a mano:
            # cada agente nuevo la dejaría desfasada, y el fallo parecería del
            # router cuando sería de la prueba.
            comprobar("    el agente existe", ruta.agente in REGISTRO, ruta.agente)
            if esperado and del_modelo:
                comprobar(
                    f"    clasifica como {esperado}",
                    ruta.destino == esperado,
                    ruta.destino,
                )
            print()
    finally:
        await router.cerrar()

    comprobar("Ollama estaba levantado", router.disponible is True, str(router.disponible))

    resumir()


if __name__ == "__main__":
    asyncio.run(main())
