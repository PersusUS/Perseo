"""Verificación de la Fase D: el correo que se tría solo.

Comprueba el criterio de la fase de punta a punta —llega un correo, el triaje
local lo clasifica, y si es relevante sale un aviso— **sin Gmail y sin Telegram**:
el buzón es un fichero JSON al que este script va añadiendo mensajes, y el canal
es el mismo servidor de mentira que usa `verificar_telegram.py`.

Lo que sí hace falta para las comprobaciones del modelo es Ollama levantado. Si
no lo está, esas se saltan y se dice; el resto —marca de agua, estreno, regla del
canal, respaldo sin modelo— no depende de él.

    python verificadores/verificar_fase_d.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perseo_core.agentes import correo  # noqa: E402
from perseo_core.infra import almacen, disparadores  # noqa: E402
from perseo_core.servicios import triaje  # noqa: E402
from perseo_core.dominio.clasificacion import CLASES, Clasificacion, IGNORAR, NO_SEGURO, REQUIERE_ACCION  # noqa: E402
from verificadores.arnes_pruebas import Nucleo, comprobar, resumir  # noqa: E402
from perseo_core.infra.bus import Bus  # noqa: E402
from verificadores.verificar_telegram import CHAT, TOKEN_FALSO, FalsoTelegram  # noqa: E402

#: Cada cuánto mira el buzón durante la prueba. En producción son 300 segundos.
INTERVALO = "2"

#: Lo que se le mete al buzón. El asunto y el extracto son la parte que **no**
#: puede salir por Telegram, así que se eligen distinguibles.
ASUNTO_SECRETO = "Presupuesto de la reforma del bano"
EXTRACTO_SECRETO = "Adjunto el desglose con los 14.200 euros y las tres partidas."

VIEJOS = [
    {
        "id": "viejo-1",
        "remitente": "newsletter@tienda.example",
        "asunto": "Rebajas de verano hasta el 70%",
        "extracto": "Ultimas horas para aprovechar los descuentos de temporada.",
    },
    {
        "id": "viejo-2",
        "remitente": "no-reply@redsocial.example",
        "asunto": "Tienes 4 notificaciones nuevas",
        "extracto": "Entra para ver la actividad de tu cuenta.",
    },
]

NUEVO = {
    "id": "nuevo-1",
    "remitente": "constructora@example.com",
    "asunto": ASUNTO_SECRETO,
    "extracto": EXTRACTO_SECRETO,
}


def escribir_buzon(ruta: Path, mensajes: list[dict[str, Any]]) -> None:
    ruta.write_text(json.dumps(mensajes, ensure_ascii=False, indent=2), encoding="utf-8")


def trabajos_de_correo(nucleo: Nucleo) -> list[dict[str, Any]]:
    _, lista = nucleo.pedir("/trabajos?limite=100", nucleo.token)
    return [t for t in (lista.get("trabajos") or []) if t.get("agente") == "correo"]


def esperar_trabajo_de_correo(nucleo: Nucleo, segundos: float = 20) -> dict[str, Any] | None:
    limite = time.monotonic() + segundos
    while time.monotonic() < limite:
        encontrados = trabajos_de_correo(nucleo)
        if encontrados:
            return encontrados[0]
        time.sleep(0.5)
    return None


def hay_ollama(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=3) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


# --------------------------------------------------------------------------- #
# Comprobaciones que no necesitan el nucleo levantado
# --------------------------------------------------------------------------- #


def comprobar_en_proceso() -> None:
    print("--- sin levantar el nucleo ---\n")

    # 1. El titular solo cuenta. Es la regla del canal, y se comprueba sobre el
    #    recuento porque es lo unico que se le pasa.
    recuento = triaje.recontar(
        [
            Clasificacion(REQUIERE_ACCION, "hay que contestar"),
            Clasificacion(IGNORAR, "publicidad"),
        ]
    )
    texto = correo.titular(recuento) or ""
    comprobar("El titular dice cuantos y de que tipo", "1 requiere acción" in texto, texto)
    comprobar("Y cuenta el total", texto.startswith("2 correos"), texto)

    # 2. Si no hay nada relevante no se molesta. Un canal que avisa de nada se
    #    silencia, y entonces tampoco avisa de lo que importa.
    solo_basura = triaje.recontar([Clasificacion(IGNORAR, "publicidad")])
    comprobar("Sin nada relevante no hay titular", correo.titular(solo_basura) is None)

    # 3. `no_seguro` cuenta como relevante: ante la duda, que lo mire una persona.
    dudoso = triaje.recontar([Clasificacion(NO_SEGURO, "no lo tengo claro")])
    comprobar("Un correo sin decidir tambien avisa", correo.titular(dudoso) is not None)

    # 4. Sin modelo local, el triaje escala en vez de descartar. Es la diferencia
    #    entre perder un correo y mirar uno de mas.
    entorno = dict(os.environ)
    with tempfile.TemporaryDirectory(prefix="perseo_triaje_") as tmp:
        os.environ["PERSEO_CORE_DATOS"] = tmp
        os.environ["PERSEO_OLLAMA"] = "http://127.0.0.1:1"  # nadie escucha ahi
        os.environ.pop("PERSEO_CORREO", None)
        cfg = almacen.cargar_configuracion()
    os.environ.clear()
    os.environ.update(entorno)

    async def sin_modelo() -> Clasificacion:
        clasificador = triaje.Triaje(cfg)
        try:
            return await clasificador.clasificar(NUEVO)
        finally:
            await clasificador.cerrar()

    clasificacion = asyncio.run(sin_modelo())
    comprobar(
        "Sin modelo local se escala a no_seguro",
        clasificacion.clase == NO_SEGURO,
        clasificacion.clase,
    )
    comprobar("Y queda anotado que no lo decidio el modelo", clasificacion.del_modelo is False)

    # 5. El disparador sin buzon configurado se retira solo, como Telegram sin
    #    token. No configurado no es lo mismo que roto.
    async def sin_buzon() -> str:
        ctx = disparadores.Contexto(cfg=cfg, bus=Bus())
        try:
            await correo._vigilar_buzon(ctx)
        except disparadores.Retirarse as motivo:
            return str(motivo)
        return ""

    comprobar(
        "Sin buzon configurado el disparador se retira",
        "buzón" in asyncio.run(sin_buzon()),
    )
    print()


# --------------------------------------------------------------------------- #
# De punta a punta
# --------------------------------------------------------------------------- #


def main() -> None:
    comprobar_en_proceso()

    url_ollama = os.environ.get("PERSEO_OLLAMA", "http://127.0.0.1:11434")
    con_modelo = hay_ollama(url_ollama)
    if not con_modelo:
        print(f"AVISO: Ollama no responde en {url_ollama}. Se salta lo que depende del modelo.\n")

    falso = FalsoTelegram()
    falso.arrancar()

    buzon = Path(tempfile.mkdtemp(prefix="perseo_buzon_")) / "buzon.json"
    escribir_buzon(buzon, VIEJOS)

    nucleo = Nucleo(
        {
            "PERSEO_CORREO": "falso",
            "PERSEO_CORREO_FALSO": str(buzon),
            "PERSEO_CORREO_INTERVALO": INTERVALO,
            "PERSEO_TELEGRAM_TOKEN": TOKEN_FALSO,
            "PERSEO_TELEGRAM_CHAT": CHAT,
            "PERSEO_TELEGRAM_API": falso.url,
            "PERSEO_URL_BASE": "http://perseo-de-prueba:8787",
        }
    )
    nucleo.arrancar()

    # 6. Estreno: un buzon con correo viejo dentro no se tria. Sin esto, el dia
    #    que se enciende el disparador llegan veinte clasificaciones de cosas ya
    #    leidas y se desactiva.
    time.sleep(float(INTERVALO) * 2)
    comprobar(
        "El buzon que ya estaba lleno no se tria al estrenar",
        trabajos_de_correo(nucleo) == [],
        f"{len(trabajos_de_correo(nucleo))} trabajo(s)",
    )

    # 7. Un correo que llega **despues** si dispara: se encola, y como
    #    disparador, no como peticion.
    escribir_buzon(buzon, VIEJOS + [NUEVO])
    trabajo = esperar_trabajo_de_correo(nucleo)
    comprobar("Un correo nuevo encola un trabajo", trabajo is not None)
    if trabajo is None:
        nucleo.volcar()
        nucleo.limpiar()
        falso.parar()
        resumir()
        return

    comprobar("Y nace con origen de disparador", trabajo.get("origen") == "disparador", str(trabajo.get("origen")))

    id_trabajo = int(trabajo["id"])
    terminado = nucleo.esperar_estado(id_trabajo, ("hecho", "fallido"), intentos=180)
    comprobar(
        "El triaje termina bien",
        terminado.get("estado") == "hecho",
        f"estado={terminado.get('estado')} error={terminado.get('error')}",
    )

    resultado = terminado.get("resultado") or {}
    clasificados = resultado.get("clasificados") or []
    comprobar("Se tria solo el correo nuevo", len(clasificados) == 1, f"{len(clasificados)}")
    if clasificados:
        clase = clasificados[0].get("clase")
        comprobar("Con una clase del vocabulario", clase in CLASES, str(clase))
        if con_modelo:
            comprobar(
                "Y la pone el modelo local, no el respaldo",
                clasificados[0].get("del_modelo") is True,
                str(clasificados[0].get("motivo")),
            )

    # 8. La regla del canal: titular por Telegram, detalle por Tailscale.
    aviso = falso.esperar_envio("correo", segundos=20)
    comprobar("El titular llega al chat", aviso is not None, str(aviso and aviso.get("text")))
    if aviso is not None:
        crudo = json.dumps(aviso, ensure_ascii=False)
        comprobar(
            "El asunto del correo NO sale por Telegram",
            ASUNTO_SECRETO not in crudo,
        )
        comprobar(
            "El cuerpo del correo NO sale por Telegram",
            EXTRACTO_SECRETO not in crudo,
        )
        comprobar(
            "Pero si un enlace para leerlo por el tailnet",
            "perseo-de-prueba" in crudo,
        )
    # El detalle si esta en la cola, que se lee por el tailnet.
    comprobar(
        "El detalle si queda en la cola",
        any(c.get("asunto") == ASUNTO_SECRETO for c in clasificados),
    )

    # 9. La marca de agua: la vuelta siguiente no vuelve a triar lo mismo.
    time.sleep(float(INTERVALO) * 3)
    comprobar(
        "No se tria dos veces el mismo correo",
        len(trabajos_de_correo(nucleo)) == 1,
        f"{len(trabajos_de_correo(nucleo))} trabajo(s) de correo",
    )

    # 10. Y sobrevive a un reinicio: la marca de agua vive en disco. Si viviera
    #     en memoria, reiniciar volveria a avisar del buzon entero.
    nucleo.reiniciar()
    time.sleep(float(INTERVALO) * 3)
    comprobar(
        "Reiniciar no vuelve a triar lo ya visto",
        len(trabajos_de_correo(nucleo)) == 1,
        f"{len(trabajos_de_correo(nucleo))} trabajo(s) de correo",
    )

    nucleo.limpiar()
    falso.parar()
    resumir()


if __name__ == "__main__":
    main()
