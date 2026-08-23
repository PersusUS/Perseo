"""Verificación del canal de Telegram, sin usar Telegram.

Se levanta un servidor de mentira que habla el mismo protocolo —`sendMessage`
basta desde que el canal quedó en una sola dirección— y se apunta el núcleo a él
con `PERSEO_TELEGRAM_API`. Así se puede comprobar lo que de verdad importa antes
de tener un bot: que el titular sale, que **el detalle no**, que ningún mensaje
lleva botones de decisión, y que la confirmación se resuelve por la vía que
queda de verdad —la web— sin que el móvil se entere de más.

Lo único que este script no puede probar es que la API real se comporte como
está documentada. Todo lo demás es el código que se va a ejecutar.

    python perseo_core/verificar_telegram.py
"""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perseo_core.arnes_pruebas import Nucleo, comprobar, puerto_libre, resumir  # noqa: E402

TOKEN_FALSO = "111:prueba"
CHAT = "4242"


class FalsoTelegram:
    """Servidor mínimo que imita la parte de la API que aún se usa: enviar."""

    def __init__(self) -> None:
        self.enviados: list[dict[str, Any]] = []
        self._cerrojo = threading.Lock()
        self.puerto = puerto_libre()
        self._servidor = ThreadingHTTPServer(("127.0.0.1", self.puerto), self._manejador())
        self._servidor.daemon_threads = True

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.puerto}"

    def arrancar(self) -> None:
        threading.Thread(target=self._servidor.serve_forever, daemon=True).start()

    def parar(self) -> None:
        self._servidor.shutdown()

    def esperar_envio(self, contiene: str, segundos: float = 8) -> dict[str, Any] | None:
        """Espera a que llegue un `sendMessage` cuyo texto contenga eso."""
        limite = time.monotonic() + segundos
        while time.monotonic() < limite:
            with self._cerrojo:
                for enviado in self.enviados:
                    if contiene in enviado.get("text", ""):
                        return enviado
            time.sleep(0.2)
        return None

    # -- el servidor ------------------------------------------------------- #

    def _manejador(self):
        falso = self

        class Manejador(BaseHTTPRequestHandler):
            def log_message(self, *_: Any) -> None:
                pass  # sin ruido en la salida del script

            def do_POST(self) -> None:  # noqa: N802  (lo exige BaseHTTPRequestHandler)
                largo = int(self.headers.get("Content-Length", "0"))
                try:
                    carga = json.loads(self.rfile.read(largo) or "{}")
                except json.JSONDecodeError:
                    carga = {}

                partes = self.path.strip("/").split("/")
                metodo = partes[-1] if partes else ""
                if metodo == "sendMessage":
                    with falso._cerrojo:
                        falso.enviados.append(carga)
                    resultado = {"message_id": 900 + len(falso.enviados)}
                else:
                    # El canal ya no lee nada: si el núcleo pide aquí otra cosa
                    # —un getUpdates, un answerCallbackQuery— es que ha vuelto
                    # código que se quitó a propósito con N-1.
                    resultado = None

                cuerpo = json.dumps({"ok": True, "result": resultado}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(cuerpo)))
                self.end_headers()
                self.wfile.write(cuerpo)

        return Manejador


def encolar_simulacro(nucleo: Nucleo, accion: str, detalle: str) -> int:
    _, trabajo = nucleo.pedir(
        "/trabajos",
        nucleo.token,
        "POST",
        {"agente": "simulacro", "peticion": {"accion": accion, "detalle": detalle}},
    )
    return int(trabajo["id"])


def main() -> None:
    falso = FalsoTelegram()
    falso.arrancar()
    print(f"Telegram de mentira en {falso.url}\n")

    nucleo = Nucleo(
        {
            "PERSEO_TELEGRAM_TOKEN": TOKEN_FALSO,
            "PERSEO_TELEGRAM_CHAT": CHAT,
            "PERSEO_TELEGRAM_API": falso.url,
            "PERSEO_URL_BASE": "http://perseo-de-prueba:8787",
        }
    )
    nucleo.arrancar()
    token = nucleo.token

    # 1. Un trabajo que pide confirmacion llega al chat, con la pregunta y sin
    #    detalle. La decision ya no se toma en el chat: se anuncia, y punto.
    secreto = "37 ficheros, entre ellos las facturas de marzo"
    id_uno = encolar_simulacro(nucleo, "borrar la carpeta de descargas", secreto)
    aviso = falso.esperar_envio("Perseo espera un sí")
    comprobar("El aviso de confirmacion llega al chat", aviso is not None)
    if aviso is None:
        nucleo.limpiar()
        falso.parar()
        resumir()
        return

    comprobar("Va al chat configurado", str(aviso.get("chat_id")) == CHAT, str(aviso.get("chat_id")))
    comprobar(
        "Lleva el resumen de la pregunta",
        "borrar la carpeta de descargas" in aviso["text"],
        aviso["text"].replace("\n", " | "),
    )
    comprobar(f"Lleva el numero de trabajo (#{id_uno})", f"#{id_uno}" in aviso["text"])

    # Lo importante del canal: titular por Telegram, detalle por Tailscale.
    comprobar(
        "El detalle NO sale por Telegram",
        secreto not in json.dumps(aviso, ensure_ascii=False),
        "el texto del detalle no aparece en el mensaje",
    )

    # 2. Ningun boton de decision. Se quitaron con N-1: quien decide esta en la
    #    llamada —de viva voz— o en las pantallas. Volver a ver aqui un
    #    callback_data significa que ha vuelto la aprobacion por Telegram.
    planos = [
        b
        for enviado in list(falso.enviados)
        for fila in ((enviado.get("reply_markup") or {}).get("inline_keyboard") or [])
        for b in fila
    ]
    comprobar(
        "Ningun mensaje lleva botones de aprobar o rechazar",
        not any("callback_data" in b for b in planos),
        str([b.get("text") for b in planos]),
    )
    comprobar(
        "Y el enlace al detalle apunta fuera del bucle local",
        any(str(b.get("url", "")).startswith("http://perseo-de-prueba:8787") for b in planos),
        str([b.get("url") for b in planos if b.get("url")]),
    )

    # 3. La confirmacion se resuelve por la web, que es la via que queda. Y lo
    #    que NO se manda despues importa tanto como lo que se manda: este
    #    trabajo lo encolo uno mismo por HTTP —origen `texto`—, asi que al
    #    terminar no se avisa al movil. Si estas encolando trabajos, estas
    #    mirando la pantalla. Lo que si se anuncia es lo que hicieron los
    #    disparadores solos, y solo si el agente supo resumirlo: eso lo fijan
    #    las pruebas de `redactar` (pruebas/test_telegram_redaccion.py).
    nucleo.pedir(f"/trabajos/{id_uno}/aprobar", token, "POST", {})
    hecho = nucleo.esperar_estado(id_uno, ("hecho", "fallido"), intentos=60)
    comprobar("La web aprueba el trabajo", hecho.get("estado") == "hecho", str(hecho.get("estado")))
    comprobar(
        "La decision queda registrada como del usuario",
        (hecho.get("confirmacion") or {}).get("decision") == "aprobado",
        str((hecho.get("confirmacion") or {}).get("decision")),
    )
    comprobar(
        "Un trabajo que encolaste tu no se anuncia al terminar",
        falso.esperar_envio("terminado", segundos=3) is None,
    )

    # 4. Rechazar desde la web cierra el trabajo sin ejecutarlo, y tampoco suena.
    id_dos = encolar_simulacro(nucleo, "vaciar la papelera", "sin vuelta atras")
    nucleo.esperar_estado(id_dos, ("esperando",))
    nucleo.pedir(f"/trabajos/{id_dos}/rechazar", token, "POST", {})
    rechazado = nucleo.esperar_estado(id_dos, ("rechazado", "hecho"), intentos=60)
    comprobar(
        "La web rechaza y el trabajo se cierra sin ejecutarse",
        rechazado.get("estado") == "rechazado" and rechazado.get("resultado") is None,
        f"estado={rechazado.get('estado')} resultado={rechazado.get('resultado')}",
    )

    nucleo.limpiar()
    falso.parar()
    resumir()


if __name__ == "__main__":
    main()
