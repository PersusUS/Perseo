"""Verificación del canal de Telegram, sin usar Telegram.

Se levanta un servidor de mentira que habla el mismo protocolo —`getUpdates`,
`sendMessage`, `answerCallbackQuery`, `editMessageReplyMarkup`— y se apunta el
núcleo a él con `PERSEO_TELEGRAM_API`. Así se puede comprobar lo que de verdad
importa antes de tener un bot: que el titular sale, que **el detalle no**, que
el botón resuelve el trabajo, y que una pulsación de otro chat no resuelve nada.

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
CHAT_INTRUSO = "9999"

#: Tope de espera de un `getUpdates` sin novedades. El núcleo pide 25 segundos;
#: aquí se corta antes para que el script no tarde una eternidad en cerrar.
ESPERA_MAXIMA = 1.5


class FalsoTelegram:
    """Servidor mínimo que imita la API de bots de Telegram."""

    def __init__(self) -> None:
        self.enviados: list[dict[str, Any]] = []
        self.respuestas: list[dict[str, Any]] = []
        self.ediciones: list[dict[str, Any]] = []
        self._pendientes: list[dict[str, Any]] = []
        self._siguiente_id = 1
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

    # -- lo que hace el "usuario" ------------------------------------------ #

    def pulsar(self, decision: str, id_trabajo: int, chat: str = CHAT) -> None:
        """Simula que alguien pulsa un botón del mensaje."""
        with self._cerrojo:
            self._pendientes.append(
                {
                    "update_id": self._siguiente_id,
                    "callback_query": {
                        "id": f"cb{self._siguiente_id}",
                        "data": f"{decision}:{id_trabajo}",
                        "message": {"message_id": 500 + self._siguiente_id, "chat": {"id": chat}},
                    },
                }
            )
            self._siguiente_id += 1

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

    def esperar_respuesta(self, segundos: float = 8) -> dict[str, Any] | None:
        limite = time.monotonic() + segundos
        while time.monotonic() < limite:
            with self._cerrojo:
                if self.respuestas:
                    return self.respuestas[-1]
            time.sleep(0.2)
        return None

    # -- el servidor ------------------------------------------------------- #

    def _manejador(self):
        falso = self

        class Manejador(BaseHTTPRequestHandler):
            def log_message(self, *_: Any) -> None:
                pass  # sin ruido en la salida del script

            def handle_error(self, *_: Any) -> None:
                # Cerrar el nucleo corta los `getUpdates` a medias, y el servidor
                # de la biblioteca estandar volcaria un rastro de pila por cada
                # uno. No es un fallo de nada: es el cliente que se ha ido.
                pass

            def do_POST(self) -> None:  # noqa: N802  (lo exige BaseHTTPRequestHandler)
                largo = int(self.headers.get("Content-Length", "0"))
                try:
                    carga = json.loads(self.rfile.read(largo) or "{}")
                except json.JSONDecodeError:
                    carga = {}

                partes = self.path.strip("/").split("/")
                metodo = partes[-1] if partes else ""
                resultado = falso._despachar(metodo, carga)
                cuerpo = json.dumps({"ok": True, "result": resultado}).encode()

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(cuerpo)))
                self.end_headers()
                self.wfile.write(cuerpo)

        return Manejador

    def _despachar(self, metodo: str, carga: dict[str, Any]) -> Any:
        if metodo == "getUpdates":
            return self._entregar_actualizaciones(carga.get("offset"))
        if metodo == "sendMessage":
            with self._cerrojo:
                self.enviados.append(carga)
            return {"message_id": 900 + len(self.enviados)}
        if metodo == "answerCallbackQuery":
            with self._cerrojo:
                self.respuestas.append(carga)
            return True
        if metodo == "editMessageReplyMarkup":
            with self._cerrojo:
                self.ediciones.append(carga)
            return True
        return True

    def _entregar_actualizaciones(self, offset: int | None) -> list[dict[str, Any]]:
        limite = time.monotonic() + ESPERA_MAXIMA
        while True:
            with self._cerrojo:
                listas = [
                    a for a in self._pendientes if offset is None or a["update_id"] >= offset
                ]
                if listas:
                    return listas
            if time.monotonic() >= limite:
                return []
            time.sleep(0.1)


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

    # 1. Un trabajo que pide confirmacion llega al chat con sus botones.
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

    botones = (aviso.get("reply_markup") or {}).get("inline_keyboard") or []
    planos = [b for fila in botones for b in fila]
    comprobar(
        "Trae los botones de aprobar y rechazar",
        {f"aprobar:{id_uno}", f"rechazar:{id_uno}"}
        <= {b.get("callback_data") for b in planos},
        str([b.get("text") for b in planos]),
    )
    comprobar(
        "Y un enlace al detalle, que apunta fuera del bucle local",
        any(b.get("url", "").startswith("http://perseo-de-prueba:8787") for b in planos),
        str([b.get("url") for b in planos if b.get("url")]),
    )

    # 2. Una pulsacion de otro chat no resuelve nada. Un bot es publico: sin
    #    este filtro, cualquiera aprobaria acciones irreversibles.
    falso.pulsar("aprobar", id_uno, chat=CHAT_INTRUSO)
    time.sleep(2.5)
    _, sigue = nucleo.pedir(f"/trabajos/{id_uno}", token)
    comprobar(
        "Una pulsacion de otro chat no resuelve el trabajo",
        sigue.get("estado") == "esperando",
        str(sigue.get("estado")),
    )
    respuesta = falso.esperar_respuesta()
    comprobar(
        "Y al intruso se le contesta que no esta autorizado",
        (respuesta or {}).get("text") == "No autorizado",
        str((respuesta or {}).get("text")),
    )

    # 3. La pulsacion buena si lo resuelve, y el trabajo termina.
    falso.pulsar("aprobar", id_uno)
    hecho = nucleo.esperar_estado(id_uno, ("hecho", "fallido"), intentos=60)
    comprobar("El boton aprueba el trabajo", hecho.get("estado") == "hecho", str(hecho.get("estado")))
    comprobar(
        "La decision queda registrada como del usuario",
        (hecho.get("confirmacion") or {}).get("decision") == "aprobado",
        str((hecho.get("confirmacion") or {}).get("decision")),
    )
    comprobar(
        "Se quitan los botones del mensaje ya resuelto",
        any(not (e.get("reply_markup") or {}).get("inline_keyboard") for e in falso.ediciones),
        f"{len(falso.ediciones)} edicion(es)",
    )
    # Y lo que NO se manda, que desde el 2026-08-22 es tan parte del canal como
    # lo que sí: este trabajo lo encoló uno mismo por HTTP —origen `texto`—, así
    # que al terminar no se avisa al móvil. Si estás encolando trabajos, estás
    # mirando la pantalla; el aviso sería contarte lo que ya ves. Lo que sí se
    # anuncia es lo que hicieron los disparadores solos, y solo si el agente
    # supo resumirlo: eso lo fijan las pruebas de `redactar`
    # (pruebas/test_telegram_redaccion.py).
    comprobar(
        "Un trabajo que encolaste tú no se anuncia al terminar",
        falso.esperar_envio("terminado", segundos=3) is None,
    )

    # 4. Volver a pulsar no lo ejecuta otra vez.
    antes = len(falso.respuestas)
    falso.pulsar("aprobar", id_uno)
    limite = time.monotonic() + 8
    ultima: dict[str, Any] = {}
    while time.monotonic() < limite:
        if len(falso.respuestas) > antes:
            ultima = falso.respuestas[-1]
            break
        time.sleep(0.2)
    comprobar(
        "Pulsar dos veces avisa de que ya estaba resuelto",
        ultima.get("text") == "Ese trabajo ya estaba resuelto",
        str(ultima.get("text")),
    )

    # 5. Rechazar desde el movil cierra el trabajo sin ejecutarlo.
    id_dos = encolar_simulacro(nucleo, "vaciar la papelera", "sin vuelta atras")
    nucleo.esperar_estado(id_dos, ("esperando",))
    falso.pulsar("rechazar", id_dos)
    rechazado = nucleo.esperar_estado(id_dos, ("rechazado", "hecho"), intentos=60)
    comprobar(
        "El boton de rechazar cierra el trabajo",
        rechazado.get("estado") == "rechazado" and rechazado.get("resultado") is None,
        f"estado={rechazado.get('estado')} resultado={rechazado.get('resultado')}",
    )

    nucleo.limpiar()
    falso.parar()
    resumir()


if __name__ == "__main__":
    main()
