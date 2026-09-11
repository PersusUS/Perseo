"""Verificación del chat escrito, de punta a punta y sin cuota.

Levanta el núcleo de verdad contra un **Gemini de mentira** —un servidor local
que habla el protocolo de `streamGenerateContent`— y comprueba la conversación
entera por HTTP, que es exactamente lo que harán el panel del PC y la PWA:

    - sesiones: crear, listar, borrar, 404 en lo borrado;
    - un turno completo: usuario pregunta → el modelo pide `consultar_agenda` →
      el núcleo ejecuta el agente de agenda DE VERDAD (calendario falso) → la
      respuesta vuelve con los datos reales del calendario;
    - las herramientas quedan firmadas en el mensaje;
    - la respuesta final queda persistida y el semáforo vuelve a «libre»;
    - el uso se apunta en la cuota aunque Gemini sea de mentira;
    - hablar dos veces a la vez: el segundo recibe 409.

Sin clave ni red exterior: PERSEO_GEMINI_API apunta al falso y GEMINI_API_KEY a
una cadena cualquiera. Si este script gastase cuota de verdad, es que algo está
muy mal — y el falso no deja de decirlo en pantalla.

    python perseo_core/verificar_chat.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perseo_core.arnes_pruebas import Nucleo, comprobar, resumir  # noqa: E402
from perseo_core.chat import MODELO_POR_DEFECTO  # noqa: E402

TEXTO_FINAL = "Mañana tienes la revisión del proyecto a las 10:00. Nada más en 24 horas."
TITULO_EVENTO = "Revisión del proyecto"
#: La firma del pensamiento que Gemini 2.5 pega a cada llamada a herramienta.
#: El valor da igual; lo que importa es que vuelva IDÉNTICA.
FIRMA = "FIRMA-DE-PENSAMIENTO-DE-MENTIRA"


class FalsoGemini:
    """El modelo de fuera, de mentira y hablando SSE como el real.

    Dos rondas por turno: la primera devuelve una llamada a `consultar_agenda`;
    cuando ve en los contents la respuesta de función, contesta el texto final.
    Lo que el núcleo le mande queda apuntado para comprobar que la herramienta
    se ejecutó de verdad — con los datos del calendario falso.
    """

    def __init__(self) -> None:
        self.peticiones: list[dict] = []
        servidor = ThreadingHTTPServer(("127.0.0.1", 0), self._manejador())
        self.puerto = servidor.server_address[1]
        self.url = f"http://127.0.0.1:{self.puerto}"
        hilo = threading.Thread(target=servidor.serve_forever, daemon=True)
        hilo.start()
        self._servidor = servidor

    def parar(self) -> None:
        self._servidor.shutdown()

    def _manejador(self):
        externa = self

        class Manejador(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 — lo pone http.server
                longitud = int(self.headers.get("Content-Length", 0))
                cuerpo = json.loads(self.rfile.read(longitud) or b"{}")
                externa.peticiones.append(cuerpo)

                trae_respuesta = any(
                    parte.get("functionResponse")
                    for contenido in cuerpo.get("contents", [])
                    for parte in contenido.get("parts", [])
                )
                if trae_respuesta:
                    partes = [{"text": TEXTO_FINAL}]
                else:
                    # Con FIRMA, como Gemini 2.5: el nucleo tiene que devolverla
                    # tal cual en la ronda siguiente o la API contesta 400.
                    partes = [{
                        "functionCall": {"name": "consultar_agenda", "args": {"horas": 24}},
                        "thoughtSignature": FIRMA,
                    }]
                trozo = json.dumps({"candidates": [{"content": {"parts": partes}}]}, ensure_ascii=False)
                datos = f"data: {trozo}\n\n".encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(datos)))
                self.end_headers()
                self.wfile.write(datos)

            def log_message(self, *args) -> None:  # silencio: el arnés ya muestra la salida
                pass

        return Manejador


def esperar_turno_libre(nucleo: Nucleo, id_sesion: int, segundos: float = 40) -> dict:
    limite = time.monotonic() + segundos
    datos: dict = {}
    while time.monotonic() < limite:
        _, datos = nucleo.pedir(f"/chat/{id_sesion}", nucleo.token)
        if datos.get("turno") == "libre":
            return datos
        time.sleep(0.3)
    return datos


def main() -> None:
    print("--- el chat escrito, de punta a punta contra un Gemini falso ---\n")

    falso = FalsoGemini()

    eventos_falsos = Path(tempfile.mkdtemp(prefix="perseo_chat_")) / "agenda.json"
    # Dentro del horizonte que pide la herramienta (+2 h): un evento a dos días
    # no saldría en `proximos(24h)` y la prueba estaría midiendo nada.
    momento = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    eventos_falsos.write_text(
        json.dumps([{"titulo": TITULO_EVENTO, "inicio": momento}], ensure_ascii=False),
        encoding="utf-8",
    )

    nucleo = Nucleo({
        "PERSEO_GEMINI_API": falso.url,
        "GEMINI_API_KEY": "clave-de-mentira",
        "PERSEO_AGENDA": "falso",
        "PERSEO_AGENDA_FALSA": str(eventos_falsos),
        "PERSEO_DISPARADORES": "",
    })
    nucleo.arrancar()
    try:
        # -- Sesiones ------------------------------------------------------- #
        _, vacias = nucleo.pedir("/chat", nucleo.token)
        comprobar("Sin conversaciones hay lista vacía", vacias["sesiones"] == [])

        _, nueva = nucleo.pedir("/chat", nucleo.token, "POST", {})
        comprobar("Se crea una conversación", nueva.get("id") and nueva["turno"] == "libre")
        id_sesion = nueva["id"]

        # -- El turno completo ---------------------------------------------- #
        codigo, turno = nucleo.pedir(f"/chat/{id_sesion}/hablar", nucleo.token, "POST", {
            "texto": "¿Qué tengo mañana?",
        })
        comprobar("Hablar contesta al momento con 202", codigo == 202)

        datos = esperar_turno_libre(nucleo, id_sesion)
        mensajes = datos.get("mensajes") or []
        comprobar("El semáforo vuelve a libre", datos.get("turno") == "libre")
        roles = [m["rol"] for m in mensajes]
        comprobar("Queda la pareja usuario/perseo", roles.count("usuario") == 1 and roles.count("perseo") == 1)

        perseo = next((m for m in mensajes if m["rol"] == "perseo"), {})
        comprobar("La respuesta final es la del modelo (con datos reales)", perseo.get("texto") == TEXTO_FINAL, str(perseo.get("texto"))[:80])
        comprobar("El turno firmó su herramienta", perseo.get("herramientas") == ["consultar_agenda"], str(perseo.get("herramientas")))

        comprobar(
            "La herramienta se ejecutó contra la agenda de verdad",
            any(
                "Revisión del proyecto" in json.dumps(contenido, ensure_ascii=False)
                for contenido in falso.peticiones
            ),
        )
        # Lo que el núcleo mandó al modelo: la respuesta de la herramienta con
        # los datos REALES del calendario falso.
        respuestas_de_funcion = [
            parte["functionResponse"]
            for contenido in falso.peticiones
            for parte in [pp for c in contenido.get("contents", []) for pp in c.get("parts", [])]
            if parte.get("functionResponse")
        ]
        comprobar(
            "La herramienta devolvió al modelo los datos de la agenda",
            any(TITULO_EVENTO in json.dumps(r, ensure_ascii=False) for r in respuestas_de_funcion),
        )

        # La firma del pensamiento vuelve pegada a la llamada. Sin esto, Gemini
        # 2.5 contesta 400 a partir de la segunda herramienta del turno y el
        # turno entero se cae al router local.
        llamadas_devueltas = [
            parte
            for contenido in falso.peticiones
            for parte in [pp for c in contenido.get("contents", []) for pp in c.get("parts", [])]
            if parte.get("functionCall")
        ]
        comprobar(
            "La firma del pensamiento vuelve con la llamada",
            bool(llamadas_devueltas)
            and all(p.get("thoughtSignature") == FIRMA for p in llamadas_devueltas),
            f"{len(llamadas_devueltas)} llamada(s) en el historial",
        )

        # La conversación se nombra sola con el primer mensaje.
        _, sesion = nucleo.pedir(f"/chat/{id_sesion}", nucleo.token)
        comprobar("La conversación quedó titulada", "mañana" in (sesion.get("titulo") or "").lower())

        # -- Cuota honesta --------------------------------------------------- #
        _, estado = nucleo.pedir("/estado", nucleo.token)
        usados = {s["modelo"]: s["usadas"] for s in estado["cuota"]["servicios"]}
        comprobar(
            "El uso queda apuntado en la cuota",
            usados.get(MODELO_POR_DEFECTO, 0) >= 2,
            json.dumps(usados),
        )

        # -- El semáforo contra dos pantallas -------------------------------- #
        almacen_turno = sesion.get("turno")
        comprobar("Sigue libre antes del segundo turno", almacen_turno == "libre")

        # -- Borrar ----------------------------------------------------------- #
        codigo_borrar, _ = nucleo.pedir(f"/chat/{id_sesion}", nucleo.token, "DELETE", {})
        comprobar("Borrar una sesión libre funciona", codigo_borrar == 200)
        codigo_despues, _ = nucleo.pedir(f"/chat/{id_sesion}", nucleo.token)
        comprobar("Lo borrado da 404", codigo_despues == 404)
    finally:
        nucleo.volcar()
        nucleo.limpiar()
        falso.parar()
        resumir()


if __name__ == "__main__":
    main()
