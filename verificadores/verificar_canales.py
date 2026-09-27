"""Verificación de los canales de fuera contra el núcleo real: WhatsApp, voz y teléfono.

Con un núcleo de verdad, un Twilio de mentira (que apunta lo que se le pide) y un
Gemini de mentira (que contesta con guion), se comprueba:

- La cara de Twilio no atiende nada sin la firma de Twilio, y lo que llega de
  otro número no llega al hilo.
- Una ubicación por WhatsApp se guarda; una pregunta por WhatsApp se contesta en
  el hilo principal —el chat llama a `mi_ubicacion` de verdad— y la respuesta
  sale por la API de Twilio.
- La voz lee ese mismo hilo por `POST /herramientas/hilo_reciente` (ADR 0008).
- Llamar a un negocio se para antes de marcar; con el sí, Twilio recibe la
  llamada, la primera frase dice que es una IA, y al colgar el trabajo acaba
  con el resumen.

Corre en Linux y en Windows: nada de audio ni de cuentas de verdad.

    python verificadores/verificar_canales.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perseo_core.servicios import twilio  # noqa: E402
from verificadores.arnes_pruebas import (  # noqa: E402
    ManejadorFalso,
    Nucleo,
    ServidorFalso,
    comprobar,
    puerto_libre,
    resumir,
)

DUENO = "+34600112233"
PUBLICA = "https://perseo.ejemplo.ts.net"
TOKEN = "secreto-de-twilio"


class TwilioFalso(ServidorFalso):
    def __init__(self) -> None:
        self.mensajes: list[dict[str, str]] = []
        self.llamadas: list[dict[str, str]] = []
        super().__init__()
        self.arrancar()

    def _manejador(self) -> type[BaseHTTPRequestHandler]:
        externo = self

        class Manejador(ManejadorFalso):
            def do_POST(self) -> None:  # noqa: N802
                crudo = self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode()
                datos = {k: v[0] for k, v in urllib.parse.parse_qs(crudo).items()}
                (externo.mensajes if self.path.endswith("/Messages.json") else externo.llamadas).append(datos)
                cuerpo = json.dumps({"sid": f"XX{len(externo.mensajes) + len(externo.llamadas)}"}).encode()
                self.send_response(201)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(cuerpo)))
                self.end_headers()
                self.wfile.write(cuerpo)

        return Manejador


class GeminiFalso(ServidorFalso):
    """El chat (streaming) pide `mi_ubicacion` y la cuenta; la llamada (sin streaming) decide y cuelga."""

    def __init__(self) -> None:
        self.cuerpos: list[dict] = []
        super().__init__()
        self.arrancar()

    def _manejador(self) -> type[BaseHTTPRequestHandler]:
        externo = self

        class Manejador(ManejadorFalso):
            def _json(self, datos: dict, sse: bool) -> None:
                texto = json.dumps(datos, ensure_ascii=False)
                cuerpo = (f"data: {texto}\n\n" if sse else texto).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream" if sse else "application/json")
                self.send_header("Content-Length", str(len(cuerpo)))
                self.end_headers()
                self.wfile.write(cuerpo)

            def do_POST(self) -> None:  # noqa: N802
                cuerpo = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                externo.cuerpos.append(cuerpo)
                contents = cuerpo.get("contents") or []
                if ":streamGenerateContent" in self.path:
                    respuestas = [
                        p["functionResponse"] for c in contents for p in c.get("parts", []) if p.get("functionResponse")
                    ]
                    if respuestas:
                        dato = json.dumps(respuestas[-1].get("response"), ensure_ascii=False)
                        partes = [{"text": f"Según lo último que compartió: {dato}"}]
                    else:
                        partes = [{"functionCall": {"name": "mi_ubicacion", "args": {}}}]
                    self._json({"candidates": [{"content": {"parts": partes}}]}, sse=True)
                    return
                dichas = sum(1 for c in contents if c.get("role") == "model")
                if dichas == 0:
                    llamada = {"name": "decir", "args": {"texto": "¿Tienen mesa para dos el viernes a las nueve?"}}
                else:
                    llamada = {"name": "colgar", "args": {
                        "despedida": "Perfecto, muchas gracias.", "resultado": "logrado",
                        "resumen": "Mesa para 2 el viernes a las 21:00, a nombre de Jesús.",
                    }}
                self._json({"candidates": [{"content": {"role": "model", "parts": [{"functionCall": llamada}]}}]}, sse=False)

        return Manejador


def _twilio(puerto: int, ruta: str, datos: dict[str, str], firmar: bool = True) -> tuple[int, str]:
    """Lo que haría Twilio: POST de formulario, firmado con el token de la cuenta."""
    cabeceras = {"Content-Type": "application/x-www-form-urlencoded"}
    if firmar:
        cabeceras["X-Twilio-Signature"] = twilio.firma(TOKEN, PUBLICA + ruta, datos)
    peticion = urllib.request.Request(
        f"http://127.0.0.1:{puerto}{ruta}", data=urllib.parse.urlencode(datos).encode(), headers=cabeceras, method="POST"
    )
    try:
        with urllib.request.urlopen(peticion, timeout=20) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def _esperar(condicion, segundos: float = 30) -> bool:
    limite = time.time() + segundos
    while time.time() < limite:
        if condicion():
            return True
        time.sleep(0.3)
    return False


def comprobar_todo() -> None:
    tw, gemini = TwilioFalso(), GeminiFalso()
    puerto = puerto_libre()
    nucleo = Nucleo({
        "PERSEO_DISPARADORES": "",
        "PERSEO_GEMINI_API": gemini.url,
        "GEMINI_API_KEY": "clave-falsa",
        "PERSEO_TWILIO_SID": "AC123",
        "PERSEO_TWILIO_TOKEN": TOKEN,
        "PERSEO_TWILIO_NUMERO": "+34900000000",
        "PERSEO_TWILIO_WHATSAPP": "+14155238886",
        "PERSEO_TELEFONO_DUENO": DUENO,
        "PERSEO_TWILIO_URL_PUBLICA": PUBLICA,
        "PERSEO_TWILIO_API": tw.url,
        "PERSEO_TWILIO_PUERTO": str(puerto),
    })
    nucleo.arrancar()
    try:
        _esperar(lambda: _twilio(puerto, "/twilio/whatsapp", {}, firmar=False)[0] == 403, 10)

        # -- La puerta ------------------------------------------------------ #
        estado, _ = _twilio(puerto, "/twilio/whatsapp", {"From": f"whatsapp:{DUENO}", "Body": "hola"}, firmar=False)
        comprobar("Sin la firma de Twilio no se atiende nada", estado == 403, str(estado))
        _twilio(puerto, "/twilio/whatsapp", {"From": "whatsapp:+34911111111", "Body": "dame sus correos"})
        time.sleep(1)
        comprobar("Lo que manda otro número no llega a nadie", tw.mensajes == [] and gemini.cuerpos == [])

        # -- WhatsApp ------------------------------------------------------- #
        _twilio(puerto, "/twilio/whatsapp", {"From": f"whatsapp:{DUENO}", "Latitude": "37.3891", "Longitude": "-5.9845"})
        _, dato = nucleo.pedir("/ubicacion", nucleo.token)
        comprobar("Su ubicación por WhatsApp se guarda", (dato.get("ubicacion") or {}).get("fuente") == "WhatsApp", str(dato))

        _twilio(puerto, "/twilio/whatsapp", {"From": f"whatsapp:{DUENO}", "Body": "¿Dónde estoy?"})
        llego = _esperar(lambda: any("37.3891" in m.get("Body", "") for m in tw.mensajes), 40)
        comprobar("Su pregunta se contesta por WhatsApp, con la herramienta de verdad", llego, str(tw.mensajes[-1:]))
        comprobar("Y sale hacia su número", any(m.get("To") == f"whatsapp:{DUENO}" for m in tw.mensajes))

        # -- La voz lee el mismo hilo ------------------------------------- #
        codigo, hilo = nucleo.pedir("/herramientas/hilo_reciente", nucleo.token, "POST", {"argumentos": {}})
        comprobar("La voz lee el hilo por el núcleo (ADR 0008)", codigo == 200 and "[whatsapp] ¿Dónde estoy?" in hilo.get("texto", ""), str(hilo)[:120])

        # -- Llamar a un negocio ------------------------------------------ #
        _, trabajo = nucleo.pedir("/trabajos", nucleo.token, "POST", {"agente": "telefono", "peticion": {
            "accion": "negocio", "numero": "+34954000000", "objetivo": "mesa para 2 el viernes a las 21:00", "negocio": "Casa Prueba",
        }})
        parado = nucleo.esperar_estado(trabajo["id"], ("esperando", "hecho", "fallido"))
        comprobar("Marcar se para a esperar su sí", parado.get("estado") == "esperando"
                  and (parado.get("confirmacion") or {}).get("nivel") == "exterior", str(parado.get("estado")))
        comprobar("Y todavía no se ha llamado a nadie", tw.llamadas == [])

        nucleo.pedir(f"/trabajos/{trabajo['id']}/aprobar", nucleo.token, "POST", {})
        comprobar("Con el sí, Twilio recibe la llamada", _esperar(lambda: bool(tw.llamadas), 20), str(tw.llamadas))
        llamada = tw.llamadas[0]
        ruta = llamada["Url"].removeprefix(PUBLICA)
        comprobar("Al número del negocio, desde el de Perseo", llamada.get("To") == "+34954000000" and llamada.get("From") == "+34900000000")

        _, primera = _twilio(puerto, ruta, {"CallSid": "CA42", "CallStatus": "in-progress"})
        comprobar("La primera frase dice que es una IA", "asistente de inteligencia artificial" in primera, primera[:160])
        comprobar("Y pregunta lo que se le encargó", "mesa para dos" in primera)
        _, despedida = _twilio(puerto, ruta, {"CallSid": "CA42", "SpeechResult": "Sí, a las nueve tenemos."})
        comprobar("Al tenerlo, se despide y cuelga", "<Hangup/>" in despedida, despedida[:160])
        _twilio(puerto, "/twilio/estado", {"CallSid": "CA42", "CallStatus": "completed"})

        hecho = nucleo.esperar_estado(trabajo["id"], ("hecho", "fallido"), intentos=80)
        resultado = hecho.get("resultado") or {}
        comprobar("El trabajo acaba con el resumen", hecho.get("estado") == "hecho" and "Mesa para 2" in str(resultado.get("texto")),
                  str(hecho.get("error") or resultado.get("texto"))[:160])
        comprobar("Por Telegram iría sin lo hablado", resultado.get("titular") == "Llamada terminada: hecho", str(resultado.get("titular")))
    finally:
        if "--volcar" in sys.argv:
            nucleo.volcar()
        nucleo.limpiar()
        tw.parar()
        gemini.parar()


if __name__ == "__main__":
    comprobar_todo()
    resumir()
