"""Verificación de los recados contra el núcleo real y un navegador de verdad.

Lo que se comprueba, de punta a punta:

- Un recado entra por la cola, va a su carril y abre **Chrome de verdad** por
  `@playwright/mcp`, contra un sitio de mentira en el bucle local con un
  formulario de entrada y otro de reserva.
- La contraseña sale de la **bóveda real** (DPAPI): el sitio la recibe entera, y
  el modelo —un Gemini de mentira que lleva la cuenta de todo lo que le llega—
  no la ve ni una vez.
- Pulsar «Reservar» **se para** antes de llegar al sitio, con el nivel
  `exterior` guardado en la pregunta, y el sitio no ha recibido ninguna reserva.
- Con el sí, el recado **sigue desde el mismo botón**: una sola petición más al
  modelo, la de terminar, y la reserva llega una vez.
- Ni en la base de datos, ni en el registro, ni en lo que Playwright deja
  escrito queda la contraseña en claro.

El modelo no gasta cuota: `PERSEO_GEMINI_API` apunta a un servidor falso que
contesta con guion, leyendo los refs de la página real igual que lo haría él.

Necesita Windows (la bóveda cifra con DPAPI), Node y Google Chrome.

    python verificadores/verificar_recado.py
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verificadores.arnes_pruebas import (  # noqa: E402
    ManejadorFalso,
    Nucleo,
    ServidorFalso,
    comprobar,
    resumir,
)

CLAVE = "Cl4ve-De-Prueba!"


class Sitio(ServidorFalso):
    """Un restaurante: entrar con clave, y reservar con un botón que compromete."""

    def __init__(self) -> None:
        self.claves: list[str] = []
        self.reservas = 0
        super().__init__()

    def _manejador(self) -> type[BaseHTTPRequestHandler]:
        sitio = self

        class Manejador(ManejadorFalso):
            def _html(self, cuerpo: str, codigo: int = 200) -> None:
                datos = f'<!doctype html><html><head><meta charset="utf-8"><title>Casa Prueba</title></head><body>{cuerpo}</body></html>'.encode()
                self.send_response(codigo)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(datos)))
                self.end_headers()
                self.wfile.write(datos)

            def _formulario(self) -> dict[str, str]:
                largo = int(self.headers.get("Content-Length") or 0)
                return {k: v[0] for k, v in urllib.parse.parse_qs(self.rfile.read(largo).decode()).items()}

            def do_GET(self) -> None:  # noqa: N802
                if self.path.startswith("/entrar"):
                    self._html(
                        '<h1>Entrar</h1><form method="post" action="/entrar">'
                        '<label>Email <input name="email" type="text"></label>'
                        '<label>Contraseña <input name="clave" type="password"></label>'
                        '<button type="submit">Entrar</button></form>'
                    )
                elif self.path.startswith("/reservar"):
                    self._html(
                        '<h1>Mesa del viernes</h1><form method="post" action="/reservar">'
                        '<button type="submit">Reservar mesa para 2 · 45 €</button></form>'
                    )
                else:
                    self._html("<p>nada</p>", 404)

            def do_POST(self) -> None:  # noqa: N802
                datos = self._formulario()
                if self.path.startswith("/entrar"):
                    sitio.claves.append(datos.get("clave", ""))
                    if datos.get("clave") == CLAVE:
                        self.send_response(303)
                        self.send_header("Location", "/reservar")
                        self.end_headers()
                    else:
                        self._html("<h1>Clave incorrecta</h1>", 401)
                elif self.path.startswith("/reservar"):
                    sitio.reservas += 1
                    self._html("<h1>Reserva confirmada: ABC123</h1>")

        return Manejador


def _ref(texto: str, rol: str, nombre: str) -> str:
    """El ref del último elemento con ese rol y ese principio de nombre."""
    encontrados = re.findall(rf'{rol} "{re.escape(nombre)}[^"]*" \[ref=((?:f\d+)?e\d+)\]', texto)
    return encontrados[-1] if encontrados else "e0"


def _respuestas(contenido: dict) -> list[str]:
    return [
        str((p.get("functionResponse") or {}).get("response", {}).get("resultado", ""))
        for p in contenido.get("parts") or []
        if p.get("functionResponse")
    ]


class GeminiDeGuion(ServidorFalso):
    """Contesta como Gemini, con un guion que lee los refs de la página real."""

    def __init__(self, sitio: Sitio) -> None:
        self.sitio = sitio
        self.cuerpos: list[str] = []
        super().__init__()

    def turno(self, contents: list[dict]) -> list[tuple[str, dict]]:
        n = sum(1 for c in contents if c.get("role") == "model")
        ultima = "\n".join(_respuestas(contents[-1]))
        todas = "\n".join(t for c in contents for t in _respuestas(c))
        if n == 0:
            return [("browser_navigate", {"url": f"{self.sitio.url}/entrar"})]
        if n == 1:
            return [("boveda_listar", {})]
        if n == 2:
            return [
                ("browser_type", {"target": _ref(todas, "textbox", "Email"), "element": "email", "text": "{{boveda:casa.usuario}}"}),
                ("browser_type", {"target": _ref(todas, "textbox", "Contraseña"), "element": "clave", "text": "{{boveda:casa.clave}}"}),
            ]
        if n == 3:
            return [("browser_click", {"target": _ref(ultima, "button", "Entrar"), "element": "entrar"})]
        if n == 4:
            return [("browser_click", {"target": _ref(ultima, "button", "Reservar mesa"), "element": "el botón"})]
        codigo = "Código ABC123." if "ABC123" in ultima else "Sin código."
        return [("terminar", {"resumen": f"Reservado. {codigo}"})]

    def _manejador(self) -> type[BaseHTTPRequestHandler]:
        gemini = self

        class Manejador(ManejadorFalso):
            def do_POST(self) -> None:  # noqa: N802
                crudo = self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode()
                gemini.cuerpos.append(crudo)
                contents = json.loads(crudo)["contents"]
                partes = [{"functionCall": {"name": n, "args": a}} for n, a in gemini.turno(contents)]
                datos = json.dumps({"candidates": [{"content": {"role": "model", "parts": partes}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(datos)))
                self.end_headers()
                self.wfile.write(datos)

        return Manejador


def _esperar(nucleo: Nucleo, id_trabajo: int, estados: tuple[str, ...], segundos: float) -> dict:
    limite = time.time() + segundos
    trabajo: dict = {}
    while time.time() < limite:
        _, trabajo = nucleo.pedir(f"/trabajos/{id_trabajo}", nucleo.token)
        if trabajo.get("estado") in estados:
            return trabajo
        time.sleep(0.5)
    return trabajo


def _ficheros_con_la_clave(raiz: Path) -> list[str]:
    encontrados = []
    for fichero in raiz.rglob("*"):
        if not fichero.is_file() or fichero.name == "boveda.json":
            continue
        try:
            if CLAVE.encode() in fichero.read_bytes():
                encontrados.append(str(fichero.relative_to(raiz)))
        except OSError:
            pass
    return encontrados


def comprobar_todo() -> None:
    if sys.platform != "win32":
        raise SystemExit("La bóveda cifra con DPAPI: este verificador solo corre en Windows.")
    sitio = Sitio()
    sitio.arrancar()
    gemini = GeminiDeGuion(sitio)
    gemini.arrancar()
    nucleo = Nucleo(
        {
            "PERSEO_DISPARADORES": "",
            "PERSEO_WEB_LOCAL": "1",
            "PERSEO_GEMINI_API": gemini.url,
            "GEMINI_API_KEY": "clave-falsa",
            "PERSEO_RECADO_MODELO": "modelo-de-guion",
        }
    )
    from perseo_core.servicios import boveda

    boveda.Boveda(nucleo.datos / boveda.NOMBRE_FICHERO).guardar(
        "casa", "cuenta", ["127.0.0.1"], {"usuario": "ana@correo.es", "clave": CLAVE}
    )
    nucleo.arrancar()
    try:
        _, trabajo = nucleo.pedir(
            "/trabajos", nucleo.token, "POST",
            {"agente": "recado", "peticion": {"texto": "Reserva mesa para 2 el viernes en Casa Prueba", "accion": "hacer"}},
        )
        parado = _esperar(nucleo, trabajo["id"], ("esperando", "hecho", "fallido"), 180)
        confirmacion = parado.get("confirmacion") or {}
        comprobar("El recado se para antes de reservar", parado.get("estado") == "esperando", str(parado.get("error") or parado.get("estado")))
        comprobar("La pregunta es de nivel exterior", confirmacion.get("nivel") == "exterior", str(confirmacion.get("nivel")))
        comprobar("La pregunta nombra el botón de la página", "Reservar mesa para 2" in str(confirmacion.get("resumen")), str(confirmacion.get("resumen")))
        comprobar("El sitio recibió la clave de la bóveda, entera", sitio.claves == [CLAVE], str(sitio.claves))
        comprobar("Y ninguna reserva todavía", sitio.reservas == 0, str(sitio.reservas))
        peticiones_antes = len(gemini.cuerpos)

        codigo, _ = nucleo.pedir(f"/trabajos/{trabajo['id']}/aprobar", nucleo.token, "POST", {})
        comprobar("El sí se acepta", codigo == 200, str(codigo))
        hecho = _esperar(nucleo, trabajo["id"], ("hecho", "fallido", "esperando"), 120)
        resultado = hecho.get("resultado") or {}
        comprobar("Tras el sí, el recado acaba", hecho.get("estado") == "hecho", str(hecho.get("error") or hecho.get("estado")))
        comprobar("La reserva llega una vez", sitio.reservas == 1, str(sitio.reservas))
        comprobar("El resumen trae el código", "ABC123" in str(resultado.get("texto")), str(resultado.get("texto")))
        comprobar(
            "Sigue desde el botón: una sola petición más al modelo",
            len(gemini.cuerpos) == peticiones_antes + 1,
            f"{peticiones_antes} -> {len(gemini.cuerpos)}",
        )
        comprobar("El modelo no vio la clave en ninguna petición", all(CLAVE not in c for c in gemini.cuerpos))
        comprobar("El modelo vio la referencia", any("{{boveda:casa.clave}}" in c for c in gemini.cuerpos))
    finally:
        nucleo.parar()
        fugas = _ficheros_con_la_clave(nucleo.datos)
        comprobar("Ningún fichero de <datos> guarda la clave en claro", not fugas, ", ".join(fugas))
        if "--volcar" in sys.argv:
            nucleo.volcar()
        nucleo.limpiar()
        sitio.parar()
        gemini.parar()


if __name__ == "__main__":
    comprobar_todo()
    resumir()
