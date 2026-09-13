"""Verificación de Gmail y Google Calendar, sin cuenta de Google.

Se levanta un servidor de mentira que habla el mismo protocolo —el `/token` de
OAuth, `messages`, `messages/{id}` y `events`— y se apunta el núcleo a él con
`PERSEO_GOOGLE_OAUTH`, `PERSEO_GOOGLE_GMAIL` y `PERSEO_GOOGLE_CALENDAR`. Es el
mismo truco que `verificar_telegram.py`, y sirve para lo mismo: comprobar lo que
de verdad importa antes de tener credenciales.

Lo que se comprueba, y no es poco: que el testigo se pide y se reutiliza, que un
401 lo renueva y reintenta **una** vez, que del correo **solo se piden cabeceras
y extracto** —el cuerpo no se descarga—, y que un evento de todo el día no se
cuela como si empezara a las doce de la noche.

Lo único que este script no puede probar es que la API real se comporte como está
documentada.

    python verificadores/verificar_google.py
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import sys
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perseo_core.servicios import autorizar_google, google_api  # noqa: E402
from verificadores.arnes_pruebas import (  # noqa: E402
    ManejadorFalso,
    ServidorFalso,
    comprobar,
    resumir,
)

CREDENCIALES = google_api.Credenciales(
    client_id="id-de-prueba", client_secret="secreto", refresh_token="refresco"
)

CUERPO_SECRETO = "El cuerpo entero del correo, que no debe descargarse."

#: El código de un solo uso que Google devuelve tras el consentimiento.
CODIGO = "codigo-de-un-solo-uso"


class FalsoGoogle(ServidorFalso):
    """Servidor mínimo que imita lo que se usa de Gmail y Calendar."""

    def __init__(self) -> None:
        self.testigos_pedidos = 0
        #: Lo que se ha mandado a `/token`, para poder mirar con qué se canjeó.
        self.canjes: list[dict[str, str]] = []
        self.rutas: list[str] = []
        self.parametros: list[dict[str, list[str]]] = []
        #: Cuando está en alto, la siguiente petición se contesta con un 401.
        self.caducar_una_vez = False
        super().__init__()

    def _manejador(self):
        falso = self

        class Manejador(ManejadorFalso):
            def _responder(self, codigo: int, cuerpo: Any) -> None:
                datos = json.dumps(cuerpo).encode()
                self.send_response(codigo)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(datos)))
                self.end_headers()
                self.wfile.write(datos)

            def do_POST(self) -> None:  # noqa: N802
                if self.path.rstrip("/").endswith("/token"):
                    largo = int(self.headers.get("Content-Length", "0"))
                    carga = urllib.parse.parse_qs(self.rfile.read(largo).decode())
                    falso.canjes.append({c: v[0] for c, v in carga.items()})

                    # El consentimiento pasa por aquí una vez, con un código de
                    # un solo uso, y es la única vez que Google da
                    # `refresh_token`. Después ya solo se refresca.
                    if carga.get("grant_type", [""])[0] == "authorization_code":
                        if carga.get("code", [""])[0] != CODIGO:
                            self._responder(400, {"error": "invalid_grant"})
                            return
                        self._responder(
                            200,
                            {
                                "access_token": "testigo-recien-dado",
                                "refresh_token": "refresco-de-verdad",
                                "expires_in": 3600,
                            },
                        )
                        return

                    falso.testigos_pedidos += 1
                    self._responder(
                        200,
                        {"access_token": f"testigo-{falso.testigos_pedidos}", "expires_in": 3600},
                    )
                    return
                self._responder(404, {"error": "no existe"})

            def do_GET(self) -> None:  # noqa: N802
                partes = urllib.parse.urlparse(self.path)
                falso.rutas.append(partes.path)
                falso.parametros.append(urllib.parse.parse_qs(partes.query))

                if falso.caducar_una_vez:
                    falso.caducar_una_vez = False
                    self._responder(401, {"error": {"message": "testigo caducado"}})
                    return

                if "/no-existe/" in partes.path:
                    # Marca del propio script para provocar un error de Google.
                    self._responder(404, {"error": {"message": "no existe"}})
                    return

                if partes.path.endswith("/messages"):
                    self._responder(200, {"messages": [{"id": "m1"}, {"id": "m2"}]})
                    return

                if "/messages/" in partes.path:
                    identificador = partes.path.rsplit("/", 1)[-1]
                    self._responder(
                        200,
                        {
                            "id": identificador,
                            "snippet": "Adjunto el presupuesto de la reforma.",
                            "payload": {
                                "headers": [
                                    {"name": "From", "value": "obra@example.com"},
                                    {"name": "Subject", "value": f"Presupuesto {identificador}"},
                                    {"name": "Date", "value": "Sat, 16 Aug 2026 09:00:00 +0200"},
                                ],
                                # Si alguien pidiera el mensaje entero, el cuerpo
                                # vendria aqui. Se deja para poder comprobar que
                                # NO se pide `format=full`.
                                "body": {"data": CUERPO_SECRETO},
                            },
                        },
                    )
                    return

                if partes.path.endswith("/events"):
                    dentro = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
                    self._responder(
                        200,
                        {
                            "items": [
                                {
                                    "id": "e1",
                                    "summary": "Revision medica",
                                    "start": {"dateTime": dentro},
                                    "end": {"dateTime": dentro},
                                    "location": "Centro de salud",
                                },
                                {
                                    "id": "todo-el-dia",
                                    "summary": "Cumpleanos de alguien",
                                    "start": {"date": "2026-08-16"},
                                },
                            ]
                        },
                    )
                    return

                self._responder(404, {"error": {"message": "no existe"}})

        return Manejador


def main() -> None:
    falso = FalsoGoogle()
    falso.arrancar()
    import os

    os.environ["PERSEO_GOOGLE_OAUTH"] = falso.url
    os.environ["PERSEO_GOOGLE_GMAIL"] = falso.url
    os.environ["PERSEO_GOOGLE_CALENDAR"] = falso.url
    print(f"Google de mentira en {falso.url}\n")

    # 1. Credenciales: se leen del fichero, con y sin envoltorio.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="perseo_google_") as tmp:
        ruta = Path(tmp) / "google.json"
        ruta.write_text(
            json.dumps(
                {
                    "installed": {
                        "client_id": "id",
                        "client_secret": "secreto",
                        "refresh_token": "refresco",
                    }
                }
            ),
            encoding="utf-8",
        )
        leidas = google_api.Credenciales.desde_fichero(ruta)
        comprobar("Se leen las credenciales dentro de 'installed'", leidas.client_id == "id")

        ruta.write_text(json.dumps({"client_id": "id"}), encoding="utf-8")
        try:
            google_api.Credenciales.desde_fichero(ruta)
            comprobar("Unas credenciales incompletas se rechazan", False, "no se rechazaron")
        except google_api.SinCredenciales as e:
            comprobar("Unas credenciales incompletas se rechazan", "refresh_token" in str(e))

        try:
            google_api.Credenciales.desde_fichero(Path(tmp) / "no_esta.json")
            comprobar("Sin fichero se dice que no hay credenciales", False, "no se dijo")
        except google_api.SinCredenciales:
            comprobar("Sin fichero se dice que no hay credenciales", True)

    async def contra_gmail() -> None:
        buzon = google_api.BuzonGmail(CREDENCIALES)
        try:
            mensajes = await buzon.nuevos()
            comprobar("Llegan los mensajes del buzon", len(mensajes) == 2, f"{len(mensajes)}")
            if mensajes:
                comprobar("Con remitente", mensajes[0].remitente == "obra@example.com")
                comprobar("Con asunto", mensajes[0].asunto.startswith("Presupuesto"))
                comprobar("Y con extracto", "presupuesto" in mensajes[0].extracto.lower())

            comprobar(
                "Se pide un solo testigo para todas las peticiones",
                falso.testigos_pedidos == 1,
                f"{falso.testigos_pedidos} testigo(s)",
            )

            # Lo importante para la privacidad: el cuerpo no se descarga.
            formatos = [p.get("format", [""])[0] for p in falso.parametros if "format" in p]
            comprobar(
                "Del correo solo se piden cabeceras, nunca el cuerpo",
                formatos and all(f == "metadata" for f in formatos),
                str(formatos),
            )
            comprobar(
                "Y solo tres cabeceras",
                all(
                    set(p.get("metadataHeaders", [])) <= {"From", "Subject", "Date"}
                    for p in falso.parametros
                    if "metadataHeaders" in p
                ),
            )
            comprobar(
                "El listado pide solo lo no leido y sin chats",
                any("is:unread" in q[0] and "-in:chats" in q[0] for p in falso.parametros for q in [p.get("q", [""])]),
            )

            # 2. Un 401 renueva el testigo y reintenta, una vez.
            falso.caducar_una_vez = True
            antes = falso.testigos_pedidos
            mensajes = await buzon.nuevos()
            comprobar(
                "Un testigo caducado se renueva y la peticion se repite",
                falso.testigos_pedidos == antes + 1 and len(mensajes) == 2,
                f"testigos={falso.testigos_pedidos} mensajes={len(mensajes)}",
            )
        finally:
            await buzon.cerrar()

    asyncio.run(contra_gmail())

    async def contra_calendar() -> None:
        calendario = google_api.CalendarioGoogle(CREDENCIALES)
        try:
            eventos = await calendario.proximos(timedelta(hours=2))
            comprobar("Llega el evento con hora", len(eventos) == 1, f"{len(eventos)}")
            if eventos:
                comprobar("Con su titulo", eventos[0].titulo == "Revision medica")
                comprobar("Y su sitio", eventos[0].lugar == "Centro de salud")
                comprobar("Y su momento se entiende", eventos[0].momento is not None)

            comprobar(
                "Un evento de todo el dia no se cuela",
                all(e.id != "todo-el-dia" for e in eventos),
            )
            ultimos = falso.parametros[-1]
            comprobar(
                "Se piden las citas concretas, no las series",
                ultimos.get("singleEvents") == ["true"],
                str(ultimos.get("singleEvents")),
            )
            comprobar(
                "Y acotadas al horizonte",
                "timeMin" in ultimos and "timeMax" in ultimos,
                str(sorted(ultimos)),
            )
        finally:
            await calendario.cerrar()

    asyncio.run(contra_calendar())

    # 3. Un error de Google no se traga en silencio.
    async def con_google_roto() -> None:
        buzon = google_api.BuzonGmail(CREDENCIALES)
        try:
            os.environ["PERSEO_GOOGLE_GMAIL"] = falso.url + "/no-existe"
            try:
                await buzon.nuevos()
                comprobar("Un 404 de Google se propaga", False, "no se propago")
            except RuntimeError as e:
                comprobar("Un 404 de Google se propaga", "404" in str(e), str(e)[:60])
        finally:
            os.environ["PERSEO_GOOGLE_GMAIL"] = falso.url
            await buzon.cerrar()

    asyncio.run(con_google_roto())

    comprobar_consentimiento(falso)

    falso.parar()
    resumir()


def comprobar_consentimiento(falso: FalsoGoogle) -> None:
    """El paso que hoy hace el usuario a mano: dar permiso y guardar el testigo.

    Se recorre entero sin cuenta de Google: la pantalla de consentimiento no se
    abre —se comprueba la URL que se habría abierto— y la vuelta del navegador se
    imita con una petición al servidor del bucle local que levanta el propio
    ayudante.
    """
    print("\n--- el consentimiento, sin cuenta de Google ---\n")
    os.environ["PERSEO_GOOGLE_CUENTAS"] = falso.url

    raiz = Path(tempfile.mkdtemp(prefix="perseo_google_"))
    fichero = raiz / "google.json"
    fichero.write_text(
        json.dumps({"installed": {"client_id": "id-de-prueba", "client_secret": "secreto"}}),
        encoding="utf-8",
    )

    try:
        # 1. El fichero recién descargado de la consola todavía no vale para
        #    `google_api` —le falta el testigo— y sí para el ayudante. Esa
        #    diferencia es la razón de que este módulo exista.
        try:
            google_api.Credenciales.desde_fichero(fichero)
            comprobar("Sin refresh_token, google_api se niega", False, "no se nego")
        except google_api.SinCredenciales:
            comprobar("Sin refresh_token, google_api se niega", True)
        cliente = autorizar_google.leer_cliente(fichero)
        comprobar("Y el ayudante sí lo lee", cliente == ("id-de-prueba", "secreto"), str(cliente))

        # 2. Lo que se le pide a Google: leer, y nada más.
        recogedor = autorizar_google.Recogedor()
        url = autorizar_google.url_de_consentimiento("id-de-prueba", recogedor.redireccion)
        comprobar("Se piden los dos ámbitos de solo lectura", "gmail.readonly" in url and "calendar.readonly" in url)
        comprobar(
            "Y ninguno que escriba",
            not any(a in url for a in ("gmail.send", "gmail.modify", "auth/calendar%20", "calendar.events")),
            url[:80],
        )
        comprobar("Se pide acceso sin conexión", "access_type=offline" in url)
        comprobar(
            "Y se fuerza la pantalla de permiso",
            "prompt=consent" in url,
            "sin esto, la segunda vez no hay refresh_token",
        )
        comprobar("La vuelta es al bucle local", "127.0.0.1" in recogedor.redireccion, recogedor.redireccion)

        # 3. La vuelta del navegador. Primero el favicon, que el navegador pide
        #    solo: si esa petición contara como vuelta, el proceso se daría por
        #    terminado sin código.
        recogido: dict[str, Any] = {}

        def esperar() -> None:
            try:
                recogido["codigo"] = recogedor.esperar()
            except autorizar_google.SinConsentimiento as e:
                recogido["error"] = str(e)

        hilo = threading.Thread(target=esperar, daemon=True)
        hilo.start()
        with contextlib.suppress(urllib.error.HTTPError):
            urllib.request.urlopen(recogedor.redireccion + "favicon.ico", timeout=10).read()
        urllib.request.urlopen(f"{recogedor.redireccion}?code={CODIGO}", timeout=10).read()
        hilo.join(timeout=10)
        comprobar("El código de la URL de vuelta se recoge", recogido.get("codigo") == CODIGO, str(recogido))

        # 4. El canje: el código de un solo uso por el testigo que dura.
        testigo = asyncio.run(
            autorizar_google.canjear("id-de-prueba", "secreto", CODIGO, recogedor.redireccion)
        )
        comprobar("El código se canjea por un refresh_token", testigo == "refresco-de-verdad", testigo)
        ultimo = falso.canjes[-1]
        comprobar("Se canjea como authorization_code", ultimo.get("grant_type") == "authorization_code")
        comprobar(
            "Con la misma dirección de vuelta que se anunció",
            ultimo.get("redirect_uri") == recogedor.redireccion,
            str(ultimo.get("redirect_uri")),
        )

        # 5. Y lo guardado es exactamente lo que `google_api` sabe leer.
        autorizar_google.guardar(fichero, "id-de-prueba", "secreto", testigo)
        credenciales = google_api.Credenciales.desde_fichero(fichero)
        comprobar("Lo guardado vale para google_api", credenciales.refresh_token == testigo)

        # 6. Decir que no es un final normal, no una traza.
        negado = autorizar_google.Recogedor()
        salida: dict[str, Any] = {}

        def esperar_negativa() -> None:
            try:
                negado.esperar()
            except autorizar_google.SinConsentimiento as e:
                salida["error"] = str(e)

        hilo = threading.Thread(target=esperar_negativa, daemon=True)
        hilo.start()
        urllib.request.urlopen(f"{negado.redireccion}?error=access_denied", timeout=10).read()
        hilo.join(timeout=10)
        comprobar(
            "Negar el permiso se explica en una línea",
            "access_denied" in salida.get("error", ""),
            str(salida.get("error")),
        )
    finally:
        shutil.rmtree(raiz, ignore_errors=True)
        os.environ.pop("PERSEO_GOOGLE_CUENTAS", None)
    print()


if __name__ == "__main__":
    main()
