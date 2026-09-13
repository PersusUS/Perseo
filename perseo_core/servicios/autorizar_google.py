"""El consentimiento de Google, una vez, desde aquí.

`google_api.py` sabe hablar con Gmail y con Calendar en cuanto tiene un
`refresh_token`. Conseguir ese testigo es lo único que no puede hacer solo: hay
que abrir un navegador, iniciar sesión y dar permiso. Este módulo es ese paso.

    python -m perseo_core.servicios.autorizar_google

Lo que hace: levanta un servidor en el bucle local, abre el navegador en la
pantalla de consentimiento de Google, recoge el código que Google devuelve a esa
dirección, lo canjea por un `refresh_token` y lo deja escrito en
`perseo_core/datos/google.json`, junto al `client_id` y el `client_secret` que ya
estaban ahí.

**Lee, y como mucho deja un borrador.** Los ámbitos son `gmail.readonly`,
`calendar.readonly` y —desde el 2026-08-16— `gmail.compose`.

`gmail.compose` es el ámbito más pequeño que permite escribir un borrador, y la
elección es deliberada: **no incluye enviar**. Se descartaron `gmail.send`, que
manda correo de verdad, y `gmail.modify`, que además puede borrar. Con este
testigo Perseo puede redactar y dejarlo en la carpeta de borradores, y darle a
enviar sigue siendo un gesto tuyo desde el móvil.

Consecuencia de tocar esta lista: **hay que volver a pasar por la pantalla de
consentimiento**. Un `refresh_token` lleva grabados los ámbitos con los que se
concedió, así que el que ya existe seguiría siendo de solo lectura y el borrador
fallaría con un 403 que habla de permisos insuficientes. Se vuelve a ejecutar
este ayudante y ya está — `prompt=consent` hace que Google entregue un
`refresh_token` nuevo.

**El redirect va al bucle local, no a un dominio.** Google devuelve el código
como parámetro de una URL, así que esa URL tiene que llegar a algún sitio: un
servidor de usar y tirar en `127.0.0.1` es lo que evita depender de un servicio
de terceros para recibirlo. El puerto se pide libre en el momento, así que hay
que dejar `http://127.0.0.1` como URI de redirección autorizado en la consola —
en las aplicaciones de escritorio Google acepta cualquier puerto del bucle local.


"""

from __future__ import annotations

import asyncio
import http.server
import json
import os
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Any

import aiohttp

from . import google_api
from ..infra.configuracion import cargar_configuracion

#: Lo que se pide. Ver la cabecera: `compose` escribe borradores y **no** envía.
AMBITOS = (
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
)

#: Dónde vive la pantalla de consentimiento. Se puede apuntar a otro sitio para
#: verificar el circuito sin cuenta, igual que `PERSEO_GOOGLE_OAUTH` con el
#: intercambio del código.
def _cuentas() -> str:
    return os.environ.get("PERSEO_GOOGLE_CUENTAS", "https://accounts.google.com").rstrip("/")


#: Lo que se le enseña a quien acaba de dar permiso. El navegador se queda con
#: esto abierto, así que dice qué ha pasado y qué toca ahora.
_PAGINA = """<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>Perseo</title></head>
<body style="font-family: system-ui; background: #101014; color: #e8e8ea; padding: 3rem">
<h1>{titulo}</h1>
<p>{texto}</p>
</body></html>"""


class SinConsentimiento(Exception):
    """No se pudo conseguir el testigo. Casi siempre porque alguien dijo que no."""


def leer_cliente(ruta: Path) -> tuple[str, str]:
    """Saca `client_id` y `client_secret` del fichero de la consola de Google.

    Se acepta tal cual lo descarga la consola —con las claves dentro de
    `installed`— para no obligar a editarlo a mano, igual que hace
    `google_api.Credenciales`. Aquí el `refresh_token` **no** hace falta: es
    justo lo que se viene a buscar.
    """
    if not ruta.is_file():
        raise SinConsentimiento(
            f"No hay credenciales en {ruta}. Descarga el JSON de la aplicación de "
            "escritorio desde la consola de Google Cloud y déjalo ahí."
        )
    try:
        crudo = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise SinConsentimiento(f"{ruta} no se puede leer: {e}") from None

    datos = crudo.get("installed") or crudo.get("web") or crudo
    faltan = [c for c in ("client_id", "client_secret") if not datos.get(c)]
    if faltan:
        raise SinConsentimiento(f"A {ruta} le faltan: {', '.join(faltan)}")
    return str(datos["client_id"]), str(datos["client_secret"])


def url_de_consentimiento(client_id: str, redireccion: str) -> str:
    """La pantalla que verá el usuario.

    `access_type=offline` y `prompt=consent` van juntos a propósito: sin el
    primero Google no da `refresh_token`, y sin el segundo deja de darlo a partir
    de la segunda vez que autorizas la misma aplicación — con lo que repetir el
    proceso para arreglar algo devolvería un testigo que caduca en una hora y
    parecería que el problema es otro.
    """
    parametros = {
        "client_id": client_id,
        "redirect_uri": redireccion,
        "response_type": "code",
        "scope": " ".join(AMBITOS),
        "access_type": "offline",
        "prompt": "consent",
    }
    return f"{_cuentas()}/o/oauth2/v2/auth?{urllib.parse.urlencode(parametros)}"


class Recogedor:
    """Servidor de usar y tirar que recoge el código de la URL de vuelta."""

    def __init__(self) -> None:
        self.codigo = ""
        self.error = ""
        self._servidor = http.server.HTTPServer(("127.0.0.1", 0), self._manejador())

    @property
    def redireccion(self) -> str:
        return f"http://127.0.0.1:{self._servidor.server_port}/"

    def esperar(self) -> str:
        """Atiende peticiones hasta que llegue la buena. Devuelve el código."""
        try:
            while not self.codigo and not self.error:
                self._servidor.handle_request()
        finally:
            self._servidor.server_close()
        if self.error:
            raise SinConsentimiento(f"Google devolvió: {self.error}")
        return self.codigo

    def _manejador(self):
        recogedor = self

        class Manejador(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_: Any) -> None:
                pass

            def do_GET(self) -> None:  # noqa: N802
                consulta = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
                # El navegador pide también el favicon; sin esto, esa petición se
                # tomaría por una vuelta sin código y se daría por fallado.
                if "code" not in consulta and "error" not in consulta:
                    self.send_response(404)
                    self.end_headers()
                    return

                if "error" in consulta:
                    recogedor.error = consulta["error"][0]
                    pagina = _PAGINA.format(
                        titulo="No se dio permiso",
                        texto="Perseo se queda sin Gmail ni Calendar. Puedes cerrar esto.",
                    )
                else:
                    recogedor.codigo = consulta["code"][0]
                    pagina = _PAGINA.format(
                        titulo="Listo",
                        texto="Ya puedes cerrar esta pestaña y volver al terminal.",
                    )

                crudo = pagina.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(crudo)))
                self.end_headers()
                self.wfile.write(crudo)

        return Manejador


async def canjear(client_id: str, client_secret: str, codigo: str, redireccion: str) -> str:
    """Cambia el código de un solo uso por el `refresh_token`, que dura."""
    carga = {
        "client_id": client_id,
        "client_secret": client_secret,
        "code": codigo,
        "redirect_uri": redireccion,
        "grant_type": "authorization_code",
    }
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as sesion:
        async with sesion.post(f"{google_api.URL_TESTIGO()}/token", data=carga) as respuesta:
            datos = await respuesta.json()
            if respuesta.status != 200:
                raise SinConsentimiento(
                    f"Google no aceptó el código ({respuesta.status}): "
                    f"{datos.get('error_description') or datos.get('error')}"
                )

    testigo = str(datos.get("refresh_token", ""))
    if not testigo:
        # Pasa si la aplicación ya estaba autorizada y se pidió sin
        # `prompt=consent`. El mensaje lo dice, porque el síntoma —todo va bien
        # hasta que una hora después deja de ir— no lleva a la causa.
        raise SinConsentimiento(
            "Google dio acceso pero no `refresh_token`. Quita el permiso a la "
            "aplicación en https://myaccount.google.com/permissions y repítelo."
        )
    return testigo


def guardar(ruta: Path, client_id: str, client_secret: str, refresh_token: str) -> None:
    """Deja el fichero como lo espera `google_api`: plano y con las tres claves."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(
        json.dumps(
            {
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def autorizar(ruta: Path, abrir_navegador: bool = True) -> str:
    """El paseo entero: consentimiento, código, testigo y fichero escrito."""
    client_id, client_secret = leer_cliente(ruta)
    recogedor = Recogedor()
    url = url_de_consentimiento(client_id, recogedor.redireccion)

    # `flush` en todo lo que se imprime aquí: si esto se ejecuta desde un
    # lanzador, con la salida redirigida, Python la almacena y no la suelta hasta
    # salir. Y aquí no se sale: se espera. Sin esto, quien mire la salida no ve
    # la URL que necesita para dar permiso a mano cuando el navegador no se abre
    # solo. Es, que se anotó por el detector de aplausos y vale igual aquí.
    print("Abriendo el navegador. Si no se abre solo, entra aquí:\n", flush=True)
    print(f"  {url}\n", flush=True)
    if abrir_navegador:
        webbrowser.open(url)
    print("Esperando a que des permiso…", flush=True)

    codigo = recogedor.esperar()
    testigo = asyncio.run(canjear(client_id, client_secret, codigo, recogedor.redireccion))
    guardar(ruta, client_id, client_secret, testigo)
    return testigo


def _sincrono() -> None:  # pragma: no cover - atajo para la línea de comandos
    import sys

    cfg = cargar_configuracion()
    ruta = Path(cfg.google_credenciales)
    try:
        autorizar(ruta)
    except SinConsentimiento as e:
        print(f"No: {e}")
        sys.exit(1)

    print(f"\nHecho: el refresh_token está en {ruta}.")
    print("Compruébalo con:  python -m perseo_core.servicios.google_api")
    print("Y arranca así:    PERSEO_CORREO=gmail PERSEO_AGENDA=google python -m perseo_core")


if __name__ == "__main__":  # pragma: no cover
    _sincrono()
