"""Gmail y Google Calendar, sin SDK.

Es lo que faltaba para cerrar la Fase D: hasta ahora el buzón y el calendario
eran ficheros JSON, porque las credenciales las tiene que crear el usuario. Esto
es el otro lado del puerto — cuando haya credenciales, se cambia una variable de
entorno y los agentes no se enteran.

**Sin biblioteca de Google, a propósito.** El SDK oficial arrastra `google-auth`,
`google-api-python-client` y media docena de dependencias transitivas; lo que
hace falta aquí es refrescar un testigo y hacer dos peticiones GET, y eso cabe en
`aiohttp`, que ya estaba. El mismo criterio que llevó a aiohttp en vez de FastAPI
y a las tuberías en la Fase 3: el núcleo acabará viviendo en una Raspberry Pi.

**Del correo solo se leen las cabeceras y el extracto.** Se pide `format=metadata`
para el remitente y el asunto, y se usa el `snippet` que ya devuelve Gmail. El
cuerpo entero no se descarga: para triar no hace falta, y lo que no se baja no se
puede filtrar por accidente. Sí se marca el mensaje como visto en la marca de
agua del disparador — pero **no** se toca su estado en Gmail: leerlo en el móvil
sigue siendo cosa tuya.

Nada de esto hace nada irreversible: solo lee. Mandar correo y mover eventos es
de la política de §7 y de otro día.


"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import urllib.parse
from dataclasses import dataclass
from email.message import EmailMessage
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import aiohttp

from . import almacen
from .dominio.evento import Evento
from .dominio.mensaje import Mensaje

logger = logging.getLogger(__name__)

#: Se puede apuntar a otro sitio para probar sin tocar Google de verdad, igual
#: que `PERSEO_TELEGRAM_API` con Telegram.
def _base(variable: str, por_defecto: str) -> str:
    import os

    return os.environ.get(variable, por_defecto).rstrip("/")


URL_TESTIGO = lambda: _base("PERSEO_GOOGLE_OAUTH", "https://oauth2.googleapis.com")  # noqa: E731
URL_GMAIL = lambda: _base("PERSEO_GOOGLE_GMAIL", "https://gmail.googleapis.com")  # noqa: E731
URL_CALENDAR = lambda: _base("PERSEO_GOOGLE_CALENDAR", "https://www.googleapis.com")  # noqa: E731

#: Margen con el que se considera caducado un testigo, para no usarlo justo en el
#: segundo en que expira y llevarse un 401 evitable.
MARGEN_TESTIGO = 60

#: Cuántos mensajes se piden como mucho por vuelta. El disparador ya trocea, pero
#: pedirle mil a Gmail en el primer arranque es maleducado y lento.
TOPE_MENSAJES = 25


class SinCredenciales(Exception):
    """No hay con qué hablar con Google. No es un error: es algo sin configurar."""


@dataclass(frozen=True)
class Credenciales:
    """Lo que hace falta para refrescar el acceso, y nada más.

    No hay contraseña de la cuenta por ninguna parte: el `refresh_token` lo
    concede el usuario una vez, desde el navegador, y se puede revocar sin tocar
    nada más.
    """

    client_id: str
    client_secret: str
    refresh_token: str

    @classmethod
    def desde_fichero(cls, ruta: Path) -> "Credenciales":
        if not ruta.is_file():
            raise SinCredenciales(f"No hay credenciales de Google en {ruta}")
        try:
            crudo = json.loads(ruta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise SinCredenciales(f"{ruta} no se puede leer: {e}") from None

        # Se acepta el fichero tal cual lo escupe la consola de Google, que mete
        # las claves dentro de "installed", para no obligar a editarlo a mano.
        datos = crudo.get("installed") or crudo.get("web") or crudo
        faltan = [c for c in ("client_id", "client_secret", "refresh_token") if not datos.get(c)]
        if faltan:
            raise SinCredenciales(f"A {ruta} le faltan: {', '.join(faltan)}")
        return cls(
            client_id=str(datos["client_id"]),
            client_secret=str(datos["client_secret"]),
            refresh_token=str(datos["refresh_token"]),
        )


class Sesion:
    """Mantiene un testigo de acceso vivo y hace las peticiones.

    El testigo dura una hora; el `refresh_token` no caduca salvo que se revoque.
    Se refresca cuando falta poco, y también si una petición se lleva un 401 —que
    es lo que pasa si el reloj de la máquina va torcido o si Google lo invalida
    antes de tiempo.
    """

    def __init__(self, credenciales: Credenciales, sesion: aiohttp.ClientSession) -> None:
        self._credenciales = credenciales
        self._http = sesion
        self._testigo = ""
        self._caduca = 0.0

    async def _refrescar(self) -> None:
        carga = {
            "client_id": self._credenciales.client_id,
            "client_secret": self._credenciales.client_secret,
            "refresh_token": self._credenciales.refresh_token,
            "grant_type": "refresh_token",
        }
        async with self._http.post(f"{URL_TESTIGO()}/token", data=carga) as respuesta:
            datos = await respuesta.json()
            if respuesta.status != 200:
                raise SinCredenciales(
                    f"Google no dio testigo ({respuesta.status}): "
                    f"{datos.get('error_description') or datos.get('error')}"
                )

        self._testigo = str(datos.get("access_token", ""))
        if not self._testigo:
            raise SinCredenciales("Google devolvió una respuesta sin testigo.")
        self._caduca = time.monotonic() + float(datos.get("expires_in", 3600)) - MARGEN_TESTIGO
        logger.info("Testigo de Google renovado.")

    async def pedir(self, url: str, parametros: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET autenticado. Refresca el testigo una vez si hace falta."""
        if not self._testigo or time.monotonic() >= self._caduca:
            await self._refrescar()

        for intento in (1, 2):
            cabeceras = {"Authorization": f"Bearer {self._testigo}"}
            async with self._http.get(url, params=parametros, headers=cabeceras) as respuesta:
                if respuesta.status == 401 and intento == 1:
                    # Testigo invalidado antes de tiempo: se renueva y se repite.
                    await self._refrescar()
                    continue
                datos = await respuesta.json()
                if respuesta.status != 200:
                    raise RuntimeError(
                        f"Google respondió {respuesta.status}: "
                        f"{(datos.get('error') or {}).get('message', datos)}"
                    )
                return datos
        raise RuntimeError("Google siguió rechazando el testigo tras renovarlo.")

    async def mandar(self, url: str, cuerpo: dict[str, Any]) -> dict[str, Any]:
        """POST autenticado. Mismo trato del 401 que `pedir`.

        Existe por una sola cosa: crear borradores. Todo lo demás que hace este
        módulo lee, y eso no es casualidad — el testigo pide `gmail.compose`, que
        escribe borradores y **no** envía.
        """
        if not self._testigo or time.monotonic() >= self._caduca:
            await self._refrescar()

        for intento in (1, 2):
            cabeceras = {"Authorization": f"Bearer {self._testigo}"}
            async with self._http.post(url, json=cuerpo, headers=cabeceras) as respuesta:
                if respuesta.status == 401 and intento == 1:
                    await self._refrescar()
                    continue
                datos = await respuesta.json()
                if respuesta.status not in (200, 201):
                    mensaje = (datos.get("error") or {}).get("message", datos)
                    if respuesta.status == 403:
                        raise RuntimeError(
                            f"Google respondió 403: {mensaje}. Si habla de permisos, el "
                            "testigo es de antes de gmail.compose: vuelve a ejecutar "
                            "`python -m perseo_core.autorizar_google`."
                        )
                    raise RuntimeError(f"Google respondió {respuesta.status}: {mensaje}")
                return datos
        raise RuntimeError("Google siguió rechazando el testigo tras renovarlo.")


def _cabecera(cabeceras: list[dict[str, Any]], nombre: str) -> str:
    for cabecera in cabeceras:
        if str(cabecera.get("name", "")).lower() == nombre.lower():
            return str(cabecera.get("value", ""))
    return ""


class ClienteGoogle:
    """La parte que el buzón y el calendario tenían escrita igual.

    Los dos hablan con Google por la misma puerta —una `ClientSession` con su
    plazo y una `Sesion` que renueva el testigo— y los dos la abrían tarde, a la
    primera petición, para que arrancar el núcleo sin credenciales no costara
    ni un socket. Eso eran catorce líneas idénticas en dos sitios; ahora están
    aquí, y cada cliente solo escribe lo suyo: qué le pide a Google.
    """

    def __init__(self, credenciales: Credenciales) -> None:
        self._credenciales = credenciales
        self._http: aiohttp.ClientSession | None = None
        self._sesion: Sesion | None = None

    async def _abrir(self) -> Sesion:
        if self._sesion is None:
            self._http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
            self._sesion = Sesion(self._credenciales, self._http)
        return self._sesion

    async def cerrar(self) -> None:
        if self._http is not None:
            await self._http.close()
            self._http = None
            self._sesion = None


class BuzonGmail(ClienteGoogle):
    """El buzón de verdad. Solo lee, y solo cabeceras y extracto."""

    #: Lo que se considera "entrante sin mirar". Se dejan fuera los chats, que en
    #: Gmail también son mensajes y no son correo.
    CONSULTA = "is:unread -in:chats"

    def __init__(self, credenciales: Credenciales, tope: int = TOPE_MENSAJES) -> None:
        super().__init__(credenciales)
        self._tope = tope

    async def nuevos(self) -> list[Mensaje]:
        sesion = await self._abrir()
        listado = await sesion.pedir(
            f"{URL_GMAIL()}/gmail/v1/users/me/messages",
            {"q": self.CONSULTA, "maxResults": self._tope},
        )

        mensajes: list[Mensaje] = []
        for referencia in listado.get("messages") or []:
            identificador = str(referencia.get("id", ""))
            if not identificador:
                continue
            # El listado ya trae el hilo: pedirlo aparte seria una peticion mas
            # por correo para un dato que ya esta en la mano.
            hilo = str(referencia.get("threadId", ""))
            detalle = await sesion.pedir(
                f"{URL_GMAIL()}/gmail/v1/users/me/messages/{identificador}",
                # `metadata` y tres cabeceras: el cuerpo no se descarga.
                {
                    "format": "metadata",
                    "metadataHeaders": ["From", "Subject", "Date"],
                },
            )
            cabeceras = (detalle.get("payload") or {}).get("headers") or []
            mensajes.append(
                Mensaje(
                    id=identificador,
                    remitente=_cabecera(cabeceras, "From"),
                    asunto=_cabecera(cabeceras, "Subject"),
                    extracto=str(detalle.get("snippet", "")),
                    fecha=_cabecera(cabeceras, "Date"),
                    hilo=hilo,
                )
            )
        return mensajes

    async def crear_borrador(
        self,
        para: str,
        asunto: str,
        cuerpo: str,
        hilo: str = "",
    ) -> dict[str, Any]:
        """Deja un borrador en Gmail. **No lo envía, y no puede.**

        El testigo pide `gmail.compose`, que es el ámbito más pequeño capaz de
        escribir un borrador. No incluye `send`, así que aunque alguien —el
        modelo, un correo con instrucciones dentro, un fallo de este código—
        intentara enviarlo, Google responde 403. La garantía no está en el
        cuidado de quien programa: está en el permiso que se concedió.

        Si se pasa `hilo`, el borrador cuelga de esa conversación y le llega al
        destinatario como una respuesta y no como un correo suelto.
        """
        mensaje = EmailMessage()
        mensaje["To"] = para
        mensaje["Subject"] = asunto
        # Nada de `From`: lo pone Gmail con la cuenta del testigo. Escribirlo
        # aquí solo sirve para equivocarse de dirección.
        mensaje.set_content(cuerpo)

        # base64url **sin relleno de más y sin saltos de línea**: es lo que pide
        # la API, y con el base64 normal contesta un 400 que habla de "Invalid
        # value" sin decir de qué campo.
        crudo = base64.urlsafe_b64encode(mensaje.as_bytes()).decode("ascii")

        peticion: dict[str, Any] = {"message": {"raw": crudo}}
        if hilo:
            peticion["message"]["threadId"] = hilo

        sesion = await self._abrir()
        respuesta = await sesion.mandar(
            f"{URL_GMAIL()}/gmail/v1/users/me/drafts", peticion
        )
        return {
            "id": str(respuesta.get("id", "")),
            "mensaje": str((respuesta.get("message") or {}).get("id", "")),
        }


class CalendarioGoogle(ClienteGoogle):
    """El calendario de verdad. Solo lee lo que viene."""

    def __init__(self, credenciales: Credenciales, calendario: str = "primary") -> None:
        super().__init__(credenciales)
        self._calendario = calendario

    async def proximos(self, horizonte: timedelta) -> list[Evento]:
        sesion = await self._abrir()
        ahora = datetime.now(timezone.utc)
        datos = await sesion.pedir(
            f"{URL_CALENDAR()}/calendar/v3/calendars/"
            f"{urllib.parse.quote(self._calendario)}/events",
            {
                "timeMin": ahora.isoformat(),
                "timeMax": (ahora + horizonte).isoformat(),
                # Sin esto, un evento repetido llega como una sola serie y no
                # como la cita concreta que empieza dentro de media hora.
                "singleEvents": "true",
                "orderBy": "startTime",
                "maxResults": 50,
            },
        )

        eventos: list[Evento] = []
        for crudo in datos.get("items") or []:
            inicio = crudo.get("start") or {}
            # Los de todo el día traen `date` en vez de `dateTime`, y avisar de
            # ellos "a las 00:00" no dice nada útil: se dejan fuera.
            if not inicio.get("dateTime"):
                continue
            eventos.append(
                Evento(
                    id=str(crudo.get("id", "")),
                    titulo=str(crudo.get("summary", "(sin título)")),
                    inicio=str(inicio["dateTime"]),
                    fin=str((crudo.get("end") or {}).get("dateTime", "")),
                    lugar=str(crudo.get("location", "")),
                )
            )
        return eventos


# --------------------------------------------------------------------------- #
# Enchufe
# --------------------------------------------------------------------------- #


def credenciales(cfg: almacen.Configuracion) -> Credenciales:
    """Las credenciales configuradas. Lanza `SinCredenciales` si no hay."""
    return Credenciales.desde_fichero(Path(cfg.google_credenciales))


async def comprobar(cfg: almacen.Configuracion) -> str:
    """Prueba las credenciales pidiendo un testigo. Para usarlo a mano.

        python -c "import asyncio;from perseo_core import almacen,google_api as g;\\
                   print(asyncio.run(g.comprobar(almacen.cargar_configuracion())))"
    """
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as http:
        sesion = Sesion(credenciales(cfg), http)
        await sesion._refrescar()
    return "Las credenciales de Google valen."


def _sincrono() -> None:  # pragma: no cover - atajo para la línea de comandos
    import sys

    cfg = almacen.cargar_configuracion()
    try:
        print(asyncio.run(comprobar(cfg)))
    except (SinCredenciales, RuntimeError) as e:
        print(f"No: {e}")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    _sincrono()
