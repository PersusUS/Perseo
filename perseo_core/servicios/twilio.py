"""Twilio: un número de teléfono y un WhatsApp para Perseo, hablando por su API.

Instinct vive en la mensajería y el teléfono: se le escribe por WhatsApp y se le
llama. Twilio es lo que da a Perseo lo mismo: un número que puede recibir y
hacer llamadas, y un remitente de WhatsApp. Aquí está el cliente —mandar un
mensaje, empezar una llamada—, la comprobación de que lo que llega viene de
Twilio, y el TwiML con el que se conduce una llamada.

**Cómo se habla por teléfono sin tocar audio.** Nada de flujos de audio en
tiempo real: una llamada es una serie de turnos. Twilio dice un texto
(`<Say>`), escucha y transcribe (`<Gather input="speech">`), y manda la
transcripción a una dirección nuestra, que contesta con el siguiente turno.
Es más lento que la voz de la app, y a cambio no hay audio que convertir ni
socket que mantener vivo, y se prueba con peticiones HTTP normales.

**Lo que llega se comprueba.** Twilio firma cada petición (`X-Twilio-Signature`,
HMAC-SHA1 de la URL y los parámetros con el token de la cuenta). Lo que no trae
una firma buena no se atiende: es la única forma de que un webhook abierto a
internet no sea una puerta para cualquiera.

**Qué hace falta de fuera, y lo pone él.** Una cuenta de Twilio con un número,
y una dirección pública que llegue a este PC —`tailscale funnel` sobre el puerto
de `caras/twilio.py`—. Sin eso, todo esto se retira solo al arrancar.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import aiohttp

logger = logging.getLogger(__name__)

NOMBRE_FICHERO = "twilio.json"

#: Lo más largo que se deja en un solo mensaje. Twilio corta a 1.600 caracteres;
#: lo que pase se manda en trozos.
TOPE_MENSAJE = 1500


@dataclass(frozen=True)
class Cuenta:
    sid: str
    token: str
    #: El número de Perseo (E.164), para llamadas y SMS.
    numero: str
    #: El remitente de WhatsApp de la cuenta, con o sin `whatsapp:` delante.
    whatsapp: str
    #: El teléfono del señor Persus (E.164): el único que puede hablarle.
    dueno: str
    #: Por dónde llega Twilio a este PC, sin barra final.
    url_publica: str
    api: str = "https://api.twilio.com"
    #: La voz de `<Say>`. Polly Lucía es la de castellano de España que Twilio da.
    voz: str = "Polly.Lucia"

    @property
    def remitente_whatsapp(self) -> str:
        return self.whatsapp if self.whatsapp.startswith("whatsapp:") else f"whatsapp:{self.whatsapp}"


def cargar(directorio_datos: Path | str) -> Cuenta | None:
    """La cuenta, del entorno o de `<datos>/twilio.json`. `None` si falta algo esencial.

    El token no se escribe nunca en el código ni en `entorno.json`, que se
    comparte con la app: o variable de entorno, o el fichero de `<datos>`.
    """
    fichero: dict[str, Any] = {}
    ruta = Path(directorio_datos) / NOMBRE_FICHERO
    if ruta.exists():
        try:
            fichero = json.loads(ruta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.error("No se puede leer %s: %s", ruta, e)

    def valor(clave: str, variable: str) -> str:
        return str(os.environ.get(variable) or fichero.get(clave) or "").strip()

    cuenta = Cuenta(
        sid=valor("sid", "PERSEO_TWILIO_SID"),
        token=valor("token", "PERSEO_TWILIO_TOKEN"),
        numero=valor("numero", "PERSEO_TWILIO_NUMERO"),
        whatsapp=valor("whatsapp", "PERSEO_TWILIO_WHATSAPP"),
        dueno=valor("dueno", "PERSEO_TELEFONO_DUENO"),
        url_publica=valor("url_publica", "PERSEO_TWILIO_URL_PUBLICA").rstrip("/"),
        api=(valor("api", "PERSEO_TWILIO_API") or "https://api.twilio.com").rstrip("/"),
        voz=valor("voz", "PERSEO_TWILIO_VOZ") or "Polly.Lucia",
    )
    if not (cuenta.sid and cuenta.token and cuenta.dueno and cuenta.url_publica):
        return None
    return cuenta


def normalizar_numero(numero: str) -> str:
    """`whatsapp:+34 600 11 22 33` y `+34600112233` son el mismo teléfono."""
    limpio = str(numero or "").strip().lower().removeprefix("whatsapp:")
    return "".join(c for c in limpio if c.isdigit() or c == "+")


def es_del_dueno(cuenta: Cuenta, numero: str) -> bool:
    return bool(numero) and normalizar_numero(numero) == normalizar_numero(cuenta.dueno)


# --------------------------------------------------------------------------- #
# La firma
# --------------------------------------------------------------------------- #


def firma(token: str, url: str, parametros: dict[str, str]) -> str:
    """La firma que pone Twilio: la URL seguida de cada clave y valor, en orden."""
    datos = url + "".join(clave + parametros[clave] for clave in sorted(parametros))
    return base64.b64encode(hmac.new(token.encode(), datos.encode("utf-8"), hashlib.sha1).digest()).decode()


def firma_valida(token: str, url: str, parametros: dict[str, str], recibida: str) -> bool:
    return bool(recibida) and secrets.compare_digest(firma(token, url, parametros), recibida)


# --------------------------------------------------------------------------- #
# TwiML
# --------------------------------------------------------------------------- #


def _say(cuenta: Cuenta, texto: str) -> str:
    return f'<Say language="es-ES" voice="{escape(cuenta.voz)}">{escape(texto)}</Say>'


def decir_y_escuchar(cuenta: Cuenta, texto: str, accion: str) -> str:
    """Dice `texto` y escucha la respuesta, que llegará transcrita a `accion`.

    Si no contestan nada, `<Gather>` sigue de largo y el `<Redirect>` vuelve a
    la misma dirección marcando `silencio=1`: quien la atiende decide si
    insistir o colgar.
    """
    separador = "&amp;" if "?" in accion else "?"
    return (
        '<?xml version="1.0" encoding="UTF-8"?><Response>'
        f'<Gather input="speech" language="es-ES" speechTimeout="auto" action="{escape(accion)}" method="POST">'
        f"{_say(cuenta, texto)}</Gather>"
        f'<Redirect method="POST">{escape(accion)}{separador}silencio=1</Redirect>'
        "</Response>"
    )


def esperar(cuenta: Cuenta, texto: str, siguiente: str, segundos: int = 3) -> str:
    """Un «un momento» y volver a preguntar: para lo que tarda más que un turno."""
    partes = [_say(cuenta, texto)] if texto else []
    return (
        '<?xml version="1.0" encoding="UTF-8"?><Response>'
        + "".join(partes)
        + f'<Pause length="{int(segundos)}"/><Redirect method="POST">{escape(siguiente)}</Redirect></Response>'
    )


def decir_y_colgar(cuenta: Cuenta, texto: str) -> str:
    return f'<?xml version="1.0" encoding="UTF-8"?><Response>{_say(cuenta, texto)}<Hangup/></Response>'


def vacio() -> str:
    return '<?xml version="1.0" encoding="UTF-8"?><Response/>'


# --------------------------------------------------------------------------- #
# La API
# --------------------------------------------------------------------------- #


class ClienteTwilio:
    def __init__(self, cuenta: Cuenta) -> None:
        self.cuenta = cuenta
        self._sesion: aiohttp.ClientSession | None = None

    async def _post(self, recurso: str, datos: dict[str, str]) -> dict[str, Any]:
        if self._sesion is None or self._sesion.closed:
            self._sesion = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        url = f"{self.cuenta.api}/2010-04-01/Accounts/{self.cuenta.sid}/{recurso}.json"
        auth = aiohttp.BasicAuth(self.cuenta.sid, self.cuenta.token)
        async with self._sesion.post(url, data=datos, auth=auth) as r:
            cuerpo = await r.json(content_type=None)
            if r.status >= 400:
                mensaje = cuerpo.get("message") if isinstance(cuerpo, dict) else str(cuerpo)[:200]
                raise RuntimeError(f"Twilio respondió {r.status}: {mensaje}")
            return cuerpo

    async def mensaje(self, texto: str, canal: str = "whatsapp", para: str = "") -> list[str]:
        """Manda `texto` al dueño (o a `para`), en trozos si es largo. Devuelve los sid."""
        para = para or self.cuenta.dueno
        if canal == "whatsapp":
            origen, destino = self.cuenta.remitente_whatsapp, f"whatsapp:{normalizar_numero(para)}"
        else:
            origen, destino = self.cuenta.numero, normalizar_numero(para)
        trozos = [texto[i : i + TOPE_MENSAJE] for i in range(0, len(texto), TOPE_MENSAJE)] or [""]
        sids = []
        for trozo in trozos:
            creado = await self._post("Messages", {"From": origen, "To": destino, "Body": trozo})
            sids.append(str(creado.get("sid", "")))
        return sids

    async def llamar(self, para: str, ruta_twiml: str) -> str:
        """Empieza una llamada; Twilio pedirá el primer turno a `url_publica + ruta_twiml`."""
        if not self.cuenta.numero:
            raise RuntimeError("Para llamar hace falta el número de Perseo (PERSEO_TWILIO_NUMERO).")
        base = self.cuenta.url_publica
        creada = await self._post(
            "Calls",
            {
                "To": normalizar_numero(para),
                "From": self.cuenta.numero,
                "Url": base + ruta_twiml,
                "Method": "POST",
                "StatusCallback": base + "/twilio/estado",
                "StatusCallbackMethod": "POST",
            },
        )
        return str(creada.get("sid", ""))

    async def cerrar(self) -> None:
        if self._sesion is not None:
            await self._sesion.close()
            self._sesion = None
