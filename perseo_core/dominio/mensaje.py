"""Un correo reducido a lo que hace falta para decidir qué hacer con él.

Aquí abajo por lo mismo que `Evento`: lo construye quien habla con Gmail y lo
consume el agente `correo`, y tenerlo dentro del agente era el otro medio ciclo.
Sigue siendo deliberadamente pobre —remitente, asunto, un extracto— porque lo
que viaja al clasificador es esto y no el correo entero.
"""

from __future__ import annotations

import email.utils
import re
from dataclasses import asdict, dataclass
from typing import Any

#: Las cabeceras que delatan un envío sin persona detrás. `List-*` la ponen los
#: boletines y las listas; `Auto-Submitted` las respuestas automáticas (RFC 3834);
#: `Precedence: bulk|list|junk` los envíos masivos de antes de eso. Ninguna es
#: contenido: dicen cómo se mandó el correo, no qué dice.
CABECERAS_DE_ENVIO = ("List-Unsubscribe", "List-Id", "Auto-Submitted", "Precedence")

#: Buzones que nadie lee. Hacen falta además de las cabeceras por dos cosas: los
#: correos triados antes de pedirlas no las tienen, y hay avisos transaccionales
#: —GitHub, Stripe— que no se declaran lista porque no se puede uno dar de baja.
_BUZON_AUTOMATICO = re.compile(
    r"^(no-?reply|do-?not-?reply|notifications?|notify|alerts?|updates?|news(letter)?|"
    r"info|support|soporte|hello|hola|team|equipo|marketing|mailer(-daemon)?|bounces?|"
    r"postmaster|billing|facturacion|accounts?|security|noticias|avisos)([+._-].*)?$",
    re.IGNORECASE,
)

#: Dominios de reenvío de plataformas: la dirección cambia por usuario (Luma usa
#: `usr-…@user.luma-mail.com`), así que el buzón no delata nada y el dominio sí.
#: Es lo que sonó el primer día: el recordatorio de un evento de hacía diez días.
_DOMINIOS_AUTOMATICOS = ("luma-mail.com",)


def es_automatico(remitente: str, cabeceras: dict[str, str] | None = None) -> bool:
    """Si el correo lo mandó una máquina y no una persona.

    Se equivoca hacia el lado de «es una persona»: un correo automático que se
    cuela cuesta una llamada de más; uno de persona tomado por máquina es un
    hilo que se cae sin aviso, que es justo lo que el seguimiento existe para
    evitar.
    """
    cabeceras = {k.lower(): str(v or "").strip() for k, v in (cabeceras or {}).items()}
    if cabeceras.get("list-unsubscribe") or cabeceras.get("list-id"):
        return True
    if cabeceras.get("auto-submitted", "no").lower() not in ("", "no"):
        return True
    if cabeceras.get("precedence", "").lower() in ("bulk", "list", "junk"):
        return True
    direccion = email.utils.parseaddr(remitente or "")[1].lower()
    buzon, _, dominio = direccion.partition("@")
    if not dominio:
        return False
    if any(dominio == d or dominio.endswith("." + d) for d in _DOMINIOS_AUTOMATICOS):
        return True
    return bool(_BUZON_AUTOMATICO.match(buzon))


@dataclass(frozen=True)
class Mensaje:
    """Lo mínimo que hace falta para triar. Deliberadamente no es el correo entero."""

    id: str
    remitente: str
    asunto: str
    extracto: str = ""
    fecha: str = ""
    #: El hilo al que pertenece, cuando el buzón lo sabe. Lo usa el borrador para
    #: que la respuesta cuelgue de la conversación en vez de nacer suelta.
    hilo: str = ""
    #: Si lo mandó una máquina: un boletín, un aviso, una plataforma. El triaje
    #: no lo usa —su prompt está medido—; lo usa el seguimiento, que por un
    #: correo automático sin contestar no llama.
    automatico: bool = False

    def a_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def desde_dict(cls, crudo: dict[str, Any]) -> "Mensaje":
        return cls(
            id=str(crudo.get("id", "")),
            remitente=str(crudo.get("remitente", "")),
            asunto=str(crudo.get("asunto", "")),
            extracto=str(crudo.get("extracto", "")),
            fecha=str(crudo.get("fecha", "")),
            hilo=str(crudo.get("hilo", "")),
            automatico=bool(crudo.get("automatico", False)),
        )
