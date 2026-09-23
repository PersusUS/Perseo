"""La tarea `PerseoRevivir`, escrita entera y no con los valores de fábrica.

**Por qué XML y no `schtasks /SC MINUTE`.** Con la línea corta, Windows rellena
lo que no se dice con sus valores por defecto, y tres de ellos iban en contra
justo de lo que la tarea existe para hacer. Medido el 2026-09-23 en el portátil
del señor Persus:

  · `DisallowStartIfOnBatteries = True`: **con batería no se ejecuta nunca**. Ese
    día el núcleo cayó a las 14:21, el portátil siguió al 23 % descargando, y en
    tres horas nadie lo levantó. La tarea que evita «tres días apagado» (H-53)
    no servía en cuanto se desenchufaba el cargador;
  · `StopIfGoingOnBatteries = True`: al desenchufar, Windows para la tarea que
    esté corriendo, y con ella lo que haya lanzado si sigue en su *job*;
  · `StartWhenAvailable = False`: la vuelta que toca mientras el portátil duerme
    se pierde, y al despertar hay que esperar a la siguiente.

Aquí van las tres al revés, y dos disparadores más: **al volver de suspensión**
(evento 1 de `Power-Troubleshooter`, que en esta máquina sale también al salir
del modo de espera moderno) y **al desbloquear la sesión**. Son los dos momentos
en que más probable es que el núcleo no esté y más se nota.

Este módulo no toca Windows: arma el XML y lo lee. Quien lo registra es
`manage_startup.py`, y así esto se prueba en cualquier sistema.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime
from xml.sax.saxutils import escape

MINUTOS_ENTRE_REVISIONES = 10

#: Esperas tras despertar o desbloquear. La red tarda unos segundos en volver
#: después de la suspensión, y un núcleo que arranca sin red arranca sin
#: Telegram ni Google hasta la siguiente vuelta.
ESPERA_AL_DESPERTAR = "PT30S"
ESPERA_AL_DESBLOQUEAR = "PT15S"

#: Lo que tiene que decir la tarea registrada para no fallar en silencio.
AJUSTES_QUE_IMPORTAN = {
    "DisallowStartIfOnBatteries": "false",
    "StopIfGoingOnBatteries": "false",
    "StartWhenAvailable": "true",
}

_NS = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}

_AL_DESPERTAR = (
    "<QueryList><Query Id=\"0\" Path=\"System\"><Select Path=\"System\">"
    "*[System[Provider[@Name='Microsoft-Windows-Power-Troubleshooter'] and EventID=1]]"
    "</Select></Query></QueryList>"
)


def xml(ejecutable: str, argumentos: str, directorio: str, usuario: str, desde: datetime) -> str:
    """El XML de la tarea, listo para `schtasks /Create /XML`."""
    inicio = desde.replace(hour=0, minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%S")
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Perseo: levanta el núcleo y el detector si se han caído. Ver commands/tarea_revivir.py.</Description>
  </RegistrationInfo>
  <Triggers>
    <TimeTrigger>
      <Repetition>
        <Interval>PT{MINUTOS_ENTRE_REVISIONES}M</Interval>
        <StopAtDurationEnd>false</StopAtDurationEnd>
      </Repetition>
      <StartBoundary>{inicio}</StartBoundary>
      <Enabled>true</Enabled>
    </TimeTrigger>
    <EventTrigger>
      <Enabled>true</Enabled>
      <Subscription>{escape(_AL_DESPERTAR)}</Subscription>
      <Delay>{ESPERA_AL_DESPERTAR}</Delay>
    </EventTrigger>
    <SessionStateChangeTrigger>
      <Enabled>true</Enabled>
      <StateChange>SessionUnlock</StateChange>
      <UserId>{escape(usuario)}</UserId>
      <Delay>{ESPERA_AL_DESBLOQUEAR}</Delay>
    </SessionStateChangeTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>{escape(usuario)}</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{escape(ejecutable)}</Command>
      <Arguments>{escape(argumentos)}</Arguments>
      <WorkingDirectory>{escape(directorio)}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


def problemas(xml_registrado: str) -> list[str]:
    """Lo que la tarea que hay registrada hace mal, en palabras.

    Lista vacía si está bien. Un XML que no se entiende también es un
    problema: mejor decirlo que dar por buena una tarea que no se ha podido leer.
    """
    try:
        raiz = ET.fromstring(xml_registrado.strip().lstrip("﻿").encode("utf-16"))
    except ET.ParseError:
        return ["no se ha podido leer la tarea registrada"]

    ajustes = raiz.find("t:Settings", _NS)
    encontrados = {}
    for nombre in AJUSTES_QUE_IMPORTAN:
        nodo = ajustes.find(f"t:{nombre}", _NS) if ajustes is not None else None
        # Lo que no se escribe vale lo de fábrica, que es justo lo que falla.
        encontrados[nombre] = (nodo.text or "").strip().lower() if nodo is not None else None

    salida = []
    if encontrados["DisallowStartIfOnBatteries"] != "false":
        salida.append("con batería no se ejecuta")
    if encontrados["StopIfGoingOnBatteries"] != "false":
        salida.append("se para al desenchufar el cargador")
    if encontrados["StartWhenAvailable"] != "true":
        salida.append("la vuelta que toca con el portátil dormido se pierde")
    disparadores = raiz.find("t:Triggers", _NS)
    if disparadores is None or disparadores.find("t:EventTrigger", _NS) is None:
        salida.append("no mira al volver de suspensión")
    return salida
