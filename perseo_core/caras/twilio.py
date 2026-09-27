"""La cara de Twilio: WhatsApp, SMS y llamadas, en un puerto aparte.

Twilio necesita llegar a este PC desde internet para avisar de que ha llegado un
mensaje o de lo que han dicho en una llamada. Abrir el núcleo entero a internet
sería abrir también el panel, la cola y todo lo demás. Por eso esta cara es **un
servidor aparte**, en `127.0.0.1:PERSEO_TWILIO_PUERTO` (8788), que solo sabe de
las rutas `/twilio/…`: lo que se publica con `tailscale funnel 8788` es esto y
nada más.

**Todo lo que entra se comprueba dos veces.** Primero la firma de Twilio
(`servicios/twilio.firma_valida`): sin ella no se atiende nada. Después, quién:
solo el teléfono del dueño habla con Perseo. Un mensaje o una llamada de otro
número no llega al hilo ni al modelo; se contesta que no, y se apunta.

**Por qué las respuestas van en dos tiempos.** Twilio espera la respuesta de un
webhook unos quince segundos, y un turno del hilo con herramientas puede tardar
más. Así que en WhatsApp se contesta vacío al momento y la respuesta sale
después por la API; y al teléfono se dice «un momento» y se vuelve a preguntar
cada pocos segundos hasta que la respuesta está.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any

from aiohttp import web

from ..agentes import telefono
from ..infra.bus import Bus
from ..infra.configuracion import Configuracion
from ..servicios import hilo, recordatorios, twilio, ubicacion

logger = logging.getLogger(__name__)

PUERTO_POR_DEFECTO = 8788

#: Vueltas de «un momento» antes de dejar la respuesta para el hilo escrito.
TOPE_ESPERAS = 20

_DESPEDIDAS = re.compile(r"\b(adi[oó]s|hasta luego|cuelga|nada m[aá]s|eso es todo|chao)\b", re.I)


class CaraTwilio:
    def __init__(self, cfg: Configuracion, bus: Bus) -> None:
        self._cfg = cfg
        self._bus = bus
        self._parar = asyncio.Event()
        #: Turnos del hilo en marcha, por llamada: el «un momento» los espera.
        self._turnos: dict[str, asyncio.Task[str]] = {}
        self._esperas: dict[str, int] = {}
        self._en_curso: set[asyncio.Task[Any]] = set()

    def detener(self) -> None:
        self._parar.set()

    # -- ciclo de vida ------------------------------------------------------- #

    def app(self) -> web.Application:
        app = web.Application()
        app.add_routes(
            [
                web.post("/twilio/whatsapp", self._mensaje),
                web.post("/twilio/voz", self._entrante),
                web.post("/twilio/charla/{id}", self._charla),
                web.post("/twilio/charla/{id}/espera", self._espera),
                web.post("/twilio/negocio/{id}", self._negocio),
                web.post("/twilio/estado", self._estado),
            ]
        )
        return app

    async def ejecutar(self) -> None:
        cuenta = telefono.cuenta()
        if cuenta is None:
            logger.info("Twilio sin configurar: ni WhatsApp ni teléfono (ver docs/CONFIGURACION.md).")
            return
        puerto = int(os.environ.get("PERSEO_TWILIO_PUERTO", PUERTO_POR_DEFECTO))
        corredor = web.AppRunner(self.app(), access_log=None)
        await corredor.setup()
        await web.TCPSite(corredor, "127.0.0.1", puerto).start()
        logger.info("Twilio escuchando en 127.0.0.1:%d; público en %s.", puerto, cuenta.url_publica)
        try:
            await self._parar.wait()
        finally:
            await corredor.cleanup()

    # -- lo común ------------------------------------------------------------ #

    async def _formulario(self, peticion: web.Request) -> dict[str, str]:
        """Los parámetros, si la firma es buena. Si no, 403 y a otra cosa."""
        cuenta = telefono.cuenta()
        datos = {k: str(v) for k, v in (await peticion.post()).items()}
        url = (cuenta.url_publica if cuenta else "") + peticion.path_qs
        if cuenta is None or not twilio.firma_valida(
            cuenta.token, url, datos, peticion.headers.get("X-Twilio-Signature", "")
        ):
            logger.warning("Twilio: petición a %s sin firma válida; no se atiende.", peticion.path)
            raise web.HTTPForbidden(text="firma no válida")
        return datos

    @staticmethod
    def _twiml(xml: str) -> web.Response:
        return web.Response(text=xml, content_type="text/xml")

    def _lanzar(self, corrutina: Any) -> None:
        tarea = asyncio.create_task(corrutina)
        self._en_curso.add(tarea)
        tarea.add_done_callback(self._en_curso.discard)

    # -- WhatsApp y SMS ------------------------------------------------------ #

    async def _mensaje(self, peticion: web.Request) -> web.Response:
        datos = await self._formulario(peticion)
        cuenta = telefono.cuenta()
        assert cuenta is not None
        remitente = datos.get("From", "")
        if not twilio.es_del_dueno(cuenta, remitente):
            logger.warning("Twilio: mensaje de un número que no es el suyo; se ignora.")
            return self._twiml(twilio.vacio())
        canal = "whatsapp" if remitente.startswith("whatsapp:") else "sms"
        if datos.get("Latitude") and datos.get("Longitude"):
            ubicacion.guardar(
                self._cfg.directorio_datos, datos["Latitude"], datos["Longitude"],
                recordatorios.ahora_local(), fuente="WhatsApp" if canal == "whatsapp" else "SMS",
            )
            self._lanzar(self._contestar("Ubicación guardada.", canal))
            return self._twiml(twilio.vacio())
        texto = datos.get("Body", "").strip()
        if texto:
            self._lanzar(self._turno_escrito(texto, canal))
        return self._twiml(twilio.vacio())

    async def _turno_escrito(self, texto: str, canal: str) -> None:
        try:
            respuesta = await hilo.hablar(self._cfg.directorio_datos, texto, canal)
        except Exception as e:  # noqa: BLE001 — que el mensaje no se quede sin respuesta
            logger.exception("Twilio: el turno por %s falló.", canal)
            respuesta = f"No he podido contestar: {e}"
        await self._contestar(respuesta, canal)

    async def _contestar(self, texto: str, canal: str) -> None:
        cliente = telefono.cliente()
        if cliente is None:
            return
        try:
            await cliente.mensaje(texto, canal)
        except Exception as e:  # noqa: BLE001
            logger.warning("Twilio: no se pudo mandar la respuesta por %s: %s", canal, e)

    # -- Hablar con él por teléfono ---------------------------------------- #

    async def _entrante(self, peticion: web.Request) -> web.Response:
        """Él llama al número de Perseo. Nadie más: el resto oye que no y cuelga."""
        datos = await self._formulario(peticion)
        cuenta = telefono.cuenta()
        assert cuenta is not None
        if not twilio.es_del_dueno(cuenta, datos.get("From", "")):
            logger.warning("Twilio: llamada de un número que no es el suyo; se rechaza.")
            return self._twiml(twilio.decir_y_colgar(
                cuenta, "Este número es de un asistente personal y no atiende otras llamadas. Adiós."
            ))
        llamada = await asyncio.to_thread(telefono.nueva, "entrante", numero=cuenta.dueno, sid=datos.get("CallSid", ""))
        return self._twiml(twilio.decir_y_escuchar(
            cuenta, "Dígame.", f"{cuenta.url_publica}/twilio/charla/{llamada['id']}"
        ))

    async def _charla(self, peticion: web.Request) -> web.Response:
        datos = await self._formulario(peticion)
        cuenta = telefono.cuenta()
        assert cuenta is not None
        id_llamada = peticion.match_info["id"]
        llamada = await asyncio.to_thread(telefono.leer, id_llamada)
        if llamada is None or llamada.get("tipo") not in ("entrante", "movil"):
            return self._twiml(twilio.decir_y_colgar(cuenta, "Se ha cortado. Adiós."))
        aqui = f"{cuenta.url_publica}/twilio/charla/{id_llamada}"
        dicho = datos.get("SpeechResult", "").strip()

        if dicho:
            llamada["silencios"] = 0
            await asyncio.to_thread(telefono.guardar, llamada)
            if _DESPEDIDAS.search(dicho):
                return self._twiml(twilio.decir_y_colgar(cuenta, "Hasta luego."))
            self._turnos[id_llamada] = asyncio.create_task(
                hilo.hablar(self._cfg.directorio_datos, dicho, "teléfono")
            )
            self._esperas[id_llamada] = 0
            return self._twiml(twilio.esperar(cuenta, "Un momento.", f"{aqui}/espera", 2))

        if peticion.query.get("silencio"):
            llamada["silencios"] = int(llamada.get("silencios") or 0) + 1
            await asyncio.to_thread(telefono.guardar, llamada)
            if llamada["silencios"] >= 2:
                return self._twiml(twilio.decir_y_colgar(cuenta, "Le dejo. Hasta luego."))
            return self._twiml(twilio.decir_y_escuchar(cuenta, "¿Sigue ahí?", aqui))

        # Primera vez que se pide: la llamada que hace Perseo empieza por el motivo.
        saludo = llamada.get("motivo") if llamada.get("tipo") == "movil" else "Dígame."
        return self._twiml(twilio.decir_y_escuchar(cuenta, telefono.para_decir(str(saludo)), aqui))

    async def _espera(self, peticion: web.Request) -> web.Response:
        await self._formulario(peticion)
        cuenta = telefono.cuenta()
        assert cuenta is not None
        id_llamada = peticion.match_info["id"]
        aqui = f"{cuenta.url_publica}/twilio/charla/{id_llamada}"
        tarea = self._turnos.get(id_llamada)
        if tarea is None:
            return self._twiml(twilio.decir_y_escuchar(cuenta, "¿Decía?", aqui))
        if tarea.done():
            self._turnos.pop(id_llamada, None)
            try:
                respuesta = tarea.result()
            except Exception as e:  # noqa: BLE001
                respuesta = f"No he podido: {e}"
            return self._twiml(twilio.decir_y_escuchar(cuenta, telefono.para_decir(respuesta), aqui))
        self._esperas[id_llamada] = self._esperas.get(id_llamada, 0) + 1
        if self._esperas[id_llamada] > TOPE_ESPERAS:
            # El turno sigue y su respuesta queda en el hilo: se lee por escrito.
            return self._twiml(twilio.decir_y_colgar(cuenta, "Esto va para largo. Se lo dejo por escrito en el hilo."))
        return self._twiml(twilio.esperar(cuenta, "", f"{aqui}/espera", 3))

    # -- Llamar a un negocio ------------------------------------------------- #

    async def _negocio(self, peticion: web.Request) -> web.Response:
        datos = await self._formulario(peticion)
        return self._twiml(await telefono.turno_negocio(
            peticion.match_info["id"], datos.get("SpeechResult", "").strip() or None,
            bool(peticion.query.get("silencio")),
        ))

    async def _estado(self, peticion: web.Request) -> web.Response:
        datos = await self._formulario(peticion)
        estado = datos.get("CallStatus", "")
        if estado in ("completed", "busy", "no-answer", "failed", "canceled"):
            await asyncio.to_thread(telefono.terminada, datos.get("CallSid", ""), estado)
        return web.Response(text="")
