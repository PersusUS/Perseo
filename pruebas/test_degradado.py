"""Lo que está roto por fuera se dice una vez, y a quien lo puede arreglar.

Entre el 6 y el 23 de septiembre de 2026 el permiso de Google estuvo caducado.
El registro se llenó con 1.460 trazas iguales —una cada cinco minutos por
disparador— y nadie se enteró: el correo y la agenda estuvieron diecisiete días
callados. Además, reautorizar no bastaba: el núcleo seguía con el permiso viejo
en memoria hasta que alguien lo reiniciaba.

Aquí se fija lo contrario: una línea al caer, un aviso al móvil, vueltas cada
vez más espaciadas, una línea al volver, y el permiso nuevo recogido solo.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import replace
from pathlib import Path

import pytest

from perseo_core.caras import telegram
from perseo_core.infra import disparadores
from perseo_core.infra.bus import Bus, Evento
from perseo_core.servicios import google_api


class BusQueApunta(Bus):
    def __init__(self) -> None:
        super().__init__()
        self.publicados: list[tuple[str, dict]] = []

    def publicar(self, tipo: str, **datos):
        self.publicados.append((tipo, datos))
        return super().publicar(tipo, **datos)


def _plan(cfg, bus: Bus, *nombres: str) -> disparadores.Planificador:
    return disparadores.Planificador(replace(cfg, disparadores=nombres), bus)


def _registrar(monkeypatch, nombre: str, revisar) -> None:
    monkeypatch.setitem(
        disparadores.REGISTRO,
        nombre,
        disparadores.Disparador(nombre=nombre, intervalo=0.001, revisar=revisar),
    )


def _caducado() -> disparadores.Degradado:
    return disparadores.Degradado("google", "El permiso caducó.", google_api.ARREGLO)


# --------------------------------------------------------------------------- #
# El planificador
# --------------------------------------------------------------------------- #


def test_una_pieza_caida_se_avisa_una_vez_y_sin_traza(cfg, monkeypatch, caplog) -> None:
    vueltas = []

    async def revisar(ctx):
        vueltas.append(1)
        if len(vueltas) <= 4:
            raise _caducado()
        raise disparadores.Retirarse("fin de la prueba")

    _registrar(monkeypatch, "prueba", revisar)
    monkeypatch.setattr(disparadores, "ESPERA_MAXIMA_DEGRADADO", 0.01)
    bus = BusQueApunta()
    with caplog.at_level(logging.INFO, logger="perseo_core.infra.disparadores"):
        asyncio.run(_plan(cfg, bus, "prueba").ejecutar())

    assert [t for t, _ in bus.publicados] == ["sistema.degradado"]
    _, datos = bus.publicados[0]
    assert datos["pieza"] == "google" and datos["arreglo"] == google_api.ARREGLO
    avisos = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(avisos) == 1
    assert avisos[0].exc_info is None
    assert google_api.ARREGLO in avisos[0].getMessage()


def test_dos_disparadores_de_la_misma_pieza_avisan_una_sola_vez(cfg, monkeypatch) -> None:
    """Correo y agenda van los dos por Google: un permiso caducado es un aviso, no dos."""
    cuentas = {"a": 0, "b": 0}

    def hacer(nombre):
        async def revisar(ctx):
            cuentas[nombre] += 1
            if cuentas[nombre] <= 2:
                raise _caducado()
            raise disparadores.Retirarse("fin")

        return revisar

    _registrar(monkeypatch, "a", hacer("a"))
    _registrar(monkeypatch, "b", hacer("b"))
    monkeypatch.setattr(disparadores, "ESPERA_MAXIMA_DEGRADADO", 0.01)
    bus = BusQueApunta()
    asyncio.run(_plan(cfg, bus, "a", "b").ejecutar())
    assert [t for t, _ in bus.publicados] == ["sistema.degradado"]


def test_cuando_vuelve_se_dice_y_se_recupera_el_ritmo(cfg, monkeypatch, caplog) -> None:
    vueltas = []

    async def revisar(ctx):
        vueltas.append(1)
        if len(vueltas) == 1:
            raise _caducado()
        if len(vueltas) == 2:
            return
        raise disparadores.Retirarse("fin")

    _registrar(monkeypatch, "prueba", revisar)
    bus = BusQueApunta()
    with caplog.at_level(logging.INFO, logger="perseo_core.infra.disparadores"):
        asyncio.run(_plan(cfg, bus, "prueba").ejecutar())

    assert [t for t, _ in bus.publicados] == ["sistema.degradado", "sistema.recuperado"]
    assert any("vuelve a funcionar" in r.getMessage() for r in caplog.records)


def test_degradado_espacia_las_vueltas_hasta_el_tope(cfg, monkeypatch) -> None:
    esperas: list[float] = []
    vueltas = []

    async def revisar(ctx):
        vueltas.append(1)
        if len(vueltas) <= 6:
            raise _caducado()
        raise disparadores.Retirarse("fin")

    async def esperar(corutina, timeout):
        esperas.append(timeout)
        corutina.close()
        raise asyncio.TimeoutError

    _registrar(monkeypatch, "prueba", revisar)
    monkeypatch.setattr(disparadores, "ESPERA_MAXIMA_DEGRADADO", 0.016)
    monkeypatch.setattr(disparadores.asyncio, "wait_for", esperar)
    asyncio.run(_plan(cfg, BusQueApunta(), "prueba").ejecutar())

    assert esperas[:5] == pytest.approx([0.002, 0.004, 0.008, 0.016, 0.016])


def test_un_fallo_de_red_es_una_linea_sin_traza(cfg, monkeypatch, caplog) -> None:
    """Tras despertar de la suspensión el DNS tarda: eso no es un error del código."""
    vueltas = []

    async def revisar(ctx):
        vueltas.append(1)
        if len(vueltas) == 1:
            raise ConnectionResetError("se cortó")
        raise disparadores.Retirarse("fin")

    _registrar(monkeypatch, "prueba", revisar)
    with caplog.at_level(logging.WARNING, logger="perseo_core.infra.disparadores"):
        asyncio.run(_plan(cfg, BusQueApunta(), "prueba").ejecutar())
    avisos = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(avisos) == 1 and avisos[0].exc_info is None


# --------------------------------------------------------------------------- #
# Google: el permiso caducado, y el nuevo sin reiniciar
# --------------------------------------------------------------------------- #


class _Respuesta:
    def __init__(self, estado: int, cuerpo: dict) -> None:
        self.status = estado
        self._cuerpo = cuerpo

    async def json(self) -> dict:
        return self._cuerpo

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False


class _HttpFalso:
    def __init__(self, respuesta: _Respuesta) -> None:
        self._respuesta = respuesta
        self.cargas: list[dict] = []

    def post(self, url: str, data: dict | None = None, **_):
        self.cargas.append(data or {})
        return self._respuesta


def test_invalid_grant_es_un_permiso_caducado_y_dice_como_arreglarlo() -> None:
    http = _HttpFalso(
        _Respuesta(400, {"error": "invalid_grant", "error_description": "Token has been expired or revoked."})
    )
    sesion = google_api.Sesion(google_api.Credenciales("id", "s", "r"), http)
    with pytest.raises(google_api.TestigoCaducado) as error:
        asyncio.run(sesion._refrescar())
    assert google_api.ARREGLO in str(error.value)
    # Sigue siendo SinCredenciales: lo que ya lo capturaba, lo sigue capturando.
    assert isinstance(error.value, google_api.SinCredenciales)


def test_otro_400_no_se_confunde_con_un_permiso_caducado() -> None:
    http = _HttpFalso(_Respuesta(400, {"error": "invalid_client"}))
    sesion = google_api.Sesion(google_api.Credenciales("id", "s", "r"), http)
    with pytest.raises(google_api.SinCredenciales) as error:
        asyncio.run(sesion._refrescar())
    assert not isinstance(error.value, google_api.TestigoCaducado)


def _escribir(ruta: Path, refresh: str, cuando: float) -> None:
    ruta.write_text(
        json.dumps({"client_id": "id", "client_secret": "s", "refresh_token": refresh}),
        encoding="utf-8",
    )
    os.utime(ruta, (cuando, cuando))


def test_el_permiso_nuevo_se_recoge_sin_reiniciar(tmp_path: Path) -> None:
    ruta = tmp_path / "google.json"
    _escribir(ruta, "viejo", 1_000_000)
    cliente = google_api.BuzonGmail(google_api.Credenciales.desde_fichero(ruta))

    _escribir(ruta, "nuevo", 2_000_000)  # lo que hace autorizar_google
    cliente._recargar_si_cambiaron()
    assert cliente._credenciales.refresh_token == "nuevo"


def test_un_fichero_a_medio_escribir_no_tira_las_credenciales_buenas(tmp_path: Path) -> None:
    ruta = tmp_path / "google.json"
    _escribir(ruta, "bueno", 1_000_000)
    cliente = google_api.CalendarioGoogle(google_api.Credenciales.desde_fichero(ruta))

    ruta.write_text('{"client_id": ', encoding="utf-8")
    os.utime(ruta, (2_000_000, 2_000_000))
    cliente._recargar_si_cambiaron()
    assert cliente._credenciales.refresh_token == "bueno"


def test_el_buzon_caducado_pone_el_disparador_en_degradado(cfg, monkeypatch) -> None:
    from perseo_core.agentes import correo

    class BuzonCaducado:
        async def nuevos(self):
            raise google_api.TestigoCaducado("caducó")

    monkeypatch.setattr(correo, "_buzon", BuzonCaducado())
    monkeypatch.setattr(
        correo, "_vistos", disparadores.Vistos(ruta=Path(cfg.directorio_datos) / "v.json")
    )
    ctx = disparadores.Contexto(cfg=cfg, bus=Bus())
    with pytest.raises(disparadores.Degradado) as error:
        asyncio.run(correo._vigilar_buzon(ctx))
    assert error.value.pieza == "google"
    assert error.value.arreglo == google_api.ARREGLO


# --------------------------------------------------------------------------- #
# El aviso al móvil
# --------------------------------------------------------------------------- #


def test_telegram_avisa_de_la_pieza_caida_con_su_arreglo() -> None:
    evento = Evento(
        tipo="sistema.degradado",
        datos={"pieza": "google", "motivo": "El permiso caducó.", "arreglo": google_api.ARREGLO},
    )
    texto, botones = telegram.redactar(evento, "https://perseo.tailnet")
    assert texto.startswith("Google no funciona")
    assert google_api.ARREGLO in texto
    assert botones[0][0]["url"].startswith("https://perseo.tailnet")


def test_telegram_avisa_cuando_vuelve() -> None:
    evento = Evento(tipo="sistema.recuperado", datos={"pieza": "google"})
    texto, _ = telegram.redactar(evento, "https://perseo.tailnet")
    assert texto == "Google vuelve a funcionar."


# --------------------------------------------------------------------------- #
# El registro de accesos
# --------------------------------------------------------------------------- #


def test_los_sondeos_que_van_bien_no_llenan_el_registro() -> None:
    """3.318 de 4.624 líneas de acceso eran el mismo sondeo cada ocho segundos."""
    from perseo_core.__main__ import _SinSondeos

    filtro = _SinSondeos()

    def linea(texto: str) -> logging.LogRecord:
        return logging.LogRecord("aiohttp.access", logging.INFO, "", 0, texto, None, None)

    assert not filtro.filter(linea('127.0.0.1 [x] "POST /tareas/recoger HTTP/1.1" 200 174 "-" "-"'))
    assert not filtro.filter(linea('127.0.0.1 [x] "GET /salud HTTP/1.1" 200 427 "-" "-"'))
    # Un sondeo que falla, o cualquier otra ruta, se sigue viendo.
    assert filtro.filter(linea('127.0.0.1 [x] "POST /tareas/recoger HTTP/1.1" 500 80 "-" "-"'))
    assert filtro.filter(linea('127.0.0.1 [x] "GET /estado HTTP/1.1" 200 3700 "-" "-"'))
