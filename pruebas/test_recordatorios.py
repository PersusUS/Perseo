"""Recordatorios: apuntar, repetir, quitar, avisar a la hora y avisar tarde.

Hasta el 2026-09-23 Perseo no sabía recordar nada a una hora. Lo que se fija
aquí es lo que no puede fallar sin que nadie se entere: la hora que se calcula,
que el aviso salga una vez y solo una, que el que se repite vuelva a su sitio y
que el que venció con el núcleo apagado avise diciendo que llega tarde.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from perseo_core.agentes import recordatorios as agente
from perseo_core.infra import disparadores, politica
from perseo_core.infra.bus import Bus
from perseo_core.servicios import catalogo
from perseo_core.servicios import recordatorios as rec

ZONA = timezone(timedelta(hours=2))
#: Un miércoles cualquiera, a media mañana.
AHORA = datetime(2026, 9, 23, 10, 0, tzinfo=ZONA)


# --------------------------------------------------------------------------- #
# Las fechas
# --------------------------------------------------------------------------- #


def test_en_minutos_cuenta_desde_ahora() -> None:
    assert rec.calcular_cuando(AHORA, en_minutos=20) == AHORA + timedelta(minutes=20)


def test_una_fecha_sin_zona_es_hora_local() -> None:
    manana = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    cuando = rec.calcular_cuando(datetime.now().astimezone(), fecha_hora=f"{manana}T09:00")
    assert (cuando.hour, cuando.minute) == (9, 0)
    assert cuando.tzinfo is not None


@pytest.mark.parametrize(
    ("argumentos", "dice"),
    [
        ({}, "falta cuándo"),
        ({"en_minutos": 0}, "más de cero"),
        ({"en_minutos": "veinte"}, "no es un número"),
        ({"fecha_hora": "mañana"}, "no es una fecha"),
        ({"fecha_hora": "2026-09-23T09:00+02:00"}, "ya han pasado"),
        ({"en_minutos": 60 * 24 * 400}, "más de un año"),
    ],
)
def test_lo_que_no_se_puede_apuntar_se_dice_en_palabras(argumentos, dice) -> None:
    with pytest.raises(ValueError, match=dice):
        rec.calcular_cuando(AHORA, **argumentos)


def test_manana_a_las_nueve_sin_saber_que_dia_es() -> None:
    assert rec.calcular_cuando(AHORA, hora="09:00", dias=1) == AHORA.replace(hour=9) + timedelta(days=1)


def test_una_hora_sola_es_la_proxima_vez_que_llega() -> None:
    assert rec.calcular_cuando(AHORA, hora="18:30") == AHORA.replace(hour=18, minute=30)
    # «A las nueve» dicho a las diez es mañana.
    assert rec.calcular_cuando(AHORA, hora="9:00") == AHORA.replace(hour=9) + timedelta(days=1)


def test_el_dia_de_la_semana_es_el_proximo_y_el_de_hoy_es_el_que_viene() -> None:
    jueves = rec.calcular_cuando(AHORA, hora="17:00", dia_semana="jueves")
    assert jueves == AHORA.replace(hour=17) + timedelta(days=1)
    miercoles = rec.calcular_cuando(AHORA, hora="17:00", dia_semana="Miércoles")
    assert miercoles == AHORA.replace(hour=17) + timedelta(days=7)


@pytest.mark.parametrize(
    ("argumentos", "dice"),
    [
        ({"hora": "nueve"}, "no es una hora"),
        ({"hora": "09:00", "dias": 0}, "ya ha pasado"),
        ({"hora": "09:00", "dia_semana": "festivo"}, "no es un día"),
        ({"hora": "09:00", "dias": -1}, "cero o más"),
    ],
)
def test_las_horas_imposibles_se_dicen(argumentos, dice) -> None:
    with pytest.raises(ValueError, match=dice):
        rec.calcular_cuando(AHORA, **argumentos)


def test_describir_habla_como_una_persona() -> None:
    assert rec.describir(AHORA.replace(hour=18), AHORA) == "hoy a las 18:00"
    assert rec.describir(AHORA + timedelta(days=1), AHORA) == "mañana a las 10:00"
    assert rec.describir(AHORA + timedelta(days=2), AHORA) == "el viernes 25 a las 10:00"
    assert rec.describir(AHORA + timedelta(days=30), AHORA) == "el 23/10/2026 a las 10:00"


def test_lo_diario_vuelve_al_dia_siguiente_y_lo_laborable_se_salta_el_fin_de_semana() -> None:
    viernes = datetime(2026, 9, 25, 8, 0, tzinfo=ZONA)
    despues = viernes + timedelta(minutes=1)
    assert rec.siguiente(viernes, "diario", despues) == viernes + timedelta(days=1)
    assert rec.siguiente(viernes, "laborables", despues).weekday() == 0  # lunes
    assert rec.siguiente(viernes, "semanal", despues) == viernes + timedelta(weeks=1)
    assert rec.siguiente(viernes, "nunca", despues) is None


def test_lo_diario_que_se_perdio_varios_dias_no_avisa_por_cada_uno() -> None:
    hace_tres_dias = AHORA - timedelta(days=3)
    proxima = rec.siguiente(hace_tres_dias, "diario", AHORA)
    assert AHORA < proxima <= AHORA + timedelta(days=1)


# --------------------------------------------------------------------------- #
# El fichero
# --------------------------------------------------------------------------- #


def test_apuntar_y_listar(tmp_path: Path) -> None:
    rec.crear(tmp_path, "Llamar al fontanero", AHORA + timedelta(hours=1))
    rec.crear(tmp_path, "Sacar la basura", AHORA + timedelta(minutes=5))
    assert [r["texto"] for r in rec.pendientes(tmp_path)] == ["Sacar la basura", "Llamar al fontanero"]


def test_un_texto_vacio_o_una_repeticion_rara_no_se_apuntan(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        rec.crear(tmp_path, "   ", AHORA)
    with pytest.raises(ValueError, match="no es una repetición"):
        rec.crear(tmp_path, "x", AHORA, repetir="cada rato")


def test_cancelar_por_el_principio_y_sin_tildes(tmp_path: Path) -> None:
    rec.crear(tmp_path, "Llamar a Álvaro", AHORA + timedelta(hours=1))
    assert rec.cancelar(tmp_path, "llamar a alvaro")["texto"] == "Llamar a Álvaro"
    assert rec.pendientes(tmp_path) == []


def test_si_encajan_dos_no_se_quita_ninguno(tmp_path: Path) -> None:
    rec.crear(tmp_path, "Llamar a Ana", AHORA + timedelta(hours=1))
    rec.crear(tmp_path, "Llamar a Luis", AHORA + timedelta(hours=2))
    with pytest.raises(ValueError, match="encajan 2"):
        rec.cancelar(tmp_path, "llamar")
    assert len(rec.pendientes(tmp_path)) == 2


def test_un_fichero_roto_no_tumba_nada(tmp_path: Path) -> None:
    rec.ruta(tmp_path).write_text("{roto", encoding="utf-8")
    assert rec.pendientes(tmp_path) == []
    rec.crear(tmp_path, "Otra vez", AHORA + timedelta(hours=1))
    assert len(json.loads(rec.ruta(tmp_path).read_text(encoding="utf-8"))) == 1


# --------------------------------------------------------------------------- #
# Vencer
# --------------------------------------------------------------------------- #


def test_el_que_vence_avisa_una_vez_y_se_va(tmp_path: Path) -> None:
    rec.crear(tmp_path, "Té", AHORA + timedelta(minutes=5))
    assert rec.vencidos(tmp_path, AHORA) == []
    vencidos = rec.vencidos(tmp_path, AHORA + timedelta(minutes=5))
    assert [v["texto"] for v in vencidos] == ["Té"]
    assert rec.vencidos(tmp_path, AHORA + timedelta(minutes=6)) == []
    assert rec.pendientes(tmp_path) == []


def test_el_que_se_repite_vuelve_a_su_sitio(tmp_path: Path) -> None:
    rec.crear(tmp_path, "Pastilla", AHORA, repetir="diario")
    assert len(rec.vencidos(tmp_path, AHORA)) == 1
    (queda,) = rec.pendientes(tmp_path)
    assert datetime.fromisoformat(queda["cuando"]) == AHORA + timedelta(days=1)


def test_el_que_vencio_con_el_nucleo_apagado_dice_cuanto_llega_tarde(tmp_path: Path) -> None:
    rec.crear(tmp_path, "Reunión", AHORA)
    (vencido,) = rec.vencidos(tmp_path, AHORA + timedelta(minutes=47))
    assert vencido["retraso_min"] == 47
    assert "47 min tarde" in agente._aviso(vencido)


# --------------------------------------------------------------------------- #
# El agente y el disparador
# --------------------------------------------------------------------------- #


@pytest.fixture()
def iniciado(cfg):
    agente.iniciar(cfg)
    yield cfg
    agente._cfg = None


def _trabajo(**peticion) -> dict:
    return {"peticion": peticion, "quien": "Persus"}


def test_apuntar_repite_la_hora_ya_resuelta(iniciado) -> None:
    r = asyncio.run(agente._recordatorios(_trabajo(accion="crear", texto="Llamar", en_minutos=30)))
    assert r["texto"].startswith("Apuntado: te lo recuerdo hoy a las") or "mañana" in r["texto"]
    assert "«Llamar»" in r["texto"]
    (guardado,) = rec.pendientes(iniciado.directorio_datos)
    assert guardado["quien"] == "Persus"


def test_un_error_de_fecha_vuelve_al_modelo_para_que_se_corrija(iniciado) -> None:
    r = asyncio.run(agente._recordatorios(_trabajo(accion="crear", texto="x", fecha_hora="ayer")))
    assert r["texto"].startswith("No se ha apuntado")
    assert rec.pendientes(iniciado.directorio_datos) == []


def test_listar_y_cancelar_por_el_agente(iniciado) -> None:
    asyncio.run(agente._recordatorios(_trabajo(accion="crear", texto="Regar", en_minutos=90, repetir="diario")))
    lista = asyncio.run(agente._recordatorios(_trabajo(accion="listar")))["texto"]
    assert "Regar" in lista and "cada día" in lista
    quitado = asyncio.run(agente._recordatorios(_trabajo(accion="cancelar", texto="reg")))["texto"]
    assert quitado == "Quitado: «Regar»."


def test_el_titular_de_telegram_no_lleva_el_texto(iniciado) -> None:
    """Titular por Telegram, detalle por Tailscale: lo dictado es contenido suyo."""
    vencido = {"texto": "Secreto de Estado", "cuando": AHORA.isoformat(), "retraso_min": 0}
    r = asyncio.run(agente._recordatorios(_trabajo(accion="avisar", recordatorios=[vencido])))
    assert r["titular"] == "Recordatorio de las 10:00"
    assert "Secreto" not in r["titular"]
    assert "Secreto de Estado" in r["texto"]


def test_el_disparador_encola_el_aviso_de_lo_vencido(cfg, monkeypatch) -> None:
    rec.crear(cfg.directorio_datos, "Ya", datetime.now().astimezone() - timedelta(seconds=1))
    encolados = []

    async def encolar(self, agente_nombre, peticion):
        encolados.append((agente_nombre, peticion))
        return {"id": 1}

    monkeypatch.setattr(disparadores.Contexto, "encolar", encolar)
    asyncio.run(agente._vigilar_recordatorios(disparadores.Contexto(cfg=cfg, bus=Bus())))
    ((nombre, peticion),) = encolados
    assert nombre == "recordatorios" and peticion["accion"] == "avisar"
    assert peticion["recordatorios"][0]["texto"] == "Ya"


# --------------------------------------------------------------------------- #
# Las piezas que lo atan al resto
# --------------------------------------------------------------------------- #


def test_apuntar_y_quitar_son_reversibles_y_mirar_es_libre() -> None:
    assert politica.nivel("recordatorios", {"accion": "crear"}) == politica.REVERSIBLE
    assert politica.nivel("recordatorios", {"accion": "cancelar"}) == politica.REVERSIBLE
    assert politica.nivel("recordatorios", {"accion": "listar"}) == politica.LIBRE
    assert politica.nivel("recordatorios", {"accion": "avisar"}) == politica.LIBRE


def test_las_directas_de_rust_son_herramientas_de_voz_del_catalogo() -> None:
    """La tabla `DIRECTAS` de `nucleo.rs` y el catálogo no pueden separarse."""
    fuente = (Path(__file__).resolve().parent.parent / "RealTime/src-tauri/src/nucleo.rs").read_text(
        encoding="utf-8"
    )
    bloque = fuente[fuente.index("const DIRECTAS") : fuente.index("];", fuente.index("const DIRECTAS"))]
    directas = re.findall(r'\("([a-z_]+)", "([a-z_]+)", "([a-z_]+)"\)', bloque)
    assert directas, "no se encuentra la tabla DIRECTAS"
    voz = set(catalogo.nombres("voz"))
    for herramienta, agente_nombre, accion in directas:
        assert herramienta in voz, f"{herramienta} está en Rust y no en el catálogo de voz"
        assert politica.nivel(agente_nombre, {"accion": accion}) != politica.IRREVERSIBLE or (
            agente_nombre in politica.TABLA
        ), f"{agente_nombre}.{accion} caería en irreversible por no estar en la tabla"
