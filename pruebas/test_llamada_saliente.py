"""Que Perseo llame: recordatorios y encargos por el mismo timbre.

Lo que iba mal, y estas pruebas fijan que no vuelva:

  · el marcador se sobrescribía, y de dos avisos casi a la vez se perdía uno;
  · un encargo que terminaba sin nadie esperándolo —pedido por escrito, o desde
    el móvil— acababa en silencio;
  · y el recordatorio no llamaba: solo dejaba un trabajo en la cola.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from perseo_core.agentes import recordatorios as agente_recordatorios
from perseo_core.caras import telegram
from perseo_core.infra.bus import Bus, Evento
from perseo_core.servicios import llamada_saliente


def _marcador() -> Path:
    return Path(os.environ["PERSEO_MARCADOR_DIR"]) / llamada_saliente.NOMBRE_MARCADOR


def test_las_pruebas_no_tocan_el_marcador_de_verdad() -> None:
    assert llamada_saliente._raiz() != llamada_saliente.RAIZ_REPOSITORIO


def test_dos_avisos_seguidos_no_se_pisan() -> None:
    llamada_saliente.llamar("El encargo #1 ha terminado.")
    llamada_saliente.llamar("Recordatorio: el té")
    assert _marcador().read_text(encoding="utf-8").splitlines() == [
        "El encargo #1 ha terminado.",
        "Recordatorio: el té",
    ]


def test_un_motivo_con_saltos_de_linea_sigue_siendo_una_linea() -> None:
    llamada_saliente.llamar("uno\ndos")
    assert _marcador().read_text(encoding="utf-8") == "uno dos\n"


def test_quien_pregunta_por_un_trabajo_lo_esta_esperando() -> None:
    llamada_saliente.marcar_espera(41)
    assert llamada_saliente.alguien_espera(41)
    assert not llamada_saliente.alguien_espera(42)
    lejos = __import__("time").monotonic() + llamada_saliente.SIN_ESPERA + 1
    assert not llamada_saliente.alguien_espera(41, ahora=lejos)


def test_el_motivo_de_un_encargo_cabe_en_una_linea() -> None:
    trabajo = {"id": 7, "peticion": {"texto": "arregla el login\ny los tests"}, "resultado": {"titular": "Hecho: 3 ficheros\nmás cosas"}}
    assert llamada_saliente.motivo_de_encargo(trabajo, "hecho") == (
        "El encargo #7 (arregla el login y los tests) ha terminado: Hecho: 3 ficheros"
    )
    fallido = {"id": 8, "peticion": {"texto": "x"}, "error": "RuntimeError: sin red"}
    assert "ha fallado: RuntimeError: sin red" in llamada_saliente.motivo_de_encargo(fallido, "fallido")


# --------------------------------------------------------------------------- #
# El avisador
# --------------------------------------------------------------------------- #


class BusQueApunta(Bus):
    def __init__(self) -> None:
        super().__init__()
        self.publicados: list[str] = []

    def publicar(self, tipo: str, **datos):
        self.publicados.append(tipo)
        return super().publicar(tipo, **datos)


def _correr_avisador(eventos: list[Evento], esperar_a: list[int] = ()) -> BusQueApunta:
    bus = BusQueApunta()
    avisador = llamada_saliente.Avisador(bus)
    avisador.gracia = 0.01

    async def todo() -> None:
        tarea = asyncio.create_task(avisador.ejecutar())
        await asyncio.sleep(0.01)
        for evento in eventos:
            for id_trabajo in esperar_a:
                llamada_saliente.marcar_espera(id_trabajo)
            bus.publicar(evento.tipo, **evento.datos)
        await asyncio.sleep(0.2)
        tarea.cancel()

    asyncio.run(todo())
    return bus


def _hecho(id_trabajo: int, agente: str = "dev") -> Evento:
    return Evento(
        tipo="trabajo.hecho",
        datos={"trabajo": {"id": id_trabajo, "agente": agente, "peticion": {"texto": "algo"}, "resultado": {"titular": "listo"}}},
    )


def test_un_encargo_que_termina_sin_nadie_esperando_llama() -> None:
    bus = _correr_avisador([_hecho(901)])
    assert "El encargo #901 (algo) ha terminado: listo" in _marcador().read_text(encoding="utf-8")
    assert "aviso.encargo" in bus.publicados


def test_si_alguien_lo_espera_no_se_cuenta_dos_veces() -> None:
    bus = _correr_avisador([_hecho(902)], esperar_a=[902])
    assert not _marcador().exists()
    assert "aviso.encargo" not in bus.publicados


def test_lo_que_no_es_un_encargo_no_llama() -> None:
    _correr_avisador([_hecho(903, agente="memoria")])
    assert not _marcador().exists()


def test_telegram_dice_el_titular_del_encargo_y_nada_mas() -> None:
    evento = Evento(tipo="aviso.encargo", datos={"trabajo": {"id": 5, "peticion": {"texto": "secreto"}}, "estado": "hecho"})
    texto, _ = telegram.redactar(evento, "https://perseo")
    assert texto == "El encargo #5 ha terminado. Míralo en Perseo."


# --------------------------------------------------------------------------- #
# El recordatorio, por el mismo timbre
# --------------------------------------------------------------------------- #


def test_el_recordatorio_que_vence_hace_sonar_el_timbre(cfg) -> None:
    agente_recordatorios.iniciar(cfg)
    try:
        vencido = {"texto": "Llamar al fontanero", "cuando": "2026-09-23T17:00:00+02:00", "retraso_min": 0}
        asyncio.run(
            agente_recordatorios._recordatorios({"peticion": {"accion": "avisar", "recordatorios": [vencido]}})
        )
    finally:
        agente_recordatorios._cfg = None
    assert _marcador().read_text(encoding="utf-8") == "Recordatorio: Llamar al fontanero\n"
