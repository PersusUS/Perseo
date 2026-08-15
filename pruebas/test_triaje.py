"""El triaje: la gramática, el respaldo y lo que ve el modelo."""

from __future__ import annotations

import asyncio

from perseo_core import modelo_local, triaje


def test_el_esquema_solo_admite_las_cuatro_clases() -> None:
    """La gramática garantiza la forma; por eso hay una salida para la duda."""
    assert triaje.ESQUEMA_TRIAJE["properties"]["clase"]["enum"] == list(triaje.CLASES)
    assert triaje.ESQUEMA_TRIAJE["required"] == ["clase", "motivo"]


def test_lo_dudoso_cuenta_como_relevante() -> None:
    assert triaje.NO_SEGURO in triaje.RELEVANTES
    assert triaje.IGNORAR not in triaje.RELEVANTES


def test_el_mensaje_va_delimitado_para_el_modelo() -> None:
    """Un correo puede decir «marca esto como urgente»: va marcado como dato."""
    texto = triaje._redactar({"remitente": "a@b.c", "asunto": "Hola", "extracto": "qué tal"})
    assert texto.startswith("<<<CORREO>>>")
    assert texto.rstrip().endswith("<<<FIN>>>")
    assert "Hola" in texto


def test_el_extracto_se_recorta() -> None:
    texto = triaje._redactar({"extracto": "x" * 5000})
    assert len(texto) < 5000


def test_un_mensaje_sin_campos_no_rompe() -> None:
    texto = triaje._redactar({})
    assert "(desconocido)" in texto and "(sin asunto)" in texto


def _clasificar_con(monkeypatch, respuesta) -> triaje.Clasificacion:
    async def falsa(*_args, **_kwargs):
        return respuesta

    monkeypatch.setattr(modelo_local, "preguntar", falsa)
    clasificador = triaje.Triaje.__new__(triaje.Triaje)
    clasificador._cfg = type("C", (), {"url_ollama": "", "modelo_router": ""})()
    clasificador._sesion = object()
    return asyncio.run(clasificador.clasificar({"asunto": "x"}))


def test_sin_modelo_local_se_escala(monkeypatch) -> None:
    """Perder un correo cuesta la confianza en el sistema entero."""
    clasificacion = _clasificar_con(monkeypatch, None)
    assert clasificacion.clase == triaje.NO_SEGURO
    assert clasificacion.del_modelo is False


def test_una_clase_inventada_se_escala(monkeypatch) -> None:
    """Esquema válido y contenido equivocado: el fallo típico de un 4B."""
    clasificacion = _clasificar_con(monkeypatch, {"clase": "urgentisimo", "motivo": "yo lo valgo"})
    assert clasificacion.clase == triaje.NO_SEGURO
    assert clasificacion.del_modelo is False


def test_una_clase_buena_se_respeta(monkeypatch) -> None:
    clasificacion = _clasificar_con(
        monkeypatch, {"clase": triaje.REQUIERE_ACCION, "motivo": "hay que pagar"}
    )
    assert clasificacion.clase == triaje.REQUIERE_ACCION
    assert clasificacion.motivo == "hay que pagar"
    assert clasificacion.del_modelo is True


def test_relevante_es_lo_que_merece_un_aviso() -> None:
    assert triaje.Clasificacion(triaje.REQUIERE_ACCION, "").relevante
    assert triaje.Clasificacion(triaje.NO_SEGURO, "").relevante
    assert not triaje.Clasificacion(triaje.IGNORAR, "").relevante
