"""El suplente: quién responde cuando el modelo de casa no está.

Lo que se comprueba aquí es el orden y los límites, no que Gemma acierte. El
orden importa porque el local es gratis y no manda nada fuera; los límites,
porque sin gramática la respuesta puede llegar de cualquier forma.
"""

from __future__ import annotations

from perseo_core import modelo_local


def test_sin_clave_no_hay_suplente() -> None:
    assert not modelo_local.Suplente(clave="", modelo="gemma-4-31b-it").utilizable


def test_sin_modelo_tampoco() -> None:
    """Vacío es apagado, y es lo que viene de fábrica: manda texto a un tercero."""
    assert not modelo_local.Suplente(clave="una-clave", modelo="").utilizable


def test_con_los_dos_si() -> None:
    assert modelo_local.Suplente(clave="una-clave", modelo="gemma-4-31b-it").utilizable


def test_un_json_pelado_se_lee() -> None:
    assert modelo_local.primer_objeto('{"clase": "ignorar"}') == {"clase": "ignorar"}


def test_un_json_envuelto_en_vallas_se_lee() -> None:
    """Un modelo grande sin gramática contesta bien y envuelto más veces de las
    que uno espera; exigirlo pelado tiraría respuestas correctas."""
    crudo = 'Claro:\n```json\n{"clase": "requiere_accion"}\n```\nEspero que ayude.'
    assert modelo_local.primer_objeto(crudo) == {"clase": "requiere_accion"}


def test_una_valla_sin_etiqueta_tambien() -> None:
    assert modelo_local.primer_objeto('```\n{"a": 1}\n```') == {"a": 1}


def test_un_json_con_una_frase_delante_se_rescata() -> None:
    assert modelo_local.primer_objeto('El resultado es {"a": 1}') == {"a": 1}


def test_una_lista_no_vale() -> None:
    """El resto del sistema espera un objeto: una lista es un fallo, no un dato."""
    assert modelo_local.primer_objeto('[1, 2, 3]') is None


def test_texto_sin_json_no_inventa_nada() -> None:
    assert modelo_local.primer_objeto("No he entendido la pregunta.") is None


def test_texto_vacio() -> None:
    assert modelo_local.primer_objeto("") is None


def test_una_llave_suelta_no_rompe() -> None:
    assert modelo_local.primer_objeto("{esto no cierra") is None
