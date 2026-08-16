"""Lo que hace falta para configurar el canal: sacar el `chat_id` de la nada.

El resto del módulo —el filtro del chat, los botones, la regla de "titular por
Telegram, detalle por Tailscale"— se comprueba de punta a punta en
`verificar_telegram.py`, contra un Telegram de mentira.
"""

from __future__ import annotations

from perseo_core import telegram


def actualizacion(chat: dict, envoltorio: str = "message") -> dict:
    return {"update_id": 1, envoltorio: {"chat": chat, "text": "/start"}}


def test_un_chat_que_escribe_aparece_con_su_nombre() -> None:
    chats = telegram.chats_vistos([actualizacion({"id": 12345, "first_name": "Jesús"})])
    assert chats == [("12345", "Jesús")]


def test_el_identificador_sale_como_texto() -> None:
    """Se compara con lo que hay en el fichero, que es texto: 12345 != '12345'."""
    (identificador, _), = telegram.chats_vistos([actualizacion({"id": 12345})])
    assert identificador == "12345"


def test_el_usuario_se_añade_al_nombre() -> None:
    chats = telegram.chats_vistos(
        [actualizacion({"id": 1, "first_name": "Jesús", "username": "persus"})]
    )
    assert chats == [("1", "Jesús (@persus)")]


def test_un_chat_sin_nombre_no_deja_la_linea_vacia() -> None:
    assert telegram.chats_vistos([actualizacion({"id": 7})]) == [("7", "sin nombre")]


def test_el_mismo_chat_dos_veces_sale_una() -> None:
    chats = telegram.chats_vistos(
        [actualizacion({"id": 1, "first_name": "A"}), actualizacion({"id": 1, "first_name": "A"})]
    )
    assert len(chats) == 1


def test_varios_chats_salen_en_orden_de_aparicion() -> None:
    """Con más de uno hay que elegir, y elegir mal es dejar entrar a un desconocido."""
    chats = telegram.chats_vistos(
        [actualizacion({"id": 9, "first_name": "Otro"}), actualizacion({"id": 1, "first_name": "Yo"})]
    )
    assert [i for i, _ in chats] == ["9", "1"]


def test_el_chat_tambien_se_ve_en_un_mensaje_editado() -> None:
    chats = telegram.chats_vistos([actualizacion({"id": 3}, envoltorio="edited_message")])
    assert [i for i, _ in chats] == ["3"]


def test_el_chat_tambien_se_ve_en_la_pulsacion_de_un_boton() -> None:
    pulsacion = {"update_id": 1, "callback_query": {"message": {"chat": {"id": 4}}}}
    assert [i for i, _ in telegram.chats_vistos([pulsacion])] == ["4"]


def test_lo_que_no_trae_chat_no_rompe_nada() -> None:
    basura = ["no soy un dict", {}, {"message": {}}, {"message": {"chat": {}}}, {"callback_query": {}}]
    assert telegram.chats_vistos(basura) == []
