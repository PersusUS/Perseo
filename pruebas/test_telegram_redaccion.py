"""Qué se manda al móvil y qué no.

Estas reglas son lo que más se va a discutir de `telegram.py` —el señor Persus
las mandó replantear el 2026-08-22 porque los avisos «no servían para nada»— y
son también lo que más fácil se rompe sin enterarse: un aviso de más no falla,
solo hace que el canal se silencie. Por eso están fijadas aquí.
"""

from perseo_core.infra.bus import Evento
from perseo_core.caras.telegram import redactar, resumir_peticion

URL = "http://perseo.tailnet:8787"


def evento(tipo: str, **trabajo) -> Evento:
    trabajo.setdefault("id", 7)
    trabajo.setdefault("agente", "correo")
    trabajo.setdefault("origen", "disparador")
    return Evento(tipo=tipo, datos={"trabajo": trabajo})


def test_lo_que_espera_un_si_siempre_se_manda() -> None:
    mensaje = redactar(
        evento("trabajo.espera_confirmacion", confirmacion={"resumen": "Enviar el correo a Ana"}),
        URL,
    )
    assert mensaje is not None
    texto, botones = mensaje
    assert "Enviar el correo a Ana" in texto
    # El pie dice de quién y cuál: sin eso había que abrir la web para saberlo.
    assert "correo · #7" in texto


def test_ningun_mensaje_lleva_botones_de_decision() -> None:
    """Desde N-1 (2026-08-22) Telegram solo avisa: la decisión se da por voz en
    una llamada o en las pantallas. Si algún día vuelve a aparecer aquí un
    `callback_data`, es que alguien está devolviendo al canal lo que se quitó."""
    casos = [
        evento("trabajo.espera_confirmacion", confirmacion={"resumen": "pregunta"}),
        evento("trabajo.fallido", error="x", peticion={"texto": "y"}),
        evento("trabajo.hecho", resultado={"titular": "titular"}),
    ]
    for caso in casos:
        mensaje = redactar(caso, URL)
        if mensaje is None:
            continue
        planos = [b for fila in mensaje[1] for b in fila]
        assert not any("callback_data" in b for b in planos), str(planos)
        assert all(b.get("url", "").startswith("http") for b in planos), str(planos)


def test_el_detalle_no_sale_por_telegram() -> None:
    """Titular por Telegram, detalle por Tailscale: es la regla del módulo."""
    mensaje = redactar(
        evento(
            "trabajo.espera_confirmacion",
            confirmacion={"resumen": "Enviar el correo", "detalle": "Querida Ana, el informe…"},
        ),
        URL,
    )
    assert mensaje is not None
    assert "Querida Ana" not in mensaje[0]


def test_un_fallo_siempre_avisa_y_trae_la_primera_linea() -> None:
    mensaje = redactar(
        evento("trabajo.fallido", error="ConnectionError: no hay red\n  File ...\n  File ...",
               peticion={"texto": "resumir el correo"}),
        URL,
    )
    assert mensaje is not None
    assert "Falló: resumir el correo" in mensaje[0]
    assert "ConnectionError: no hay red" in mensaje[0]
    # La traza entera no: no cabe en una notificación y no se lee en el móvil.
    assert "File ..." not in mensaje[0]


def test_lo_que_termina_sin_titular_no_molesta() -> None:
    assert redactar(evento("trabajo.hecho", resultado={"ruta": "x.md"}), URL) is None


def test_lo_que_termina_con_titular_si() -> None:
    mensaje = redactar(
        evento("trabajo.hecho", resultado={"titular": "3 correos, 1 requiere acción"}), URL
    )
    assert mensaje is not None
    assert mensaje[0].startswith("3 correos, 1 requiere acción")


def test_lo_que_lanzaste_tu_no_se_reenvia_al_movil() -> None:
    """Si lo encolaste desde el panel o la llamada, estás mirando la pantalla."""
    mensaje = redactar(
        evento("trabajo.hecho", origen="panel", resultado={"titular": "Hecho"}), URL
    )
    assert mensaje is None


def test_cancelado_y_rechazado_no_se_anuncian() -> None:
    assert redactar(evento("trabajo.rechazado"), URL) is None
    assert redactar(evento("trabajo.cancelado"), URL) is None


def test_sin_trabajo_no_hay_mensaje() -> None:
    assert redactar(Evento(tipo="trabajo.hecho", datos={}), URL) is None


def test_un_lote_de_correo_no_se_vuelca_entero() -> None:
    """El trabajo de correo trae el buzón dentro; volcarlo manda veinte correos
    por un canal de terceros."""
    resumen = resumir_peticion({"peticion": {"mensajes": [{"id": 1}, {"id": 2}]}})
    assert resumen == "2 correos del buzón"
