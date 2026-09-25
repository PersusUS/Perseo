"""Leer la instantánea de Playwright y decidir qué sale de casa, sin navegador."""

from __future__ import annotations

import pytest

from perseo_core.servicios import navegacion as nav

INSTANTANEA = """### Page
- Page URL: https://www.resy.com/reservar?x=1
### Snapshot
```yaml
- generic [active] [ref=f1e1]:
  - heading "Mesa del viernes" [level=1] [ref=f1e2]
  - textbox "Email" [ref=e5]: ana@correo.es
  - button "Reservar mesa para 2 · 45 €" [ref=f1e4]
  - link "Ver carta" [ref=e11] [cursor=pointer]:
```"""


def test_saca_cada_ref_con_su_rol_y_su_nombre() -> None:
    elementos = nav.elementos(INSTANTANEA)
    assert elementos["e5"] == nav.Elemento("e5", "textbox", "Email")
    assert elementos["e11"].rol == "link"


def test_los_refs_de_un_marco_tambien() -> None:
    """Tras navegar, Playwright numera por marco: `f1e4`. Sin esto no se encontraba el botón."""
    assert nav.elementos(INSTANTANEA)["f1e4"].nombre == "Reservar mesa para 2 · 45 €"
    assert nav.REF.match("f1e4") and nav.REF.match("e12")


@pytest.mark.parametrize("objetivo", ["button:has-text('Pagar')", "#pagar", "e12 >> x", ""])
def test_un_selector_no_es_un_ref(objetivo: str) -> None:
    assert not nav.REF.match(objetivo)


def test_la_url_y_el_anfitrion() -> None:
    assert nav.url_de(INSTANTANEA) == "https://www.resy.com/reservar?x=1"
    assert nav.anfitrion(nav.url_de(INSTANTANEA)) == "www.resy.com"


@pytest.mark.parametrize(
    "boton",
    ["Pagar 45 €", "Confirmar reserva", "Reservar mesa", "Realizar pago", "Enviar mensaje",
     "Finalizar compra", "Place order", "Pay $12.50", "Book now", "Subscribe", "Darme de baja"],
)
def test_lo_que_compromete_suena_a_exterior(boton: str) -> None:
    assert nav.suena_a_exterior(boton)


@pytest.mark.parametrize(
    "boton",
    ["Continuar", "Siguiente", "Buscar", "Aceptar cookies", "Mis reservas", "Métodos de pago",
     "Carrito de compra", "Página 2", "Ver carta"],
)
def test_los_pasos_de_en_medio_no(boton: str) -> None:
    """Pararlos convertiría cada recado en una ristra de preguntas."""
    assert nav.suena_a_exterior(boton) is None


def test_basta_con_que_uno_de_los_dos_nombres_suene() -> None:
    """El de la página y el que escribe el modelo: si cualquiera suena a pagar, se para."""
    assert nav.suena_a_exterior("Continuar", "Pagar ahora")
    assert nav.suena_a_exterior("Pagar 45 €", "continue")


@pytest.mark.parametrize(
    "texto, esperado",
    [("Pagar 45 €", 45.0), ("45,90 EUR", 45.9), ("€ 12", 12.0), ("Pay $12.50", 12.5),
     ("1.234,56 €", 1234.56), ("1,234.56 $", 1234.56), ("Reservar", None)],
)
def test_el_importe(texto: str, esperado: float | None) -> None:
    assert nav.importe(texto) == esperado


def test_recortar_avisa_de_lo_que_falta() -> None:
    recortado = nav.recortar("x" * 50, tope=10)
    assert recortado.startswith("x" * 10) and "40 caracteres más" in recortado


def test_sin_codigo_quita_lo_que_repite_playwright() -> None:
    respuesta = "### Ran Playwright code\n```js\nawait page.fill('S3creta!');\n```\n### Page\n- Page URL: x"
    assert "S3creta" not in nav.sin_codigo(respuesta)
    assert "Page URL" in nav.sin_codigo(respuesta)
