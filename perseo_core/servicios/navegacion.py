"""Leer lo que devuelve el navegador, sin creérselo.

Piezas puras del agente `recado`, aparte para poder probarlas sin navegador ni
modelo: sacar de una respuesta de `@playwright/mcp` en qué página se está, qué
elemento es cada `ref`, y si pulsar uno de ellos **sale de casa**.

**Por qué el nombre se saca de la instantánea y no de lo que dice el modelo.**
Cada llamada de Playwright lleva un `element`, la descripción del elemento que
escribe el modelo. Eso es lo que el modelo *cree* que pulsa, o lo que una página
hostil le ha convencido de escribir. El nombre accesible que sale de la
instantánea para ese `ref` es lo que el navegador va a pulsar de verdad, y es el
que decide. Se miran los dos, y basta con que uno suene a pagar para parar.
"""

from __future__ import annotations

import re
import unicodedata
import urllib.parse
from dataclasses import dataclass

#: `- button "Pagar 45 €" [ref=e10]`, `- textbox "Email" [ref=e5]: valor`,
#: `- link "Ver carta" [ref=e11] [cursor=pointer]:`. El nombre es opcional:
#: un `generic` no lo lleva.
_LINEA_REF = re.compile(
    r'^\s*-\s+(?P<rol>[a-zA-Z]+)(?:\s+"(?P<nombre>(?:[^"\\]|\\.)*)")?[^\n]*?\[ref=(?P<ref>(?:f\d+)?e\d+)\]',
    re.M,
)
_URL = re.compile(r"^- Page URL:\s*(\S+)", re.M)

#: Un `target` que sea un ref de la instantánea. Playwright acepta además
#: selectores CSS, y eso es justo lo que no se deja: con un selector no hay
#: nombre que mirar en la instantánea, y la comprobación de arriba se queda
#: ciega.
REF = re.compile(r"^(?:f\d+)?e\d+$")


@dataclass(frozen=True)
class Elemento:
    ref: str
    rol: str
    nombre: str
    #: El que tiene el foco (`[active]`): es el que recibe un Enter o un espacio.
    activo: bool = False


def sin_acentos(texto: str) -> str:
    descompuesto = unicodedata.normalize("NFKD", texto or "")
    return "".join(c for c in descompuesto if not unicodedata.combining(c)).lower()


def elementos(instantanea: str) -> dict[str, Elemento]:
    """Cada `ref` de la instantánea, con su rol y su nombre accesible."""
    vistos: dict[str, Elemento] = {}
    for m in _LINEA_REF.finditer(instantanea or ""):
        nombre = (m.group("nombre") or "").replace('\\"', '"')
        activo = "[active]" in m.group(0)
        vistos[m.group("ref")] = Elemento(m.group("ref"), m.group("rol").lower(), nombre, activo)
    return vistos


def activo(elementos_: dict[str, Elemento]) -> Elemento | None:
    """El elemento con el foco, el más interior si hay varios (el último en la lista)."""
    con_foco = [e for e in elementos_.values() if e.activo]
    return con_foco[-1] if con_foco else None


def url_de(respuesta: str) -> str:
    """La URL de la página según la respuesta, o cadena vacía si no la trae."""
    encontrada = _URL.search(respuesta or "")
    return encontrada.group(1).strip() if encontrada else ""


def anfitrion(url: str) -> str:
    try:
        return (urllib.parse.urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


# --------------------------------------------------------------------------- #
# Lo que sale de casa
# --------------------------------------------------------------------------- #

#: Verbos que, en un botón, cierran algo con alguien de fuera. Sin acentos y en
#: minúsculas, porque se comparan contra `sin_acentos(nombre)`.
#:
#: Lo que **no** está, y por qué: «continuar», «siguiente», «buscar» y
#: «aceptar» a secas. Son los pasos de en medio de cualquier web —y «aceptar»
#: es el de las cookies—: pararlos convertiría cada recado en una ristra de
#: preguntas, que es como se aprende a decir que sí sin leer. El paso que
#: compromete de verdad lleva casi siempre uno de los de abajo.
_VERBOS_EXTERIORES = (
    # castellano
    r"\bpag(ar|a|ue)\b", r"realizar (el )?pago", r"\bcomprar\b", r"compra ahora",
    r"reserv(ar|a)\b", r"confirm(ar|a)\b",
    r"envi(ar|a)\b", r"finaliz(ar|a)", r"tramit(ar|a)", r"contrat(ar|a)",
    r"suscrib", r"don(ar|a)\b", r"publicar", r"solicit(ar|a)\b", r"firm(ar|a)\b",
    r"realizar pedido", r"hacer pedido", r"registrar(me|se)", r"crear cuenta",
    r"dar(me|se) de alta", r"dar(me|se) de baja", r"cancelar (suscripcion|pedido|reserva)",
    # inglés
    r"\bpay\b", r"\bbuy\b", r"purchase", r"place order", r"\border now\b", r"\bbook\b",
    r"\breserve\b", r"\bconfirm\b", r"\bsubmit\b", r"\bsend\b", r"subscribe",
    r"\bdonate\b", r"\bpublish\b", r"sign up", r"\bregister\b", r"create account",
    r"unsubscribe", r"cancel (subscription|order|booking|reservation)",
)
_EXTERIOR = re.compile("|".join(f"(?:{v})" for v in _VERBOS_EXTERIORES))


def suena_a_exterior(*textos: str) -> str | None:
    """El trozo que suena a comprometerse, si alguno de los textos lo tiene."""
    for texto in textos:
        encontrado = _EXTERIOR.search(sin_acentos(texto))
        if encontrado:
            return encontrado.group(0)
    return None


#: «45 €», «45,90 EUR», «€45», «$12.50», «1.234,56 €». No pretende leer
#: cualquier formato del mundo: sirve para comparar con un tope, y si no lo
#: entiende devuelve `None` y el tope no decide nada.
_IMPORTE = re.compile(
    r"(?:(?P<pre>[€$£])\s*(?P<a>\d[\d.,\s]*\d|\d))|(?:(?P<b>\d[\d.,\s]*\d|\d)\s*(?P<post>€|eur\b|euros?\b|usd\b|\$|£))",
    re.I,
)


def _a_numero(crudo: str) -> float | None:
    limpio = crudo.replace(" ", "").replace(" ", "")
    if "," in limpio and "." in limpio:
        # El que va último es el decimal: «1.234,56» y «1,234.56».
        if limpio.rfind(",") > limpio.rfind("."):
            limpio = limpio.replace(".", "").replace(",", ".")
        else:
            limpio = limpio.replace(",", "")
    elif "," in limpio:
        entero, _, decimales = limpio.rpartition(",")
        limpio = f"{entero.replace(',', '')}.{decimales}" if len(decimales) <= 2 else limpio.replace(",", "")
    elif limpio.count(".") == 1 and len(limpio.rpartition(".")[2]) == 3:
        limpio = limpio.replace(".", "")  # «1.234» en castellano son mil y pico
    try:
        return float(limpio)
    except ValueError:
        return None


def importe(texto: str) -> float | None:
    """El mayor importe que aparece en el texto, o `None`."""
    mayor: float | None = None
    for m in _IMPORTE.finditer(texto or ""):
        valor = _a_numero(m.group("a") or m.group("b") or "")
        if valor is not None and (mayor is None or valor > mayor):
            mayor = valor
    return mayor


# --------------------------------------------------------------------------- #
# Lo que vuelve al modelo
# --------------------------------------------------------------------------- #

#: Lo que entra de una respuesta. Una web normal da instantáneas de cinco a
#: quince mil caracteres; las que pasan de aquí son listados enormes donde lo
#: que importa suele estar arriba.
TOPE_RESPUESTA = 16000


def recortar(texto: str, tope: int = TOPE_RESPUESTA) -> str:
    if len(texto) <= tope:
        return texto
    return texto[:tope] + f"\n[… recortado: {len(texto) - tope} caracteres más. Si lo que buscas no está, usa browser_snapshot con `target` de la sección.]"


def sin_codigo(respuesta: str) -> str:
    """Quita el bloque «Ran Playwright code»: repite lo pedido y ocupa sitio."""
    return re.sub(r"### Ran Playwright code\n```js\n.*?```\n?", "", respuesta or "", flags=re.S).strip()
