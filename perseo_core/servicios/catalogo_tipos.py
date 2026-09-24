"""La forma de una herramienta del catálogo: sus parámetros y su texto por cara.

Aparte de `catalogo.py` para que el catálogo pueda repartirse en varios
ficheros sin importarse en círculo: todos importan de aquí.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Parametro:
    """Un argumento de una herramienta.

    `voz` y `chat` son la misma explicación contada para cada cara. Si solo hay
    una, vale para las dos: que una cara no tenga texto propio no significa que
    no tenga el parámetro.
    """

    nombre: str
    tipo: str
    voz: str = ""
    chat: str = ""
    #: El `enum` del esquema. Uno solo, y por eso ya no puede haber dos listas
    #: de acciones distintas para `controlar_pc`.
    opciones: tuple[str, ...] = ()
    obligatorio: bool = False

    def descripcion(self, cara: str) -> str:
        propia = self.voz if cara == "voz" else self.chat
        return propia or self.chat or self.voz


@dataclass(frozen=True)
class Herramienta:
    """Una herramienta, con una descripción por cara.

    Una cadena vacía significa **esta cara no la tiene**, y es información, no
    un olvido: `ver_pantalla` necesita ojos y solo existe en la llamada;
    `encargar_codigo` tarda minutos y solo existe por escrito.
    """

    nombre: str
    voz: str = ""
    chat: str = ""
    parametros: tuple[Parametro, ...] = ()

    def esta_en(self, cara: str) -> bool:
        return bool(self.voz if cara == "voz" else self.chat)

    def descripcion(self, cara: str) -> str:
        return self.voz if cara == "voz" else self.chat
