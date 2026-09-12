"""Las etiquetas del triaje de correo, y el resultado de ponerlas.

El vocabulario baja aquí con el tipo: `Clasificacion.relevante` se define contra
`RELEVANTES`, así que separarlos dejaría al dominio mirando hacia arriba. Quién
decide **cuál** de estas etiquetas poner sigue siendo cosa de
`servicios/triaje.py`; qué etiquetas existen es del dominio.
"""

from __future__ import annotations

from dataclasses import dataclass


IGNORAR = "ignorar"
INTERESANTE = "interesante"
REQUIERE_ACCION = "requiere_accion"
NO_SEGURO = "no_seguro"

CLASES = (IGNORAR, INTERESANTE, REQUIERE_ACCION, NO_SEGURO)

#: Clases que merecen un aviso. `no_seguro` está dentro a propósito: ante la
#: duda, que lo mire una persona.
RELEVANTES = (INTERESANTE, REQUIERE_ACCION, NO_SEGURO)


@dataclass(frozen=True)
class Clasificacion:
    """El resultado de triar un mensaje."""

    clase: str
    motivo: str
    #: Falso cuando la etiqueta no la puso el modelo sino el respaldo. Se guarda
    #: porque un día de triaje entero sin modelo local se tiene que notar.
    del_modelo: bool = True

    @property
    def relevante(self) -> bool:
        return self.clase in RELEVANTES
