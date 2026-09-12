"""Pruebas de regresión de seguridad del agente `pc`.

Vienen de la herramienta de v1, que las llevaba dentro en un `__main__`. Al mudar
la herramienta al núcleo se mudan también, porque son lo que impide que este
módulo vuelva a ser lo que era antes de.

Cada caso es un intento de inyección que la versión de entonces habría
ejecutado. **Ninguno debe llegar a tocar el sistema**, y ninguno de los que se
comprueban aquí abre ventanas ni teclea nada: todos se rechazan antes.

    python verificadores/verificar_pc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perseo_core import aplicaciones, pc  # noqa: E402
from verificadores.arnes_pruebas import comprobar, resumir  # noqa: E402

INYECCIONES = [
    ("abrir_app", "notepad & calc", "encadenado con &"),
    ("abrir_app", "notepad && shutdown /s /t 0", "encadenado con &&"),
    ("abrir_app", "a | del /q C:\\*", "tuberia"),
    ("abrir_app", "cmd", "shell fuera de la lista"),
    ("abrir_app", "powershell", "shell fuera de la lista"),
    ("abrir_app", "file:///C:/Windows", "esquema no permitido"),
    ("abrir_app", "javascript:alert(1)", "esquema no permitido"),
    ("atajo_teclado", "ctrl,alt,delete,f4,esc,tab", "demasiadas teclas"),
    ("atajo_teclado", "ctrl,shutdown", "tecla inventada"),
    ("mover_raton", "abc,def", "coordenadas no numericas"),
    ("volumen", "; rm -rf /", "valor no reconocido"),
    ("click_raton", "triple; shutdown", "tipo de clic inventado"),
    ("apagar_equipo", "ya", "accion inexistente"),
]


def main() -> None:
    print("Intentos de inyeccion — ninguno debe ejecutarse\n")
    for accion, parametro, motivo in INYECCIONES:
        resultado = pc.controlar(accion, parametro)
        comprobar(f"Bloqueado: {motivo}", resultado.startswith("Error:"), f"{accion}({parametro!r})")

    print("\nSaneado del texto a teclear (sin teclear nada de verdad)\n")
    for entrada, motivo in [
        ("formatear\nsi", "salto de linea"),
        ("dato\ty\rotro", "tabulador y retorno"),
    ]:
        salida = pc._texto_imprimible(entrada)
        comprobar(
            f"Se quita el control: {motivo}",
            not any(c in salida for c in "\n\r\t"),
            f"{entrada!r} -> {salida!r}",
        )

    print("\nLimites\n")
    comprobar(
        "Un texto larguisimo se rechaza",
        pc.controlar("escribir_teclado", "x" * (pc.MAX_LONGITUD_TEXTO + 1)).startswith("Error:"),
    )
    comprobar(
        "Y uno que solo tiene caracteres de control, tambien",
        pc.controlar("escribir_teclado", "\n\t\r").startswith("Error:"),
    )
    comprobar(
        "La lista blanca no lleva ningun interprete",
        not ({"cmd", "powershell", "wt", "terminal", "regedit"} & set(aplicaciones.APLICACIONES_PERMITIDAS)),
        ", ".join(sorted(aplicaciones.APLICACIONES_PERMITIDAS)),
    )
    comprobar(
        "Ni las teclas permitidas incluyen ninguna de sistema",
        "printscreen" not in pc._TECLAS_PERMITIDAS and "volumemute" not in pc._TECLAS_PERMITIDAS,
    )

    resumir()


if __name__ == "__main__":
    main()
