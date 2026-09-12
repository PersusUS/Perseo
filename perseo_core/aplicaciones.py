"""Qué se puede abrir en esta máquina, y cómo se abre sin pasar por un shell.

La lista blanca vive aquí y no dentro del agente `pc` por dos razones. La de
forma: `proyectos` también la necesita, y un servicio no debe tirar de un
agente. La de fondo: esta lista **es** la frontera de seguridad de todo lo que
Perseo puede lanzar, y una frontera se guarda en un sitio con nombre, no dentro
del primero que la usó.

Lo que no está aquí no se abre. Lo que está, se abre sin shell: `Popen` con una
lista de argumentos, o el manejador de protocolo que Windows tenga registrado.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import urllib.parse
import webbrowser
from pathlib import Path

logger = logging.getLogger(__name__)


# ─── Lista blanca de aplicaciones ──────────────────────────────────────────
#
# ("exe", ...) se lanza como proceso con una lista de argumentos.
# ("uri", ...) se abre con el manejador de protocolo registrado en Windows.
#
# Deliberadamente EXCLUIDOS, y no por olvido: cmd, powershell, terminal, wt,
# regedit, y cualquier intérprete. Poder abrir un shell haría inútil el resto de
# este archivo, porque el modelo podría teclear dentro de él con
# `escribir_teclado`.
APLICACIONES_PERMITIDAS: dict[str, tuple[str, str]] = {
    "notepad": ("exe", "notepad.exe"),
    "bloc de notas": ("exe", "notepad.exe"),
    "calculadora": ("exe", "calc.exe"),
    "calc": ("exe", "calc.exe"),
    "paint": ("exe", "mspaint.exe"),
    "explorador": ("exe", "explorer.exe"),
    "chrome": ("exe", "chrome.exe"),
    "firefox": ("exe", "firefox.exe"),
    "edge": ("uri", "microsoft-edge:"),
    "obsidian": ("uri", "obsidian:"),
    "ajustes": ("uri", "ms-settings:"),
    "correo": ("uri", "mailto:"),
    # Ampliado el 2026-08-23 a petición del señor Persus: que el modo live
    # pueda abrir lo que se usa. Los esquemas URI solo saltan si el programa
    # registró el suyo; si no está instalado, el intento falla con un error
    # claro y no pasa nada.
    "word": ("uri", "ms-word:"),
    "excel": ("uri", "ms-excel:"),
    "powerpoint": ("uri", "ms-powerpoint:"),
    "vscode": ("uri", "vscode:"),
    "visual studio code": ("uri", "vscode:"),
    "whatsapp": ("uri", "whatsapp:"),
    "telegram": ("uri", "telegram:"),
    "steam": ("uri", "steam:"),
}

ESQUEMAS_URL_PERMITIDOS = frozenset({"http", "https"})


def es_url(texto: str) -> bool:
    return texto.lower().startswith(("http://", "https://", "www."))


def abrir_url(url: str) -> str:
    """Abre una URL tras validar su esquema. No pasa por el shell."""
    if url.lower().startswith("www."):
        url = "https://" + url

    partes = urllib.parse.urlparse(url)
    if partes.scheme.lower() not in ESQUEMAS_URL_PERMITIDOS:
        return (
            f"Error: esquema de URL no permitido ('{partes.scheme}'). "
            f"Solo se admiten: {', '.join(sorted(ESQUEMAS_URL_PERMITIDOS))}."
        )
    if not partes.netloc:
        return "Error: la URL no tiene un dominio válido."

    webbrowser.open(url)
    return f"Éxito: se ha abierto '{url}' en el navegador."


def _en_app_paths(nombre: str) -> str | None:
    """Dónde dice Windows que está un programa, según «App Paths».

    Es la lista que usa el diálogo Ejecutar, y la razón de que escribir
    `chrome.exe` ahí funcione mientras `CreateProcess` —que es lo que hay debajo
    de `subprocess`— falla con «no se encuentra el archivo»: Chrome no está en el
    PATH y nunca lo estuvo. Costó una llamada entera el 2026-08-17.

    Solo se consulta con nombres que ya han pasado la lista blanca.
    """
    if os.name != "nt":
        return None
    try:
        import winreg  # noqa: PLC0415  (solo existe en Windows)

        for raiz in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            try:
                clave = winreg.OpenKey(
                    raiz, rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{nombre}"
                )
            except OSError:
                continue
            with clave:
                ruta, _ = winreg.QueryValueEx(clave, "")
                if ruta and Path(ruta).is_file():
                    return str(ruta)
    except (OSError, ImportError) as e:
        logger.warning("No se pudo consultar App Paths para %s: %s", nombre, e)
    return None


def resolver_ejecutable(objetivo: str) -> str | None:
    """La ruta completa de un programa de la lista blanca, o `None`.

    Se mira el PATH primero y el registro después, que es el orden en el que lo
    haría una persona escribiendo el nombre.
    """
    return shutil.which(objetivo) or _en_app_paths(objetivo)


def lanzar(tipo: str, objetivo: str) -> None:
    """Lanza una aplicación sin shell. `objetivo` viene de la lista blanca."""
    if tipo == "uri":
        webbrowser.open(objetivo)
        return

    # Con la ruta completa y no con el nombre: `Popen(["chrome.exe"])` solo
    # funciona si está en el PATH, y los navegadores no lo están.
    ruta = resolver_ejecutable(objetivo)
    if ruta is None:
        raise FileNotFoundError(f"No se encuentra '{objetivo}' en esta máquina.")
    subprocess.Popen([ruta], shell=False)
