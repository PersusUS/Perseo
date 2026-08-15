"""Agente `pc`: abrir aplicaciones, teclear y mover el ratón.

Viene de la herramienta de control del PC de v1, que se jubiló con el puente de
tuberías. Lo que se mueve no es solo el código: es su regla de seguridad, que
sigue siendo la razón de que este módulo esté escrito como está.

REGLA DE SEGURIDAD DE ESTE MÓDULO
---------------------------------
Todo lo que llega aquí es texto generado por un modelo que además está mirando la
pantalla y la cámara. Cualquier texto visible —una web, un correo abierto, un
PDF— puede acabar influyendo en estos argumentos. Y desde la Fase D hay una vía
más: un correo entra, lo tría un agente, y su contenido acaba en la cola. Por lo
tanto, aquí dentro:

  1. **Nunca se invoca un shell.** Ni `os.system`, ni `shell=True`. El shell es
     lo que convierte "abrir spotify & formatear" en dos comandos en vez de uno.
  2. **Nunca se interpola texto del modelo dentro de una cadena de comando.**
     Siempre listas de argumentos, que el sistema operativo no vuelve a parsear.
  3. **Solo objetivos de una lista blanca explícita.** Si no está en la lista, no
     se ejecuta: no hay ruta de escape "por si acaso".

Ver bitacora/02_HALLAZGOS.md H-16, y §7 del plan para la política por niveles que
traerá el resto de la Fase E — hoy este agente se comporta igual que se comportaba
la herramienta: sin pedir confirmación.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import urllib.parse
import webbrowser
from typing import Any

from .agentes import registrar

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
    "spotify": ("uri", "spotify:"),
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
}

ESQUEMAS_URL_PERMITIDOS = frozenset({"http", "https"})

# Teclas admitidas en `atajo_teclado`. Restringido a propósito: modificadores,
# letras, dígitos, navegación y funciones. Sin teclas de sistema.
_MODIFICADORES = frozenset({"ctrl", "alt", "shift", "win"})
_TECLAS_SIMPLES = frozenset(
    list("abcdefghijklmnopqrstuvwxyz0123456789")
    + ["f%d" % n for n in range(1, 13)]
    + [
        "enter", "tab", "esc", "escape", "space", "backspace", "delete",
        "home", "end", "pageup", "pagedown", "up", "down", "left", "right",
    ]
)
_TECLAS_PERMITIDAS = _MODIFICADORES | _TECLAS_SIMPLES

MAX_TECLAS_ATAJO = 4
MAX_LONGITUD_TEXTO = 500


# ─── Utilidades internas ───────────────────────────────────────────────────


def _es_url(texto: str) -> bool:
    return texto.lower().startswith(("http://", "https://", "www."))


def _abrir_url(url: str) -> str:
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


def _lanzar(tipo: str, objetivo: str) -> None:
    """Lanza una aplicación sin shell. `objetivo` viene de la lista blanca."""
    if tipo == "uri":
        webbrowser.open(objetivo)
    else:
        subprocess.Popen([objetivo], shell=False)


def _texto_imprimible(texto: str) -> str:
    """Descarta los caracteres de control de un texto a teclear.

    Un '\\n' equivale a pulsar Enter, lo que permitiría confirmar un diálogo o
    ejecutar una línea en cualquier ventana que tuviera el foco. Se extrae como
    función aparte para poder verificarla sin teclear nada de verdad.
    """
    return "".join(c for c in texto if c.isprintable())


def _pyautogui() -> Any:
    """Carga pyautogui al usarlo, no al importar el módulo.

    El núcleo arranca en máquinas sin escritorio —la Raspberry Pi del plan es
    una— y ahí importar pyautogui falla. Que falle una acción del ratón es
    aceptable; que no arranque el núcleo, no.
    """
    import pyautogui  # noqa: PLC0415  (deliberadamente perezoso)

    return pyautogui


# ─── La acción, sin nada de asyncio ────────────────────────────────────────


def controlar(accion: str, parametro: str = "") -> str:
    """Controla el PC bajo Windows. Devuelve 'Éxito: …' o 'Error: …'.

    Es una función normal y no una corrutina a propósito: así se puede verificar
    entera —incluidos los intentos de inyección— sin levantar nada.
    """
    try:
        accion = (accion or "").strip().lower()
        parametro = parametro or ""

        if accion == "abrir_app":
            objetivo = parametro.strip()

            if _es_url(objetivo):
                return _abrir_url(objetivo)

            clave = objetivo.lower()
            if clave not in APLICACIONES_PERMITIDAS:
                permitidas = ", ".join(sorted(APLICACIONES_PERMITIDAS))
                return (
                    f"Error: '{objetivo}' no está en la lista de aplicaciones "
                    f"permitidas. Disponibles: {permitidas}."
                )

            tipo, destino = APLICACIONES_PERMITIDAS[clave]
            _lanzar(tipo, destino)
            return f"Éxito: se ha abierto '{clave}'."

        if accion == "escribir_teclado":
            if len(parametro) > MAX_LONGITUD_TEXTO:
                return f"Error: el texto supera el límite de {MAX_LONGITUD_TEXTO} caracteres."

            texto = _texto_imprimible(parametro)
            if not texto:
                return "Error: no queda texto imprimible que teclear."

            try:
                import time

                pyautogui = _pyautogui()
            except ImportError:
                return "Error: la librería 'pyautogui' no está instalada."

            time.sleep(1)  # cortesía: la ventana destino puede estar abriéndose
            pyautogui.write(texto, interval=0.01)
            return f"Éxito: se ha tecleado el texto '{texto}' en la ventana actual."

        if accion == "atajo_teclado":
            teclas = [t.strip().lower() for t in parametro.split(",") if t.strip()]

            if not teclas:
                return "Error: no se ha indicado ninguna tecla."
            if len(teclas) > MAX_TECLAS_ATAJO:
                return f"Error: máximo {MAX_TECLAS_ATAJO} teclas por atajo."

            no_validas = [t for t in teclas if t not in _TECLAS_PERMITIDAS]
            if no_validas:
                return f"Error: teclas no permitidas: {', '.join(no_validas)}."

            try:
                pyautogui = _pyautogui()
            except ImportError:
                return "Error: la librería 'pyautogui' no está instalada."

            pyautogui.hotkey(*teclas)
            return f"Éxito: se ha ejecutado el atajo '{'+'.join(teclas)}'."

        if accion == "volumen":
            modo = parametro.strip().lower()
            if modo not in ("subir", "bajar", "mutear", "silenciar"):
                # Se valida **antes** de cargar pyautogui: así el rechazo de un
                # valor inventado no depende de que la librería esté instalada.
                return "Error: valor de volumen no reconocido. Use 'subir', 'bajar' o 'mutear'."

            try:
                pyautogui = _pyautogui()
            except ImportError:
                return "Error: la librería 'pyautogui' no está instalada."

            if modo == "subir":
                pyautogui.press("volumeup", presses=5)
            elif modo == "bajar":
                pyautogui.press("volumedown", presses=5)
            else:
                pyautogui.press("volumemute")
            return f"Éxito: acción de volumen '{modo}' enviada al sistema."

        if accion == "mover_raton":
            coords = parametro.split(",")
            if len(coords) != 2:
                return "Error: formato de coordenadas inválido. Debe ser 'x,y'."
            try:
                x, y = int(coords[0].strip()), int(coords[1].strip())
            except ValueError:
                return "Error: las coordenadas deben ser números enteros."

            try:
                pyautogui = _pyautogui()
            except ImportError:
                return "Error: la librería 'pyautogui' no está instalada."

            ancho, alto = pyautogui.size()
            x = max(0, min(x, ancho - 1))
            y = max(0, min(y, alto - 1))
            pyautogui.moveTo(x, y, duration=0.5)
            return f"Éxito: ratón movido a ({x}, {y})."

        if accion == "click_raton":
            tipo = parametro.strip().lower()
            if tipo not in ("", "izquierdo", "derecho", "doble"):
                return "Error: tipo de clic no reconocido. Use 'izquierdo', 'derecho' o 'doble'."

            try:
                pyautogui = _pyautogui()
            except ImportError:
                return "Error: la librería 'pyautogui' no está instalada."

            if tipo == "derecho":
                pyautogui.rightClick()
            elif tipo == "doble":
                pyautogui.doubleClick()
            else:
                pyautogui.click()
            return f"Éxito: clic '{tipo or 'izquierdo'}' ejecutado."

        if accion == "buscar_youtube":
            termino = parametro.strip()
            if not termino:
                return "Error: no se ha indicado ningún término de búsqueda."

            # quote() escapa el término: no puede salirse de la query string.
            url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(termino)
            webbrowser.open(url)
            return f"Éxito: se ha abierto YouTube buscando '{termino}'."

        return f"Error: acción '{accion}' no soportada."

    except Exception as e:  # noqa: BLE001  (la respuesta va al modelo, no al usuario)
        logger.error("Error controlando PC: %s", e)
        return f"Error del sistema al ejecutar la acción: {e}"


# ─── El agente ─────────────────────────────────────────────────────────────


@registrar("pc")
async def _pc(trabajo: dict[str, Any]) -> dict[str, Any]:
    """Ejecuta una acción sobre el PC.

    Va por `to_thread` porque `controlar` bloquea —lanza procesos, duerme un
    segundo antes de teclear, mueve el ratón medio segundo— y el bucle de
    eventos tiene que seguir atendiendo la API mientras tanto.
    """
    peticion = trabajo.get("peticion") or {}
    accion = str(peticion.get("accion", ""))
    parametro = str(peticion.get("parametro", ""))

    texto = await asyncio.to_thread(controlar, accion, parametro)
    if texto.startswith("Error"):
        # Que falle el trabajo, y no que se complete con un texto de error
        # dentro: así se ve en la cola de la web como lo que es.
        raise RuntimeError(texto)
    return {"texto": texto}
