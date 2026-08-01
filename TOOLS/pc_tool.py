"""Control del PC bajo Windows para Perseo.

REGLA DE SEGURIDAD DE ESTE MÓDULO
---------------------------------
Todo lo que llega aquí es texto generado por un modelo de lenguaje que, además,
está observando la pantalla y la cámara del usuario en tiempo real. Cualquier
texto visible en pantalla —una web, un correo abierto, un PDF— puede acabar
influyendo en estos argumentos. Por lo tanto, en este archivo:

  1. NUNCA se invoca un shell. Ni `os.system`, ni `shell=True`. El shell es lo
     que convierte "abrir spotify & formatear" en dos comandos en vez de uno.
  2. NUNCA se interpola texto del modelo dentro de una cadena de comando.
     Siempre listas de argumentos, que el sistema operativo no vuelve a parsear.
  3. Solo se permiten objetivos de una lista blanca explícita. Si no está en la
     lista, no se ejecuta: no hay ruta de escape "por si acaso".

Ver bitacora/02_HALLAZGOS.md H-16.
"""

import logging
import subprocess
import urllib.parse
import webbrowser

logger = logging.getLogger(__name__)

# ─── Lista blanca de aplicaciones ──────────────────────────────────────────
#
# ("exe", ...) se lanza como proceso con una lista de argumentos.
# ("uri", ...) se abre con el manejador de protocolo registrado en Windows.
#
# Deliberadamente EXCLUIDOS, y no por olvido: cmd, powershell, terminal, wt,
# regedit, y cualquier intérprete. Poder abrir un shell haría inútil el resto
# de este archivo, porque el modelo podría teclear dentro de él con
# `escribir_teclado`.
APLICACIONES_PERMITIDAS: dict[str, tuple[str, str]] = {
    "spotify":        ("uri", "spotify:"),
    "notepad":        ("exe", "notepad.exe"),
    "bloc de notas":  ("exe", "notepad.exe"),
    "calculadora":    ("exe", "calc.exe"),
    "calc":           ("exe", "calc.exe"),
    "paint":          ("exe", "mspaint.exe"),
    "explorador":     ("exe", "explorer.exe"),
    "chrome":         ("exe", "chrome.exe"),
    "firefox":        ("exe", "firefox.exe"),
    "edge":           ("uri", "microsoft-edge:"),
    "obsidian":       ("uri", "obsidian:"),
    "ajustes":        ("uri", "ms-settings:"),
    "correo":         ("uri", "mailto:"),
}

ESQUEMAS_URL_PERMITIDOS = frozenset({"http", "https"})

# Teclas admitidas en `atajo_teclado`. Restringido a propósito: modificadores,
# letras, dígitos, navegación y funciones. Sin teclas de sistema.
_MODIFICADORES = frozenset({"ctrl", "alt", "shift", "win"})
_TECLAS_SIMPLES = frozenset(
    list("abcdefghijklmnopqrstuvwxyz0123456789")
    + ["f%d" % n for n in range(1, 13)]
    + ["enter", "tab", "esc", "escape", "space", "backspace", "delete",
       "home", "end", "pageup", "pagedown", "up", "down", "left", "right"]
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
        return (f"Error: esquema de URL no permitido ('{partes.scheme}'). "
                f"Solo se admiten: {', '.join(sorted(ESQUEMAS_URL_PERMITIDOS))}.")
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


# ─── Herramienta expuesta al modelo ────────────────────────────────────────

def controlar_pc(accion: str, parametro: str = "") -> str:
    """Controla funcionalidades del PC bajo Windows.

    Args:
        accion: 'abrir_app', 'escribir_teclado', 'atajo_teclado', 'volumen',
                'mover_raton', 'click_raton' o 'buscar_youtube'.
        parametro: el argumento de la acción. Siempre validado antes de usarse.

    Returns:
        Cadena que empieza por 'Éxito:' o 'Error:', que se devuelve al modelo.
    """
    try:
        accion = (accion or "").strip().lower()
        parametro = parametro or ""

        # ── Abrir aplicación o URL ──────────────────────────────────────
        if accion == "abrir_app":
            objetivo = parametro.strip()

            if _es_url(objetivo):
                return _abrir_url(objetivo)

            clave = objetivo.lower()
            if clave not in APLICACIONES_PERMITIDAS:
                permitidas = ", ".join(sorted(APLICACIONES_PERMITIDAS))
                return (f"Error: '{objetivo}' no está en la lista de aplicaciones "
                        f"permitidas. Disponibles: {permitidas}.")

            tipo, destino = APLICACIONES_PERMITIDAS[clave]
            _lanzar(tipo, destino)
            return f"Éxito: se ha abierto '{clave}'."

        # ── Teclear texto ───────────────────────────────────────────────
        elif accion == "escribir_teclado":
            if len(parametro) > MAX_LONGITUD_TEXTO:
                return (f"Error: el texto supera el límite de "
                        f"{MAX_LONGITUD_TEXTO} caracteres.")

            texto = _texto_imprimible(parametro)
            if not texto:
                return "Error: no queda texto imprimible que teclear."

            try:
                import time

                import pyautogui
            except ImportError:
                return "Error: la librería 'pyautogui' no está instalada."

            time.sleep(1)  # cortesía: la ventana destino puede estar abriéndose
            pyautogui.write(texto, interval=0.01)
            return f"Éxito: se ha tecleado el texto '{texto}' en la ventana actual."

        # ── Atajo de teclado ────────────────────────────────────────────
        elif accion == "atajo_teclado":
            teclas = [t.strip().lower() for t in parametro.split(",") if t.strip()]

            if not teclas:
                return "Error: no se ha indicado ninguna tecla."
            if len(teclas) > MAX_TECLAS_ATAJO:
                return f"Error: máximo {MAX_TECLAS_ATAJO} teclas por atajo."

            no_validas = [t for t in teclas if t not in _TECLAS_PERMITIDAS]
            if no_validas:
                return f"Error: teclas no permitidas: {', '.join(no_validas)}."

            try:
                import pyautogui
            except ImportError:
                return "Error: la librería 'pyautogui' no está instalada."

            pyautogui.hotkey(*teclas)
            return f"Éxito: se ha ejecutado el atajo '{'+'.join(teclas)}'."

        # ── Volumen ─────────────────────────────────────────────────────
        elif accion == "volumen":
            try:
                import pyautogui
            except ImportError:
                return "Error: la librería 'pyautogui' no está instalada."

            modo = parametro.strip().lower()
            if modo == "subir":
                pyautogui.press("volumeup", presses=5)
            elif modo == "bajar":
                pyautogui.press("volumedown", presses=5)
            elif modo in ("mutear", "silenciar"):
                pyautogui.press("volumemute")
            else:
                return ("Error: valor de volumen no reconocido. "
                        "Use 'subir', 'bajar' o 'mutear'.")
            return f"Éxito: acción de volumen '{modo}' enviada al sistema."

        # ── Ratón ───────────────────────────────────────────────────────
        elif accion == "mover_raton":
            try:
                import pyautogui
            except ImportError:
                return "Error: la librería 'pyautogui' no está instalada."

            coords = parametro.split(",")
            if len(coords) != 2:
                return "Error: formato de coordenadas inválido. Debe ser 'x,y'."
            try:
                x, y = int(coords[0].strip()), int(coords[1].strip())
            except ValueError:
                return "Error: las coordenadas deben ser números enteros."

            ancho, alto = pyautogui.size()
            x = max(0, min(x, ancho - 1))
            y = max(0, min(y, alto - 1))
            pyautogui.moveTo(x, y, duration=0.5)
            return f"Éxito: ratón movido a ({x}, {y})."

        elif accion == "click_raton":
            try:
                import pyautogui
            except ImportError:
                return "Error: la librería 'pyautogui' no está instalada."

            tipo = parametro.strip().lower()
            if tipo == "derecho":
                pyautogui.rightClick()
            elif tipo == "doble":
                pyautogui.doubleClick()
            elif tipo in ("", "izquierdo"):
                pyautogui.click()
            else:
                return ("Error: tipo de clic no reconocido. "
                        "Use 'izquierdo', 'derecho' o 'doble'.")
            return f"Éxito: clic '{tipo or 'izquierdo'}' ejecutado."

        # ── Búsqueda en YouTube ─────────────────────────────────────────
        elif accion == "buscar_youtube":
            termino = parametro.strip()
            if not termino:
                return "Error: no se ha indicado ningún término de búsqueda."

            # quote() escapa el término: no puede escaparse de la query string.
            url = ("https://www.youtube.com/results?search_query="
                   + urllib.parse.quote(termino))
            webbrowser.open(url)
            return f"Éxito: se ha abierto YouTube buscando '{termino}'."

        else:
            return f"Error: acción '{accion}' no soportada."

    except Exception as e:
        logger.error("Error controlando PC: %s", e)
        return f"Error del sistema al ejecutar la acción: {e}"


# ─── Pruebas de regresión de seguridad ─────────────────────────────────────
# Cada caso es un intento de inyección que la versión anterior de este módulo
# habría ejecutado. Ninguno debe llegar a tocar el sistema.

if __name__ == "__main__":
    casos = [
        ("abrir_app",       "notepad & calc",              "encadenado con &"),
        ("abrir_app",       "spotify && shutdown /s /t 0", "encadenado con &&"),
        ("abrir_app",       "a | del /q C:\\*",            "tubería"),
        ("abrir_app",       "cmd",                         "shell fuera de la lista"),
        ("abrir_app",       "powershell",                  "shell fuera de la lista"),
        ("abrir_app",       "file:///C:/Windows",          "esquema no permitido"),
        ("abrir_app",       "javascript:alert(1)",         "esquema no permitido"),
        ("atajo_teclado",   "ctrl,alt,delete,f4,esc,tab",  "demasiadas teclas"),
        ("atajo_teclado",   "ctrl,shutdown",               "tecla inventada"),
        ("mover_raton",     "abc,def",                     "coordenadas no numéricas"),
        ("volumen",         "; rm -rf /",                  "valor no reconocido"),
        ("apagar_equipo",   "ya",                          "acción inexistente"),
    ]

    print("Pruebas de regresión de seguridad — ninguna debe ejecutarse\n")
    fallos = 0
    for accion, parametro, motivo in casos:
        resultado = controlar_pc(accion, parametro)
        bloqueado = resultado.startswith("Error:")
        estado = "BLOQUEADO" if bloqueado else "*** PASÓ ***"
        if not bloqueado:
            fallos += 1
        print(f"  [{estado}] {motivo}")
        print(f"             {accion}({parametro!r})")
        print(f"             -> {resultado}\n")

    # Saneado del texto a teclear: se comprueba la función pura, sin teclear
    # nada de verdad en la ventana que tenga el foco.
    print("  Saneado de texto (sin efectos):")
    for entrada, motivo in [("formatear\nsi", "salto de línea"),
                            ("dato\ty\rotro", "tabulador y retorno")]:
        salida = _texto_imprimible(entrada)
        limpio = not any(c in salida for c in "\n\r\t")
        if not limpio:
            fallos += 1
        print(f"    [{'BLOQUEADO' if limpio else '*** PASÓ ***'}] {motivo}: "
              f"{entrada!r} -> {salida!r}")
    total = len(casos) + 2

    print(f"\nResultado: {total - fallos}/{total} bloqueados.")
    if fallos:
        raise SystemExit(1)
