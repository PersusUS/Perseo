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
     lo que convierte "abrir notepad & formatear" en dos comandos en vez de uno.
  2. **Nunca se interpola texto del modelo dentro de una cadena de comando.**
     Siempre listas de argumentos, que el sistema operativo no vuelve a parsear.
  3. **Solo objetivos de una lista blanca explícita.** Si no está en la lista, no
     se ejecuta: no hay ruta de escape "por si acaso".

La política por niveles que gobierna al resto del sistema todavía no llega
aquí: hoy este agente se comporta igual que se comportaba la herramienta que
lo precedió, sin pedir confirmación.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import urllib.parse
import webbrowser
from typing import Any

from . import aplicaciones
from .agentes import registrar

logger = logging.getLogger(__name__)

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

#: Segundos que se le dan a una aplicación recién abierta para arrancar y tomar
#: el foco antes del primer teclado. En la llamada del 2026-08-24 el ctrl+l de
#: la búsqueda de una aplicación recién abierta salió antes de que la ventana
#: estuviera lista y el atajo se perdió.
ESPERA_TRAS_ABRIR_APP = 3.0

#: Cuándo se abrió la última aplicación. `None` es "hace tanto que no cuenta".
_instante_ultimo_abrir: float | None = None


def _apuntar_apertura() -> None:
    global _instante_ultimo_abrir
    _instante_ultimo_abrir = time.monotonic()


def _esperar_tras_abrir_app(minimo: float = 0.0) -> None:
    """Duerme lo que falte para que la app recién abierta tenga el foco.

    El plazo se cuenta desde que `abrir_app` terminó y se paga UNA sola vez:
    el primer teclado tras abrir cubre el resto del plazo, y los pasos
    siguientes no repiten la pausa.
    """
    global _instante_ultimo_abrir
    if _instante_ultimo_abrir is not None:
        restante = ESPERA_TRAS_ABRIR_APP - (time.monotonic() - _instante_ultimo_abrir)
        minimo = max(minimo, restante)
        _instante_ultimo_abrir = None
    if minimo > 0:
        time.sleep(minimo)


# ─── Utilidades internas ───────────────────────────────────────────────────


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


# ─── El ratón ──────────────────────────────────────────────────────────────
#
# Un clic sin coordenadas cae donde esté el cursor, que es donde lo dejó la
# persona. En una llamada del 2026-08-17 el modelo dijo "hago clic en el primer
# resultado" y clicó en otra cosa: no ve la pantalla, así que no sabía dónde
# estaba ese resultado ni que el ratón no se había movido. Lo que sigue existe
# para que eso devuelva un error en vez de un clic en cualquier parte.

#: Dónde dejó Perseo el ratón la última vez. `None` es "no lo he tocado".
_ultimo_destino: tuple[int, int] | None = None


def _apuntar_movimiento(destino: tuple[int, int] | None) -> None:
    global _ultimo_destino
    _ultimo_destino = destino


def _coordenadas(crudo: str) -> tuple[int, int] | str:
    """Convierte 'x,y' en un par de enteros, o devuelve el error a enseñar."""
    partes = crudo.split(",")
    if len(partes) != 2:
        return "Error: formato de coordenadas inválido. Debe ser 'x,y'."
    try:
        return int(partes[0].strip()), int(partes[1].strip())
    except ValueError:
        return "Error: las coordenadas deben ser números enteros."


def _partir_clic(parametro: str) -> tuple[str, tuple[int, int] | str | None]:
    """Separa el tipo de clic de las coordenadas, que pueden no venir.

    Se aceptan las tres formas que usa el modelo sin ponerse de acuerdo consigo
    mismo: `''`, `'izquierdo'`, `'300,450'` y `'derecho 300,450'`.
    """
    texto = parametro.strip().lower()
    if not texto:
        return "", None

    piezas = texto.replace(";", " ").split()
    tipo = ""
    coordenadas: tuple[int, int] | str | None = None
    for pieza in piezas:
        if "," in pieza:
            coordenadas = _coordenadas(pieza)
        else:
            # Si vienen dos palabras que no son coordenadas, la segunda deja el
            # tipo en algo que la lista blanca rechazará. Es lo que se quiere.
            tipo = pieza if not tipo else f"{tipo} {pieza}"
    return tipo, coordenadas


def _dentro_de_la_pantalla(pyautogui: Any, x: int, y: int) -> tuple[int, int]:
    ancho, alto = pyautogui.size()
    return max(0, min(x, ancho - 1)), max(0, min(y, alto - 1))


def _el_raton_sigue_donde_lo_dejamos(pyautogui: Any, margen: int = 3) -> bool:
    """Si el cursor está donde lo puso Perseo, con margen de unos píxeles.

    El margen no es por capricho: el sistema redondea, y un ratón con
    aceleración no siempre para en el píxel exacto que se le pidió.
    """
    if _ultimo_destino is None:
        return False
    try:
        x, y = pyautogui.position()
    except Exception:  # noqa: BLE001  (sin cursor legible, no se clica a ciegas)
        return False
    return abs(x - _ultimo_destino[0]) <= margen and abs(y - _ultimo_destino[1]) <= margen


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

            if aplicaciones.es_url(objetivo):
                return aplicaciones.abrir_url(objetivo)

            clave = objetivo.lower()
            if clave not in aplicaciones.APLICACIONES_PERMITIDAS:
                permitidas = ", ".join(sorted(aplicaciones.APLICACIONES_PERMITIDAS))
                return (
                    f"Error: '{objetivo}' no está en la lista de aplicaciones "
                    f"permitidas. Disponibles: {permitidas}."
                )

            tipo, destino = aplicaciones.APLICACIONES_PERMITIDAS[clave]
            try:
                aplicaciones.lanzar(tipo, destino)
            except FileNotFoundError:
                # Está permitida pero no instalada, o instalada donde no se
                # encuentra. Que lo diga así y no "error del sistema": es lo
                # único que el usuario puede arreglar.
                return f"Error: '{clave}' está permitida pero no se encuentra instalada."
            _apuntar_apertura()
            return f"Éxito: se ha abierto '{clave}'."

        if accion == "escribir_teclado":
            if len(parametro) > MAX_LONGITUD_TEXTO:
                return f"Error: el texto supera el límite de {MAX_LONGITUD_TEXTO} caracteres."

            texto = _texto_imprimible(parametro)
            if not texto:
                return "Error: no queda texto imprimible que teclear."

            try:
                pyautogui = _pyautogui()
            except ImportError:
                return "Error: la librería 'pyautogui' no está instalada."

            # Cortesía: la ventana destino puede estar abriéndose; y si acaba
            # de abrirse una app, se le da su plazo completo antes de teclear.
            _esperar_tras_abrir_app(minimo=1.0)
            pyautogui.write(texto, interval=0.01)
            return f"Éxito: se ha tecleado el texto '{texto}' en la ventana actual."

        if accion == "atajo_teclado":
            # Separadores válidos: coma y signo más ("ctrl, l" o "ctrl+l").
            # El modelo escribe el segundo con naturalidad, y rechazarlo por
            # ortografía es perder el atajo entero.
            teclas = [t.strip().lower() for t in re.split(r"[,+]", parametro) if t.strip()]

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

            # Si la app acaba de abrirse, que tenga el foco antes del atajo:
            # si no, el atajo cae en la ventana que hubiera antes.
            _esperar_tras_abrir_app()
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
            destino = _coordenadas(parametro)
            if isinstance(destino, str):
                return destino

            try:
                pyautogui = _pyautogui()
            except ImportError:
                return "Error: la librería 'pyautogui' no está instalada."

            x, y = _dentro_de_la_pantalla(pyautogui, *destino)
            pyautogui.moveTo(x, y, duration=0.5)
            _apuntar_movimiento((x, y))
            return f"Éxito: ratón movido a ({x}, {y})."

        if accion == "click_raton":
            tipo, destino = _partir_clic(parametro)
            if tipo not in ("", "izquierdo", "derecho", "doble"):
                return "Error: tipo de clic no reconocido. Use 'izquierdo', 'derecho' o 'doble'."
            if isinstance(destino, str):
                return destino

            try:
                pyautogui = _pyautogui()
            except ImportError:
                return "Error: la librería 'pyautogui' no está instalada."

            if destino is not None:
                x, y = _dentro_de_la_pantalla(pyautogui, *destino)
                pyautogui.moveTo(x, y, duration=0.3)
                _apuntar_movimiento((x, y))
            elif not _el_raton_sigue_donde_lo_dejamos(pyautogui):
                return (
                    "Error: no se puede clicar a ciegas. El ratón está donde lo dejó el "
                    "usuario, no donde lo puso Perseo. Indica dónde clicar con "
                    "'click_raton' y coordenadas 'x,y', o mueve el ratón antes con "
                    "'mover_raton'."
                )

            if tipo == "derecho":
                pyautogui.rightClick()
            elif tipo == "doble":
                pyautogui.doubleClick()
            else:
                pyautogui.click()

            donde = f" en ({destino[0]}, {destino[1]})" if destino else ""
            return f"Éxito: clic '{tipo or 'izquierdo'}' ejecutado{donde}."

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
