"""Que lo que Perseo abre salga delante, y no parpadeando en la barra de tareas.

**Por qué no sale solo.** Windows solo deja pasar al frente a quien tiene el
foco o acaba de recibir la última pulsación. El núcleo es un `pythonw` en
segundo plano: no tiene ni lo uno ni lo otro, y lo que lanza hereda la
restricción. Chrome se abría detrás de la llamada, WhatsApp se quedaba
parpadeando naranja, y `escribir_teclado` tecleaba luego en la ventana que
hubiera delante, que no era la que se acababa de abrir.

**Qué ventana.** Se apuntan las ventanas que había antes de lanzar, y la que
aparezca nueva es la buena. Si no aparece ninguna —el programa ya estaba abierto
y solo se activó, como WhatsApp o una pestaña más en Chrome— se busca la de ese
programa por el nombre de su ejecutable.

**Cómo se pasa al frente.** Primero enganchando la entrada al hilo de la ventana
que tiene el foco (`AttachThreadInput`), que es la manera limpia. Si Windows
sigue sin dejar, una pulsación de Alt sintética: libera el candado del primer
plano, y es lo que hacen los lanzadores. Nada de esto cambia ajustes del
sistema: el candado sigue puesto para todo lo demás.

Fuera de Windows no hace nada.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Iterable
from pathlib import PureWindowsPath

logger = logging.getLogger(__name__)

#: Cuánto se espera a que aparezca la ventana. Word o Steam en frío tardan;
#: más de esto y lo más probable es que no se haya abierto nada.
PLAZO = 8.0
PAUSA = 0.2


def elegir(
    antes: set[int], ahora: Iterable[tuple[int, str]], procesos: Iterable[str]
) -> int | None:
    """Qué ventana hay que traer: una nueva, del programa si se sabe cuál; si no
    hay nueva, una que ya estaba y es de ese programa.

    Es pura a propósito: la parte de Windows solo enumera y empuja, y lo que se
    puede equivocar se prueba sin Windows.
    """
    buscados = {p.lower() for p in procesos}
    lista = list(ahora)
    nuevas = [(h, exe) for h, exe in lista if h not in antes]
    for h, exe in nuevas:
        if exe.lower() in buscados:
            return h
    if nuevas and not buscados:
        return nuevas[0][0]
    for h, exe in lista:
        if exe.lower() in buscados:
            return h
    # Una ventana nueva de otro ejecutable también vale: más de una app nace de
    # un lanzador (Steam, Office) cuyo nombre no es el de la ventana final.
    return nuevas[0][0] if nuevas else None


if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _dwmapi = ctypes.WinDLL("dwmapi")

    _ENUM = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    _user32.EnumWindows.argtypes = [_ENUM, wintypes.LPARAM]
    _user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    _user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    _user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    _user32.GetWindow.argtypes = [wintypes.HWND, ctypes.c_uint]
    _user32.GetWindow.restype = wintypes.HWND
    _user32.GetForegroundWindow.restype = wintypes.HWND
    for _f in ("IsWindowVisible", "GetWindowTextLengthW", "IsIconic", "SetForegroundWindow",
               "BringWindowToTop"):
        getattr(_user32, _f).argtypes = [wintypes.HWND]
    _user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    _user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)
    ]
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _dwmapi.DwmGetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]

    _GWL_EXSTYLE = -20
    _WS_EX_TOOLWINDOW = 0x00000080
    _GW_OWNER = 4
    _DWMWA_CLOAKED = 14
    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _SW_RESTORE = 9
    _VK_MENU = 0x12
    _KEYEVENTF_KEYUP = 0x2

    def _ejecutable(hwnd: int) -> str:
        pid = wintypes.DWORD()
        _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        proceso = _kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
        if not proceso:
            return ""
        try:
            tam = wintypes.DWORD(1024)
            ruta = ctypes.create_unicode_buffer(tam.value)
            if not _kernel32.QueryFullProcessImageNameW(proceso, 0, ruta, ctypes.byref(tam)):
                return ""
            return PureWindowsPath(ruta.value).name
        finally:
            _kernel32.CloseHandle(proceso)

    def _es_de_verdad(hwnd: int) -> bool:
        """Una ventana que una persona reconocería: visible, con título, sin
        dueño, no de herramientas y no «encapotada» (los marcos de las apps de
        la tienda que Windows tiene escondidos)."""
        if not _user32.IsWindowVisible(hwnd) or not _user32.GetWindowTextLengthW(hwnd):
            return False
        if _user32.GetWindow(hwnd, _GW_OWNER):
            return False
        if _user32.GetWindowLongW(hwnd, _GWL_EXSTYLE) & _WS_EX_TOOLWINDOW:
            return False
        encapotada = wintypes.DWORD()
        _dwmapi.DwmGetWindowAttribute(hwnd, _DWMWA_CLOAKED, ctypes.byref(encapotada), ctypes.sizeof(encapotada))
        return not encapotada.value

    def ventanas() -> list[tuple[int, str]]:
        """Las ventanas de primer nivel, de delante atrás, con su ejecutable."""
        salida: list[tuple[int, str]] = []

        def cada(hwnd: int, _: int) -> bool:
            if _es_de_verdad(hwnd):
                salida.append((int(hwnd), _ejecutable(hwnd)))
            return True

        _user32.EnumWindows(_ENUM(cada), 0)
        return salida

    def _empujar(hwnd: int) -> bool:
        if _user32.IsIconic(hwnd):
            _user32.ShowWindow(hwnd, _SW_RESTORE)
        delante = _user32.GetForegroundWindow()
        propio = _kernel32.GetCurrentThreadId()
        suyo = _user32.GetWindowThreadProcessId(delante, None) if delante else 0
        enganchado = bool(suyo and suyo != propio and _user32.AttachThreadInput(propio, suyo, True))
        try:
            _user32.BringWindowToTop(hwnd)
            _user32.SetForegroundWindow(hwnd)
        finally:
            if enganchado:
                _user32.AttachThreadInput(propio, suyo, False)
        if _user32.GetForegroundWindow() == hwnd:
            return True
        _user32.keybd_event(_VK_MENU, 0, 0, 0)
        _user32.keybd_event(_VK_MENU, 0, _KEYEVENTF_KEYUP, 0)
        _user32.SetForegroundWindow(hwnd)
        return _user32.GetForegroundWindow() == hwnd

else:

    def ventanas() -> list[tuple[int, str]]:
        return []

    def _empujar(hwnd: int) -> bool:
        return False


def instantanea() -> set[int]:
    """Las ventanas que hay ahora, para saber luego cuál es la nueva."""
    try:
        return {h for h, _ in ventanas()}
    except OSError as e:
        logger.debug("No se pudieron enumerar las ventanas: %s", e)
        return set()


def traer_al_frente(antes: set[int], procesos: Iterable[str] = (), plazo: float = PLAZO) -> bool:
    """Espera a la ventana de lo que se acaba de abrir y la pone delante."""
    if os.name != "nt":
        return False
    procesos = tuple(procesos)
    limite = time.monotonic() + plazo
    # Si ya había una del programa, antes de darla por buena se deja un momento
    # a que nazca la nueva: si no, un segundo Chrome traería el primero.
    gracia = time.monotonic() + 1.0
    while time.monotonic() < limite:
        try:
            lista = ventanas()
        except OSError as e:
            logger.debug("No se pudieron enumerar las ventanas: %s", e)
            return False
        hay_nueva = any(h not in antes for h, _ in lista)
        if hay_nueva or time.monotonic() >= gracia:
            hwnd = elegir(antes, lista, procesos)
            if hwnd is not None:
                ok = _empujar(hwnd)
                logger.info("Al frente %s: %s", "sí" if ok else "no", hwnd)
                return ok
        time.sleep(PAUSA)
    return False


def al_frente_en_segundo_plano(antes: set[int], procesos: Iterable[str] = ()) -> None:
    """Lo mismo sin hacer esperar a quien abrió: la respuesta sale ya y la
    ventana se trae en cuanto aparezca."""
    threading.Thread(
        target=traer_al_frente, args=(antes, tuple(procesos)), name="al-frente", daemon=True
    ).start()
