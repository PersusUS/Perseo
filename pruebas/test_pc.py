"""El agente `pc`: ninguna inyección llega a tocar el sistema.

Ninguna de estas pruebas abre una ventana ni teclea nada: todas comprueban casos
que se rechazan **antes** de llamar a nadie.
"""

from __future__ import annotations

import sys

import pytest

from perseo_core import pc


@pytest.mark.parametrize(
    ("accion", "parametro", "motivo"),
    [
        ("abrir_app", "notepad & calc", "encadenado con &"),
        ("abrir_app", "notepad && shutdown /s /t 0", "encadenado con &&"),
        ("abrir_app", "a | del /q C:\\*", "tuberia"),
        ("abrir_app", "cmd", "shell fuera de la lista"),
        ("abrir_app", "powershell", "shell fuera de la lista"),
        ("abrir_app", "file:///C:/Windows", "esquema no permitido"),
        ("abrir_app", "javascript:alert(1)", "esquema no permitido"),
        ("atajo_teclado", "ctrl,alt,delete,f4,esc,tab", "demasiadas teclas"),
        ("atajo_teclado", "ctrl,shutdown", "tecla inventada"),
        ("atajo_teclado", "", "sin teclas"),
        ("mover_raton", "abc,def", "coordenadas no numericas"),
        ("mover_raton", "10", "coordenadas incompletas"),
        ("volumen", "; rm -rf /", "valor no reconocido"),
        ("click_raton", "triple; shutdown", "tipo de clic inventado"),
        ("apagar_equipo", "ya", "accion inexistente"),
    ],
)
def test_las_inyecciones_se_bloquean(accion: str, parametro: str, motivo: str) -> None:
    assert pc.controlar(accion, parametro).startswith("Error:"), motivo


def test_una_url_http_no_es_una_inyeccion() -> None:
    """El rechazo tiene que ser por el esquema, no por ser una URL."""
    assert pc._abrir_url.__doc__ is not None  # la función existe y está documentada
    partes = pc._es_url("https://example.com")
    assert partes is True


def test_el_texto_a_teclear_pierde_los_controles() -> None:
    """Un '\\n' equivale a pulsar Enter en la ventana que tenga el foco."""
    assert pc._texto_imprimible("formatear\nsi") == "formatearsi"
    assert pc._texto_imprimible("dato\ty\rotro") == "datoyotro"


def test_un_texto_larguisimo_se_rechaza() -> None:
    largo = "x" * (pc.MAX_LONGITUD_TEXTO + 1)
    assert pc.controlar("escribir_teclado", largo).startswith("Error:")


def test_un_texto_solo_de_controles_se_rechaza() -> None:
    assert pc.controlar("escribir_teclado", "\n\t\r").startswith("Error:")


def test_la_lista_blanca_no_lleva_interpretes() -> None:
    """Poder abrir un shell haría inútil todo lo demás del módulo."""
    prohibidos = {"cmd", "powershell", "wt", "terminal", "regedit", "bash"}
    assert not (prohibidos & set(pc.APLICACIONES_PERMITIDAS))


def test_las_teclas_permitidas_no_llevan_teclas_de_sistema() -> None:
    assert "printscreen" not in pc._TECLAS_PERMITIDAS
    assert "volumemute" not in pc._TECLAS_PERMITIDAS


def test_los_esquemas_de_url_permitidos_son_dos() -> None:
    assert pc.ESQUEMAS_URL_PERMITIDOS == {"http", "https"}


def test_una_url_con_esquema_raro_se_rechaza() -> None:
    assert pc._abrir_url("ftp://archivos.example/x").startswith("Error:")


def test_una_url_sin_dominio_se_rechaza() -> None:
    assert pc._abrir_url("http:///sin-dominio").startswith("Error:")


def test_buscar_en_youtube_sin_termino_se_rechaza() -> None:
    assert pc.controlar("buscar_youtube", "   ").startswith("Error:")


def test_la_accion_no_distingue_mayusculas_ni_espacios() -> None:
    assert pc.controlar("  APAGAR_EQUIPO ", "").startswith("Error:")


# --------------------------------------------------------------------------- #
# El ratón: nunca a ciegas
# --------------------------------------------------------------------------- #


class _RatonFalso:
    """Lo justo de pyautogui para comprobar el ratón sin mover nada."""

    def __init__(self, posicion: tuple[int, int] = (0, 0)) -> None:
        self.posicion = posicion
        self.clics: list[str] = []
        self.movimientos: list[tuple[int, int]] = []

    def size(self) -> tuple[int, int]:
        return (1920, 1080)

    def position(self) -> tuple[int, int]:
        return self.posicion

    def moveTo(self, x: int, y: int, duration: float = 0) -> None:  # noqa: N802
        self.posicion = (x, y)
        self.movimientos.append((x, y))

    def click(self) -> None:
        self.clics.append("izquierdo")

    def rightClick(self) -> None:  # noqa: N802
        self.clics.append("derecho")

    def doubleClick(self) -> None:  # noqa: N802
        self.clics.append("doble")


@pytest.fixture()
def raton(monkeypatch: pytest.MonkeyPatch) -> _RatonFalso:
    falso = _RatonFalso()
    monkeypatch.setattr(pc, "_pyautogui", lambda: falso)
    monkeypatch.setattr(pc, "_ultimo_destino", None)
    return falso


def test_clicar_sin_haber_movido_el_raton_se_rechaza(raton: _RatonFalso) -> None:
    """El fallo de la llamada del 2026-08-17: el modelo dijo «clico el primer
    resultado» y clicó donde estaba el ratón de la persona."""
    respuesta = pc.controlar("click_raton", "izquierdo")
    assert respuesta.startswith("Error:")
    assert not raton.clics


def test_clicar_con_coordenadas_mueve_primero(raton: _RatonFalso) -> None:
    respuesta = pc.controlar("click_raton", "300,450")
    assert respuesta.startswith("Éxito:")
    assert raton.movimientos == [(300, 450)]
    assert raton.clics == ["izquierdo"]


def test_el_tipo_de_clic_sigue_valiendo_con_coordenadas(raton: _RatonFalso) -> None:
    assert pc.controlar("click_raton", "derecho 300,450").startswith("Éxito:")
    assert raton.clics == ["derecho"]


def test_mover_y_luego_clicar_si_vale(raton: _RatonFalso) -> None:
    """Mover y clicar es el camino bueno: Perseo sabe dónde está el cursor."""
    assert pc.controlar("mover_raton", "800,600").startswith("Éxito:")
    assert pc.controlar("click_raton", "").startswith("Éxito:")
    assert raton.clics == ["izquierdo"]


def test_si_el_usuario_mueve_el_raton_despues_no_se_clica(raton: _RatonFalso) -> None:
    """Lo más importante: entre mover y clicar, la persona usa su ratón."""
    pc.controlar("mover_raton", "800,600")
    raton.posicion = (20, 20)
    assert pc.controlar("click_raton", "").startswith("Error:")
    assert not raton.clics


def test_las_coordenadas_no_se_salen_de_la_pantalla(raton: _RatonFalso) -> None:
    pc.controlar("click_raton", "99999,99999")
    assert raton.movimientos == [(1919, 1079)]


def test_unas_coordenadas_rotas_no_clican(raton: _RatonFalso) -> None:
    assert pc.controlar("click_raton", "abc,def").startswith("Error:")
    assert not raton.clics


# --------------------------------------------------------------------------- #
# Abrir una aplicación que no está en el PATH
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(sys.platform != "win32", reason="App Paths es del registro de Windows")
def test_el_bloc_de_notas_se_encuentra() -> None:
    """Lo que sí está en el PATH tiene que seguir encontrándose."""
    assert pc.resolver_ejecutable("notepad.exe")


@pytest.mark.skipif(sys.platform != "win32", reason="App Paths es del registro de Windows")
def test_chrome_se_encuentra_aunque_no_este_en_el_path() -> None:
    """El fallo de la llamada del 2026-08-17: `Popen(['chrome.exe'])` falla
    porque los navegadores no están en el PATH. Windows los resuelve por el
    registro, y ahora esto también.

    Si esta máquina no tiene Chrome, la prueba no tiene nada que decir.
    """
    if not pc._en_app_paths("chrome.exe"):
        pytest.skip("Chrome no está instalado en esta máquina")
    assert pc.resolver_ejecutable("chrome.exe")


def test_un_ejecutable_inventado_no_se_encuentra() -> None:
    assert pc.resolver_ejecutable("no-existe-de-verdad.exe") is None


def test_una_app_permitida_pero_no_instalada_lo_dice(monkeypatch: pytest.MonkeyPatch) -> None:
    """«No está instalada» es lo único que el usuario puede arreglar; «error del
    sistema al ejecutar la acción» no le dice nada."""
    monkeypatch.setattr(pc, "resolver_ejecutable", lambda _: None)
    respuesta = pc.controlar("abrir_app", "chrome")
    assert respuesta.startswith("Error:") and "no se encuentra instalada" in respuesta
