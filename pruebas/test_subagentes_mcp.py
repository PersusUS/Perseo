"""El servidor MCP de subagentes: con qué motor trabaja y cuándo miente.

Lo que se prueba aquí es lo que costó una tarde de encargos fantasma el
2026-08-24: `opencode run` sin `--auto` se deniega a sí mismo el permiso de
escribir, sale con código 0, y el encargo quedaba apuntado como «hecho» con el
disco intacto. La misma trampa tiene el proveedor del modelo gratuito cuando su
endpoint se cae a media petición.

El circuito entero —lanzar de verdad un CLI y esperar— es de los verificadores;
esto mira la decisión, que es donde estaba el fallo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "commands"))

import subagentes_mcp  # noqa: E402


# --------------------------------------------------------------------------- #
# Qué motor se elige
# --------------------------------------------------------------------------- #


@pytest.fixture()
def todos_instalados(monkeypatch: pytest.MonkeyPatch):
    """Como si `claude` y `opencode` estuvieran los dos en el PATH."""
    monkeypatch.setattr(subagentes_mcp.shutil, "which", lambda nombre: f"/bin/{nombre}")
    monkeypatch.delenv("PERSEO_SUBAGENTE_MOTOR", raising=False)


def test_por_defecto_manda_claude(todos_instalados) -> None:
    """El gratuito se cae a ratos y no siempre lo dice: no puede ser el de serie."""
    assert subagentes_mcp._motor() == "claude"


def test_pedir_opencode_se_respeta(todos_instalados, monkeypatch) -> None:
    monkeypatch.setenv("PERSEO_SUBAGENTE_MOTOR", "opencode")
    assert subagentes_mcp._motor() == "opencode"


def test_si_el_pedido_no_esta_queda_claude(monkeypatch) -> None:
    monkeypatch.setenv("PERSEO_SUBAGENTE_MOTOR", "opencode")
    monkeypatch.setattr(
        subagentes_mcp.shutil, "which", lambda nombre: "/bin/claude" if nombre == "claude" else None
    )
    assert subagentes_mcp._motor() == "claude"


def test_sin_ningun_motor_el_error_dice_que_instalar(monkeypatch) -> None:
    monkeypatch.delenv("PERSEO_SUBAGENTE_MOTOR", raising=False)
    monkeypatch.setattr(subagentes_mcp.shutil, "which", lambda nombre: None)
    with pytest.raises(RuntimeError, match="opencode"):
        subagentes_mcp._motor()


# --------------------------------------------------------------------------- #
# La línea de órdenes
# --------------------------------------------------------------------------- #


def test_opencode_va_con_auto(monkeypatch) -> None:
    monkeypatch.setattr(subagentes_mcp.shutil, "which", lambda nombre: f"/bin/{nombre}")
    assert "--auto" in subagentes_mcp._comando("opencode", "haz algo")


def test_claude_trabaja_sin_pedir_permiso(monkeypatch) -> None:
    """Con `acceptEdits` no puede correr ni un comando: se queda esperando."""
    monkeypatch.setattr(subagentes_mcp.shutil, "which", lambda nombre: f"/bin/{nombre}")
    comando = subagentes_mcp._comando("claude", "haz algo")
    assert "bypassPermissions" in comando and "haz algo" in comando


def test_ni_asi_publica_ni_borra(monkeypatch) -> None:
    """El cerco no depende del modo de permisos: la lista de denegados manda."""
    monkeypatch.setattr(subagentes_mcp.shutil, "which", lambda nombre: f"/bin/{nombre}")
    comando = subagentes_mcp._comando("claude", "haz algo")
    assert any("git push" in argumento for argumento in comando)
    assert any(argumento.startswith("Bash(rm ") for argumento in comando)


def test_esperar_un_si_que_nadie_dara_es_un_fracaso() -> None:
    assert subagentes_mcp._fracaso_encubierto("Comando necesita tu aprobación pa correr.")


# --------------------------------------------------------------------------- #
# El fracaso que sale con código 0
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "salida",
    [
        "! permission requested: external_directory; auto-rejecting",
        "Error: The user rejected permission to use this specific tool call.",
        "Error from provider (Console): Upstream request failed: Endpoint is unavailable.",
    ],
)
def test_una_salida_que_delata_el_fracaso_se_caza(salida: str) -> None:
    assert subagentes_mcp._fracaso_encubierto(salida)


def test_un_trabajo_de_verdad_no_se_confunde_con_un_fracaso() -> None:
    bueno = "Fichero `ok.txt` creado en el escritorio con el texto hola."
    assert subagentes_mcp._fracaso_encubierto(bueno) == ""
