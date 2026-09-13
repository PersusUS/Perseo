"""La documentación, comprobada contra el código que dice describir.

Existe porque la prosa y el código no se tocan: nada falla cuando divergen, así
que divergen. El 2026-09-12 se encontraron dos casos a la vez —`docs/API.md`
documentaba un campo `instruccion` que la API no ha leído nunca, y el recuento
de pruebas estaba escrito a mano en cuatro sitios con cuatro cifras distintas—
y los dos llevaban meses ahí.

La regla que sigue este fichero: **si un dato se puede medir, no se escribe a
mano; y si se escribe a mano, hay una prueba que lo compara con la medida.**
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "commands"))

import comprobar  # noqa: E402

API_MD = RAIZ / "docs" / "API.md"
API_PY = RAIZ / "perseo_core" / "caras" / "api.py"

#: Rutas que existen y no se documentan a propósito: son los ficheros que sirve
#: la web del móvil. Documentar un icono no le sirve a nadie.
NO_SE_DOCUMENTAN = frozenset(
    {
        "GET /",
        "GET /manifest.webmanifest",
        "GET /hokusai-bg.png",
        "GET /perseo-avatar.jpg",
        "GET /icono-180.png",
        "GET /icono-512.png",
        "GET /apple-touch-icon.png",
        "GET /apple-touch-icon-precomposed.png",
        # El canje de token por cookie: lo explica la sección de autenticación en
        # prosa, con su `curl`, en vez de en la tabla de rutas.
        "POST /sesion",
    }
)

#: Rutas que la documentación nombra **para decir que no existen**. La sección
#: «Lo que la API no hace» es una promesa, no una tabla: por HTTP se encola un
#: trabajo para un agente y no se consigue una consola. Si algún día alguien
#: añade de verdad un `POST /ejecutar`, esta lista deja de ser una excepción y
#: pasa a ser la prueba que lo impide.
PROMETIDAS_INEXISTENTES = frozenset({"POST /ejecutar", "POST /shell"})

_RUTA_EN_CODIGO = re.compile(r"web\.(get|post|delete|put|patch)\(\s*\"([^\"]+)\"")
_RUTA_EN_DOCS = re.compile(r"`(GET|POST|DELETE|PUT|PATCH) (/[^`]*)`")
_ALTERNATIVAS = re.compile(r"\{[^}]*:([^}]+)\}")


def _expandir(ruta: str) -> list[str]:
    """`/t/{id}/{d:aprobar|rechazar}` -> ["/t/{id}/aprobar", "/t/{id}/rechazar"].

    aiohttp admite una expresión regular dentro de la llave; la documentación
    escribe las dos rutas por separado, que es como las usa quien lee.
    """
    encontrado = _ALTERNATIVAS.search(ruta)
    if not encontrado:
        return [ruta]
    return [
        ruta[: encontrado.start()] + opcion + ruta[encontrado.end() :]
        for opcion in encontrado.group(1).split("|")
    ]


def rutas_reales() -> set[str]:
    """Las que el núcleo registra de verdad, leídas de la tabla de `api.py`."""
    fuente = API_PY.read_text(encoding="utf-8")
    reales: set[str] = set()
    for metodo, ruta in _RUTA_EN_CODIGO.findall(fuente):
        for expandida in _expandir(ruta):
            reales.add(f"{metodo.upper()} {expandida}")
    return reales


def rutas_documentadas() -> set[str]:
    texto = API_MD.read_text(encoding="utf-8")
    return {f"{m} {r}" for m, r in _RUTA_EN_DOCS.findall(texto)}


def test_la_tabla_de_rutas_del_codigo_se_lee() -> None:
    """Red de seguridad: si `api.py` cambia de forma, esto avisa en vez de callar.

    Una prueba que compara dos conjuntos vacíos pasa siempre, y sería peor que
    no tenerla.
    """
    assert len(rutas_reales()) > 30
    assert len(rutas_documentadas()) > 25


def test_toda_ruta_documentada_existe() -> None:
    """El fallo que motivó esto: documentación describiendo una API inexistente."""
    inventadas = sorted(rutas_documentadas() - rutas_reales() - PROMETIDAS_INEXISTENTES)
    assert not inventadas, "docs/API.md documenta rutas que no existen: " + ", ".join(inventadas)


def test_lo_que_la_api_promete_no_hacer_sigue_sin_existir() -> None:
    """«No hay un `POST /ejecutar` ni un `POST /shell`» es una promesa comprobable."""
    rotas = sorted(PROMETIDAS_INEXISTENTES & rutas_reales())
    assert not rotas, "docs/API.md promete que no existen, y existen: " + ", ".join(rotas)


def test_toda_ruta_real_esta_documentada() -> None:
    """El otro lado: una ruta nueva que nadie cuenta es una ruta que nadie usa."""
    calladas = sorted(rutas_reales() - rutas_documentadas() - NO_SE_DOCUMENTAN)
    assert not calladas, "existen y no salen en docs/API.md: " + ", ".join(calladas)


def test_lo_que_se_sirve_sin_token_existe() -> None:
    """Una ruta pública que ya no existe es una excepción de seguridad huérfana."""
    from perseo_core.caras import api

    caminos = {ruta.split(" ", 1)[1] for ruta in rutas_reales()}
    # `/favicon.ico` lo pide el navegador solo y se contesta con un 404 limpio;
    # está en la lista para que ese 404 no pida token.
    sobran = sorted(set(api.RUTAS_PUBLICAS) - caminos - {"/favicon.ico"})
    assert not sobran, "RUTAS_PUBLICAS nombra rutas que no existen: " + ", ".join(sobran)


# ==========================================================================
# Los números que la documentación cita
# ==========================================================================
def test_los_numeros_escritos_a_mano_son_los_de_verdad() -> None:
    """Cuatro sitios con el recuento a mano son cuatro sitios que envejecen.

    Dónde está escrita cada cifra vive en `comprobar.CITAS`, que es lo mismo que
    usa el comando para reescribirlas. Si esto se pone rojo, el arreglo no es
    tocar la prueba:

        python commands/perseo.py cuentas --arreglar
    """
    errores = comprobar.revisar_citas()
    assert not errores, "\n".join(errores)
