"""La voz ejecuta por el núcleo lo que Rust no conoce (ADR 0008).

Antes cada herramienta de voz nueva era una línea en la tabla `DIRECTAS` de
`nucleo.rs`, que vive pegado a su techo de 900 líneas; ahora Rust manda lo que
no es suyo a `POST /herramientas/{nombre}`, que corre el mismo despacho que el
chat escrito. Lo que se ata aquí es que no quede ninguna herramienta de voz sin
nadie que la atienda, y que la ruta respete quién habla.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer

from perseo_core.agentes import chat_herramientas
from perseo_core.caras import api
from perseo_core.infra import almacen
from perseo_core.infra.bus import Bus
from perseo_core.servicios import catalogo

RAIZ = Path(__file__).resolve().parent.parent


def _atendidas() -> tuple[set[str], set[str], set[str]]:
    rust = (RAIZ / "RealTime/src-tauri/src/nucleo.rs").read_text(encoding="utf-8")
    inicio = rust.index("fn traducir(")
    fin = rust.index("{DESCONOCIDA}", inicio)
    de_rust = set(re.findall(r'"([a-z_]+)"\s*(?:\||=>)', rust[inicio:fin]))
    de_rust |= set(re.findall(r'tool_name == "([a-z_]+)"', rust))
    ts = (RAIZ / "RealTime/src/lib/llamada/gemini-live.ts").read_text(encoding="utf-8")
    de_la_app = set(re.findall(r"name === '([a-z_]+)'", ts))
    python = (RAIZ / "perseo_core/agentes/chat_herramientas.py").read_text(encoding="utf-8")
    del_nucleo = {n for n in catalogo.nombres("voz") if f'"{n}"' in python}
    return de_la_app, de_rust, del_nucleo


def test_ninguna_herramienta_de_voz_se_queda_sin_quien_la_atienda() -> None:
    de_la_app, de_rust, del_nucleo = _atendidas()
    huerfanas = [n for n in catalogo.nombres("voz") if n not in de_la_app | de_rust | del_nucleo]
    assert not huerfanas, f"Nadie atiende por voz: {huerfanas}"


def test_rust_manda_al_nucleo_lo_que_no_conoce() -> None:
    rust = (RAIZ / "RealTime/src-tauri/src/nucleo.rs").read_text(encoding="utf-8")
    assert "Err(e) if e.starts_with(DESCONOCIDA) => return herramienta_del_nucleo" in rust
    assert '"{}/herramientas/{nombre}"' in rust
    assert "const DIRECTAS" not in rust, "la tabla vieja volvió: las nuevas van por el núcleo"


def _cliente(cfg) -> TestClient:
    return TestClient(TestServer(api.crear_app(cfg, Bus(), router=None)))


def test_la_ruta_ejecuta_con_la_voz_de_quien_habla(cfg, db, monkeypatch) -> None:
    """Lo que encola una herramienta pedida por voz llega a la cola como de voz y de quien habló."""
    chat_herramientas.iniciar(cfg)

    async def prueba() -> None:
        async with _cliente(cfg) as cliente:
            r = await cliente.post(
                "/herramientas/crear_recordatorio",
                json={"argumentos": {"texto": "regar", "en_minutos": 30}, "quien": "Una visita"},
                headers={"Authorization": f"Bearer {cfg.token}"},
            )
            assert r.status == 200, await r.text()

    monkeypatch.setattr(chat_herramientas, "_encolar_y_esperar", _encolar_sin_esperar)
    asyncio.run(prueba())
    [trabajo] = [t for t in almacen.listar() if t["agente"] == "recordatorios"]
    assert trabajo["origen"] == "voz" and trabajo["quien"] == "Una visita"


async def _encolar_sin_esperar(agente, peticion, espera=30):
    """Como el de verdad, sin quedarse esperando a un trabajador que aquí no hay."""
    trabajo = await asyncio.to_thread(
        almacen.encolar, agente, peticion, chat_herramientas._origen.get(), chat_herramientas._quien.get()
    )
    return f"#{trabajo['id']}"


def test_una_herramienta_que_no_existe_es_404(cfg, db) -> None:
    async def prueba() -> int:
        async with _cliente(cfg) as cliente:
            r = await cliente.post(
                "/herramientas/borrar_todo", json={}, headers={"Authorization": f"Bearer {cfg.token}"}
            )
            return r.status

    assert asyncio.run(prueba()) == 404


def test_sin_token_no_se_ejecuta_nada(cfg, db) -> None:
    async def prueba() -> int:
        async with _cliente(cfg) as cliente:
            return (await cliente.post("/herramientas/hilo_reciente", json={})).status

    assert asyncio.run(prueba()) == 401
