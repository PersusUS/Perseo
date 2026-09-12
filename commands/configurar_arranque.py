"""Escribe `perseo_core/datos/entorno.json` mirando lo que ya está configurado.

El núcleo que arranca con Windows no hereda las variables que uno escribe en su
terminal, así que sin este fichero se levanta capado: sin correo, sin agenda y
escribiendo el vault a fichero en vez de por Obsidian. Y arranca **bien**, que es
lo peor: parece que funciona.

Este script no pregunta nada. Mira qué credenciales hay puestas y decide en
consecuencia — si están las de Google, enciende Gmail y Calendar; si está la
clave del plugin, pone el vault por Obsidian; si hay Tailscale, abre la interfaz
del tailnet para que el móvil llegue y el enlace de Telegram sirva.

    python commands/configurar_arranque.py            # lo escribe
    python commands/configurar_arranque.py --ver      # solo dice qué haría
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from perseo_core.infra.configuracion import _directorio_datos, direccion_tailscale  # noqa: E402


def _vault_de_verdad() -> Path | None:
    """Dónde está el vault grande de verdad: una carpeta de Documents con su
    `.obsidian` dentro. El de dentro de Perseo es el de fábrica de la memoria,
    no el segundo cerebro que se mira en el grafo."""
    documentos = Path.home() / "Documents"
    if not documentos.is_dir():
        return None
    for carpeta in sorted(documentos.iterdir()):
        if carpeta.is_dir() and (carpeta / ".obsidian").is_dir():
            return carpeta
    return None


def ajustes_recomendados(datos: Path, hay_tailscale: bool) -> dict[str, str]:
    """Qué poner en `entorno.json` según lo que esté configurado.

    Deliberadamente conservador: lo que no tiene credenciales no se enciende. Un
    disparador sin de dónde tirar se retira solo al arrancar, pero dejarlo puesto
    llenaría el registro de avisos que no dicen nada.
    """
    ajustes: dict[str, str] = {}

    if hay_tailscale:
        # Sin esto el enlace de "ver detalle" de Telegram apunta al bucle local
        # y en el móvil abre una página en blanco. Es.
        ajustes["PERSEO_CORE_HOST"] = "tailscale"

    if (datos / "obsidian.txt").is_file():
        ajustes["PERSEO_VAULT"] = "rest"

    # Con el vault por el plugin, Obsidian sabe dónde está el suyo; el grafo del
    # segundo cerebro lee el disco directamente y necesita la ruta por escrito.
    vault = _vault_de_verdad()
    if vault:
        ajustes["OBSIDIAN_VAULT_PATH"] = str(vault)

    if (datos / "google.json").is_file():
        credenciales = {}
        try:
            credenciales = json.loads((datos / "google.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            credenciales = {}
        # Sin `refresh_token` el consentimiento no está dado: encenderlos sería
        # programar un fallo cada cinco minutos.
        if (credenciales.get("installed") or credenciales).get("refresh_token"):
            ajustes["PERSEO_CORREO"] = "gmail"
            ajustes["PERSEO_AGENDA"] = "google"

    return ajustes


def main() -> int:
    solo_ver = "--ver" in sys.argv
    datos = Path(_directorio_datos())
    hay_tailscale = bool(direccion_tailscale())

    ajustes = ajustes_recomendados(datos, hay_tailscale)
    destino = datos / "entorno.json"

    print("Lo que arrancará con Windows:\n")
    if not ajustes:
        print("  (nada configurado todavía: el núcleo arrancará en mínimos)")
    for clave, valor in ajustes.items():
        print(f"  {clave} = {valor}")

    if not hay_tailscale:
        print("\n  [aviso] Tailscale no responde. El núcleo solo escuchará en el bucle local,")
        print(" y el enlace de Telegram no servirá desde el móvil.")

    if solo_ver:
        print(f"\nNo se ha escrito nada. Quita --ver para dejarlo en {destino}.")
        return 0

    destino.write_text(json.dumps(ajustes, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nEscrito en {destino}.")
    print("Compruébalo con:  python commands/manage_startup.py status")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
