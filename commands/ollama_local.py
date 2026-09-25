"""¿Sirve el Ollama que contesta en el 11434?

Que conteste no basta. En el PC del señor Persus hay **dos** Ollama: el de
Windows, con `qwen3:4b`, y uno dentro de WSL Ubuntu que solo tiene
`qwen2.5-coder:1.5b`. Con el de WSL encendido, `wslrelay.exe` se queda el puerto
y el 11434 da el mismo 200 que daría el bueno. `perseo on` se fiaba del 200,
decía «ya estaba», no arrancaba el de Windows, y el triaje del correo fallaba
pidiendo un modelo que ese Ollama no tiene.

Aquí se pregunta lo que importa —qué modelos tiene quien contesta, y quién es— y
se dice qué hacer. Vive fuera de `perseo.py` porque tiene costura propia, y
porque dentro el lanzador pasaba del techo de 900 líneas.

Como `perseo.py`, **no importa el núcleo**: tiene que funcionar sin sus
dependencias instaladas. Lo que repite de él —de dónde sale el modelo del router
y cuándo vale uno— lo atan dos pruebas de `pruebas/test_ollama_local.py`.
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

PUERTO = 11434

#: El modelo que pide el triaje si nadie dice otro. Es el de fábrica de
#: `perseo_core/infra/configuracion.py`; una prueba compara los dos.
MODELO_ROUTER_DE_FABRICA = "qwen3:4b"

#: Que PowerShell no abra ventana. El porqué largo está en `perseo.SIN_VENTANA`;
#: no se importa de allí porque `perseo` importa este módulo.
SIN_VENTANA = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def modelos() -> list[str] | None:
    """Los modelos de lo que conteste en el 11434, o `None` si no contesta nadie.

    `None` y `[]` no son lo mismo, y el lanzador los trata distinto: `None` es
    «arráncalo»; `[]` es «hay alguien en el puerto y no sirve». Por eso un 200
    que no es JSON cuenta como vivo y sin modelos: lo que esté ahí no es un
    Ollama que sirva, y arrancar el de Windows encima no arreglaría nada.
    """
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PUERTO}/api/tags", timeout=3) as r:
            if r.status != 200:
                return None
            cuerpo = r.read()
    except (urllib.error.URLError, OSError):
        return None
    try:
        datos = json.loads(cuerpo.decode("utf-8"))
    except ValueError:
        return []
    lista = datos.get("models") if isinstance(datos, dict) else None
    return [str(m.get("name", "")) for m in lista or [] if isinstance(m, dict)]


def modelo_router(directorio_datos: Path) -> str:
    """El modelo que pedirá el triaje, resuelto como lo resuelve el núcleo: la
    variable de entorno, luego `<datos>/entorno.json`, luego el de fábrica.

    Tiene que salir el mismo que el del núcleo. Si no, se daría por bueno un
    Ollama con un modelo que el triaje no va a pedir, que es el fallo de antes.
    """
    del_entorno = os.environ.get("PERSEO_MODELO_ROUTER")
    if del_entorno is not None:
        return del_entorno
    try:
        # `utf-8-sig` por lo mismo que en el núcleo: el Bloc de notas le pone BOM.
        guardados = json.loads((directorio_datos / "entorno.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        guardados = {}
    if isinstance(guardados, dict) and "PERSEO_MODELO_ROUTER" in guardados:
        return str(guardados["PERSEO_MODELO_ROUTER"])
    return MODELO_ROUTER_DE_FABRICA


def tiene_el_modelo(nombres: list[str], modelo: str) -> bool:
    """Si entre esos nombres está el modelo, con la regla de la pantalla de estado
    (`perseo_core/caras/estado.py`): la etiqueta exacta, u otra de la misma
    familia —`qwen3:8b` vale por `qwen3:4b`—.

    Las dos tienen que decir lo mismo: si el panel pintara verde lo que `perseo
    on` avisa, o al revés, no habría a cuál creer.
    """
    familia = modelo.split(":")[0]
    return modelo in nombres or any(n.split(":")[0] == familia for n in nombres)


def quien_tiene_el_puerto(puerto: int = PUERTO) -> str | None:
    """El nombre del proceso que escucha en ese puerto, o `None` si no se sabe.

    Es lo que distingue a los dos Ollama sin adivinar: el de Windows sale como
    `ollama`, y el de WSL como `wslrelay`, que es el proceso con el que WSL le
    pasa a Windows los puertos que se abren dentro. Por PowerShell y no por
    `netstat` porque netstat escribe el estado traducido —«ESCUCHANDO» en un
    Windows español— y habría que leerlo en cada idioma.
    """
    if os.name != "nt":
        return None
    orden = (
        f"$c = Get-NetTCPConnection -LocalPort {puerto} -State Listen "
        "-ErrorAction SilentlyContinue | Select-Object -First 1; "
        "if ($c) { (Get-Process -Id $c.OwningProcess -ErrorAction SilentlyContinue).ProcessName }"
    )
    try:
        salida = subprocess.run(
            ["powershell", "-NoProfile", "-Command", orden],
            capture_output=True,
            creationflags=SIN_VENTANA,
            text=True,
            errors="replace",
            timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return salida.strip() or None


def avisar_sin_modelo(modelo: str) -> None:
    """Dice que al Ollama del 11434 le falta el modelo, y quién es ese Ollama.

    Se nombra al dueño del puerto porque el arreglo depende de quién sea: al de
    Windows le falta un `ollama pull`; al de WSL no hay que bajarle nada, hay
    que pararlo para que el puerto quede para el de Windows.
    """
    print(f"  [ojo]        Ollama contesta en el {PUERTO}, pero sin {modelo}: no hay triaje")
    dueno = quien_tiene_el_puerto(PUERTO)
    if dueno is None:
        print("               No se sabe qué proceso tiene el puerto. Míralo con:")
        print(f"               Get-NetTCPConnection -LocalPort {PUERTO} -State Listen")
    elif dueno.lower() == "wslrelay":
        print(f"               El puerto lo tiene {dueno}: es el Ollama de WSL, no el de Windows.")
        # Allí es un servicio de systemd, activo y habilitado (medido el
        # 2026-09-24): con `stop` vuelve la próxima vez que arranque WSL.
        print("               Páralo: wsl -u root systemctl stop ollama")
        print("               (con `disable --now` en vez de `stop`, no vuelve al arrancar WSL)")
        print("               y luego `perseo on` otra vez.")
    elif dueno.lower() == "ollama":
        print(f"               Lo tiene el Ollama de Windows. Bájalo: ollama pull {modelo}")
    else:
        print(f"               El puerto lo tiene {dueno}, que no es ninguno de los dos Ollama.")
