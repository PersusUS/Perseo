"""Que los argumentos que manda el modelo encajen con lo que pide el servidor.

Un servidor MCP declara sus herramientas con un esquema JSON y rechaza lo que no
encaje. El modelo, sin embargo, escribe `time_zone` donde el esquema dice
`timezone`, manda un número donde se espera texto, o llama a `search_files` con
la forma de otra herramienta. Eso no es un fallo del modelo: es lo que pasa
cuando alguien describe una API en prosa.

Aquí está lo que acomoda una cosa a la otra antes de llamar, y lo que explica el
esquema por escrito cuando no hay acomodo posible — para que el modelo pueda
arreglarlo en el siguiente intento en vez de repetir el mismo error.

Salió de `mcp.py` el 2026-09-12: nada de esto sabe de tuberías ni de HTTP, y
tenerlo en medio hacía el fichero de los transportes ilegible.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _es_error_de_argumentos(mensaje: str) -> bool:
    """¿El servidor rechazó la llamada por los parámetros, y no por otra cosa?"""
    bajo = mensaje.lower()
    return (
        "32602" in bajo
        or "invalid arguments" in bajo
        or "input validation" in bajo
        or "validation error" in bajo
        or "missing required argument" in bajo
    )


def _esquema_de(herramientas: list[dict[str, Any]], nombre: str) -> dict[str, Any]:
    """El `inputSchema` que el servidor publicó para esa herramienta."""
    for h in herramientas:
        if str(h.get("name")) == nombre:
            esquema = h.get("inputSchema")
            return esquema if isinstance(esquema, dict) else {}
    return {}


def _propiedades(esquema: dict[str, Any]) -> dict[str, Any]:
    props = (esquema or {}).get("properties")
    return props if isinstance(props, dict) else {}


def _requeridos(esquema: dict[str, Any]) -> list[str]:
    req = (esquema or {}).get("required")
    return [str(k) for k in req] if isinstance(req, list) else []


def _pista_esquema(esquema: dict[str, Any], herramienta: str) -> str:
    """Los parámetros de una herramienta, en prosa corta, para el modelo.

    Va pegada a cualquier error de argumentos: quien se equivocó de nombre lee
    ahí mismo cómo se llaman de verdad y reintenta bien, en vez de repetir el
    mismo fallo hasta que alguien se rinde.
    """
    props = _propiedades(esquema)
    if not props:
        return ""
    requeridos = _requeridos(esquema)
    lineas = []
    for clave, valor in props.items():
        detalle = valor if isinstance(valor, dict) else {}
        tipo = detalle.get("type") or "?"
        marca = "requerido" if clave in requeridos else "opcional"
        descripcion = recortar(str(detalle.get("description") or ""), 80)
        lineas.append(f"  - {clave}: {tipo} ({marca}){' — ' + descripcion if descripcion else ''}")
    return f"\n\nParámetros de '{herramienta}':\n" + "\n".join(lineas)


#: Lo que el modelo escribe cuando no ha mirado el esquema. Un servidor MCP
#: nombra sus parámetros en inglés; Perseo piensa en español y ese idioma se le
#: cuela hasta la llamada — de ahí un `{"comando": ...}` contra un `PowerShell`
#: que espera `command`, y una llamada perdida por una palabra. La traducción
#: solo entra si el esquema tiene el nombre bueno y la llamada no lo traía ya:
#: nunca inventa un campo ni pisa lo que el modelo escribió bien.
_ALIAS_ARGUMENTOS: dict[str, tuple[str, ...]] = {
    "comando": ("command",),
    "orden": ("command",),
    "ruta": ("path",),
    "archivo": ("path",),
    "fichero": ("path",),
    "carpeta": ("path",),
    "directorio": ("path",),
    "destino": ("destination",),
    "patron": ("pattern",),
    "patrón": ("pattern",),
    "busqueda": ("pattern", "query"),
    "búsqueda": ("pattern", "query"),
    "consulta": ("query", "pattern"),
    "texto": ("text",),
    "contenido": ("content", "text"),
    "titulo": ("title",),
    "título": ("title",),
    "mensaje": ("message",),
    "modo": ("mode",),
    "atajo": ("shortcut",),
    "duracion": ("duration",),
    "duración": ("duration",),
    "zona_horaria": ("timezone",),
    "nombre": ("name",),
    "condicion": ("condition",),
    "condición": ("condition",),
}


def _acomodar(
    argumentos: dict[str, Any],
    esquema: dict[str, Any],
    por_defecto: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Los argumentos del modelo, puestos en los nombres que el servidor espera.

    Dos arreglos y ninguno más: traducir el nombre español al del esquema, y
    poner lo que el fichero declare por defecto para esa herramienta (la ruta
    del vault, por ejemplo, que el modelo nunca sabe y el servidor exige).
    """
    salida = dict(argumentos)
    props = _propiedades(esquema)
    if props:
        for clave in list(salida):
            if clave in props:
                continue
            for candidato in _ALIAS_ARGUMENTOS.get(str(clave).lower(), ()):
                if candidato in props and candidato not in salida:
                    salida[candidato] = salida.pop(clave)
                    break
    for clave, valor in (por_defecto or {}).items():
        if salida.get(clave) in (None, ""):
            salida[clave] = valor
    return salida


def _faltan_requeridos(argumentos: dict[str, Any], esquema: dict[str, Any]) -> list[str]:
    """Los requeridos que no vienen. Mejor decirlo aquí que gastar un viaje."""
    if not _propiedades(esquema):
        return []
    return [k for k in _requeridos(esquema) if argumentos.get(k) in (None, "")]


def _normalizar_search_files(args: dict[str, Any]) -> dict[str, Any]:
    """
    Convierte lenguaje natural en glob pattern para search_files del vault.
    El servidor MCP server-filesystem espera glob patterns (ej: *música*.md),
    no texto libre. Si el modelo manda palabras sueltas, las envolvemos.
    """
    args = dict(args)  # copia
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern.strip():
        return args

    # Ya parece un glob (contiene *, ?, [, ], {, })
    if any(c in pattern for c in "*?[]{"):
        return args

    # Lenguaje natural: envolvemos en *...* y añadimos .md si no tiene extensión
    palabras = pattern.strip().split()
    if len(palabras) == 1:
        base = palabras[0]
    else:
        # Múltiples palabras: probamos la más larga (más específica)
        base = max(palabras, key=len)

    # Sin añadir extensión: un `*musica.md*` solo casa con quien lleve
    # «musica.md» dentro del nombre, que no es ningún fichero. `*musica*` casa
    # con «Musica.md» y con «lista de musica.txt», que es lo que se buscaba.
    args["pattern"] = f"*{base}*"
    return args


def _parametros(herramienta: dict[str, Any]) -> list[dict[str, Any]]:
    """Los parámetros de una herramienta, tal como el catálogo los enseña."""
    esquema = herramienta.get("inputSchema")
    esquema = esquema if isinstance(esquema, dict) else {}
    requeridos = _requeridos(esquema)
    salida = []
    for clave, valor in _propiedades(esquema).items():
        detalle = valor if isinstance(valor, dict) else {}
        salida.append(
            {
                "nombre": str(clave),
                "tipo": str(detalle.get("type") or "?"),
                "requerido": clave in requeridos,
                "descripcion": recortar(str(detalle.get("description") or ""), 120),
            }
        )
    return salida


def _firma(herramienta: dict[str, Any]) -> str:
    """`nombre(requerido, [opcional])`, que es lo que el modelo necesita leer."""
    partes = [
        p["nombre"] if p["requerido"] else f"[{p['nombre']}]" for p in _parametros(herramienta)
    ]
    nombre = str(herramienta.get("name"))
    descripcion = recortar(str(herramienta.get("description") or ""), 100)
    firma = f"{nombre}({', '.join(partes)})"
    return f"{firma} — {descripcion}" if descripcion else firma


def recortar(texto: str, tope: int = 300) -> str:
    limpio = " ".join(str(texto).split())
    return limpio if len(limpio) <= tope else limpio[: tope - 1].rstrip() + "…"


# -- Ciclo de vida del módulo ------------------------------------------------ #

#: Los procesos vivos, uno por servidor configurado.
