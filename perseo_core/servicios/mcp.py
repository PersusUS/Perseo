"""Agente `mcp`: los servidores MCP como un puerto más.

MCP (Model Context Protocol) es una convención para que un programa le ofrezca
herramientas a otro: el cliente lanza al servidor como proceso hijo y le habla
por su entrada y salida estándar con **JSON-RPC 2.0, una línea por mensaje**.
Nada más. No hace falta SDK oficial — que arrastra httpx y compañía, cuando lo
que hay que hacer es escribir líneas de JSON y leerlas — y así el núcleo sigue
con su única dependencia.

Por qué esto entra en Perseo y cómo convive con lo que ya había:

1. **Es un puerto, igual que el buzón o el calendario.** Los agentes propios
   (`memoria`, `pc`, `web`) se quedan donde están: llevan reglas de seguridad
   que ningún servidor de terceros trae — memoria no borra, `pc` no tiene shell,
   `web` no alcanza la red de casa. MCP llega AL LADO para extender, y cada
   equivalente de terceros tendrá que ganarse el puesto usándose.
2. **La política manda aquí dentro también.** Cada servidor lleva su nivel en
   `<datos>/mcp.json` — `libre`, `reversible` o `irreversible` — y lo desconocido
   es irreversible, que desde el 2026-08-22 cuesta un «sí» hablado y no un clic.
   Además una lista de herramientas permitidas por servidor: si el día de mañana
   `server-filesystem` estrena `delete_everything`, no pasa nada hasta que alguien
   la escriba en esa lista.
3. **Sin shell, otra vez.** El comando del servidor va en lista de argumentos,
   nunca como cadena, y no se interpola nada del modelo en él. Lo que puede
   ejecutarse lo escribió una persona en el fichero del disco, como en
   `proyectos.json`.


"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..infra import politica
from ..infra.router import registrar
from ..infra.configuracion import Configuracion
from .mcp_argumentos import _firma, _parametros, recortar
from .mcp_transportes import (
    NOMBRE_FICHERO,
    TOPE_POR_DEFECTO,
    ErrorMcp,
    ServidorMcp,
    _url_aceptable,
    abrir_servidor,
)

logger = logging.getLogger(__name__)

def cargar_servidores(directorio_datos: Path) -> dict[str, dict[str, Any]]:
    """Lo que haya en `<datos>/mcp.json`, validado.

    Una entrada mala se descarta con aviso, como en `proyectos.py`: el fichero
    lo escribe una persona y perder los cinco servidores buenos por una coma es
    peor que perder el malo.
    """
    ruta = Path(directorio_datos) / NOMBRE_FICHERO
    if not ruta.exists():
        return {}
    try:
        crudo = json.loads(ruta.read_text(encoding="utf-8") or "{}")
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("No se pudo leer %s (%s); se sigue sin MCP.", ruta.name, e)
        return {}
    if not isinstance(crudo, dict):
        logger.warning("%s debería ser un objeto de servidores.", NOMBRE_FICHERO)
        return {}

    validos: dict[str, dict[str, Any]] = {}
    for nombre, entrada in crudo.items():
        if not isinstance(entrada, dict):
            logger.warning("Servidor %r mal escrito; se ignora.", nombre)
            continue
        url = str(entrada.get("url") or "").strip()
        comando = entrada.get("comando")
        if url:
            # Servidor REMOTO: no hay proceso hijo, hay una dirección. Se exige
            # HTTPS salvo en el bucle local, porque por ahí van a viajar los
            # argumentos que Perseo compone con lo que ve en pantalla.
            if not _url_aceptable(url):
                logger.warning(
                    "Servidor %r con url %r: solo https:// (o http:// en "
                    "127.0.0.1) ; se ignora.",
                    nombre,
                    url,
                )
                continue
            comando = []
        elif not isinstance(comando, list) or not all(isinstance(p, str) for p in comando):
            logger.warning(
                "Servidor %r sin 'comando' como lista de argumentos ni 'url'; se "
                "ignora. Una cadena nunca vale como comando: por ahí entran las "
                "comillas y los &&.",
                nombre,
            )
            continue
        nivel = str(entrada.get("nivel") or politica.IRREVERSIBLE).strip().lower()
        if nivel not in politica.NIVELES:
            logger.warning("Servidor %r con nivel %r raro; irreversible.", nombre, nivel)
            nivel = politica.IRREVERSIBLE
        herramientas = entrada.get("herramientas") or []
        if not isinstance(herramientas, list) or not all(isinstance(h, str) for h in herramientas):
            herramientas = []
        entorno = entrada.get("env") or {}
        if not isinstance(entorno, dict):
            entorno = {}
        try:
            tope = float(entrada.get("tope_segundos") or TOPE_POR_DEFECTO)
        except (TypeError, ValueError):
            tope = TOPE_POR_DEFECTO
        cabeceras = entrada.get("cabeceras") or {}
        if not isinstance(cabeceras, dict):
            cabeceras = {}
        # Lo que el modelo no puede saber y el servidor exige: la raíz del vault
        # en `search_files`, y lo que venga. Por herramienta, y solo rellena
        # hueco vacío — lo que el modelo diga, manda.
        por_defecto = entrada.get("argumentos_por_defecto") or {}
        if not isinstance(por_defecto, dict) or not all(
            isinstance(v, dict) for v in por_defecto.values()
        ):
            logger.warning(
                "Servidor %r: 'argumentos_por_defecto' debe ser herramienta -> objeto; se ignora.",
                nombre,
            )
            por_defecto = {}
        # El nivel fino: una herramienta que solo mira no tiene por qué heredar
        # el nivel del servidor que además toca el sistema. Lo raro se descarta
        # y esa herramienta vuelve al nivel del servidor, que es el prudente.
        niveles_herramienta = entrada.get("niveles_herramienta") or {}
        if not isinstance(niveles_herramienta, dict):
            niveles_herramienta = {}
        finos: dict[str, str] = {}
        for clave, valor in niveles_herramienta.items():
            fino = str(valor or "").strip().lower()
            if fino in politica.NIVELES:
                finos[str(clave)] = fino
            else:
                logger.warning(
                    "Servidor %r: nivel %r raro para %r; se queda con el del servidor.",
                    nombre,
                    valor,
                    clave,
                )
        validos[str(nombre)] = {
            "niveles_herramienta": finos,
            "argumentos_por_defecto": {str(k): dict(v) for k, v in por_defecto.items()},
            "comando": [str(p) for p in comando],
            "url": url,
            "cabeceras": {str(k): str(v) for k, v in cabeceras.items()},
            "nivel": nivel,
            "herramientas": herramientas,
            "env": {str(k): str(v) for k, v in entorno.items()},
            "tope_segundos": max(5.0, min(tope, 600.0)),
        }
    return validos


#: Lo que hay en `mcp.json`, cargado al arrancar el núcleo.
definiciones: dict[str, dict[str, Any]] = {}

_activos: dict[str, ServidorMcp] = {}


async def iniciar(cfg: Configuracion) -> None:
    """Carga la configuración y registra el nivel de cada servidor en la política.

    No arranca ningún proceso aquí: los servidores se lanzan la primera vez que
    alguien los usa. Arrancar de más sería tener cinco procesos hijos vivos para
    nada — y un núcleo que arranca sin MCP no debe pagar ni un milisegundo.
    """
    global definiciones
    definiciones = cargar_servidores(cfg.directorio_datos)
    politica.registrar_niveles(nivel_de)
    if definiciones:
        logger.info("MCP: %d servidor(es) configurado(s): %s", len(definiciones), ", ".join(definiciones))


async def detener() -> None:
    global definiciones
    for activo in _activos.values():
        await activo.detener()
    _activos.clear()
    definiciones = {}
    politica.registrar_niveles(None)


def _servidor(nombre: str) -> ServidorMcp:
    """El servidor vivo, arrancándolo si es la primera vez."""
    activo = _activos.get(nombre)
    if activo is not None and activo.vivo:
        return activo
    definicion = definiciones.get(nombre)
    if definicion is None:
        raise ErrorMcp(
            f"No hay ningún servidor MCP llamado '{nombre}'. Configurados: "
            + (", ".join(sorted(definiciones)) or "ninguno")
        )
    nuevo = abrir_servidor(nombre, definicion)
    return nuevo  # lo arranca quien llama, que sabe esperar


# -- La política por servidor ------------------------------------------------ #


def nivel_de(agente: str, peticion: dict[str, Any] | None) -> str | None:
    """El nivel del servidor que pide esta petición, o `None` si no es cosa nuestra.

    Lo consulta `politica.nivel` cuando su tabla no dice nada. El nivel viene del
    fichero que escribió una persona; lo que no esté ahí es irreversible por el
    camino de siempre.
    """
    if agente != "mcp":
        return None
    nombre = str((peticion or {}).get("servidor") or "").strip()
    definicion = definiciones.get(nombre)
    if definicion is None:
        return None  # servidor desconocido: cae en el irreversible por defecto

    herramienta = str((peticion or {}).get("herramienta") or "").strip()
    # Un servidor entero no es un nivel: `windows` tiene un `Snapshot` que solo
    # mira y un `Registry` que toca el sistema, y ponerle un único nivel a los
    # dos obliga a elegir entre preguntar por mirar o no preguntar por escribir.
    por_herramienta = definicion.get("niveles_herramienta") or {}
    if herramienta in por_herramienta:
        return str(por_herramienta[herramienta])

    if nombre == "windows" and herramienta == "PowerShell":
        argumentos = (peticion or {}).get("argumentos")
        comando = (argumentos or {}).get("command") if isinstance(argumentos, dict) else ""
        if _powershell_solo_lee(str(comando or "")):
            # Listar el escritorio no es una acción irreversible, y pararla para
            # pedir un sí a quien acaba de pedirla de viva voz sobra. Escribir,
            # borrar o instalar sigue costando su sí.
            return politica.REVERSIBLE

    return str(definicion["nivel"])


#: Lo que en PowerShell solo mira. La convención del lenguaje ayuda —`Get-*` lee,
#: `Remove-*` no— pero no basta: aquí están además los alias de toda la vida y
#: los verbos que solo dan forma a lo que ya salió (`Select-Object`, `Format-*`).
_POWERSHELL_LECTURA = frozenset(
    {
        "dir", "ls", "gci", "cat", "gc", "type", "pwd", "gl", "echo", "cd",
        "test-path", "resolve-path", "split-path", "join-path", "convert-path",
        "select-object", "select", "sort-object", "sort", "where-object", "where",
        "measure-object", "measure", "format-table", "ft", "format-list", "fl",
        "out-string", "convertto-json", "convertfrom-json", "group-object", "group",
        "select-string", "compare-object", "write-output", "write-host", "foreach-object",
    }
)

#: Lo que parte un comando en dos: una tubería, un `;`, una subexpresión, una
#: redirección. Cada trozo se juzga por separado, porque `Get-ChildItem |
#: Remove-Item` empieza leyendo y acaba borrando.
_SEPARADORES = ("|", ";", "&&", "||", "\n", "\r")


def _powershell_solo_lee(comando: str) -> bool:
    """¿Este comando solo mira? Ante la duda, no.

    No es un analizador de PowerShell y no pretende serlo: es una lista de lo
    que se reconoce como lectura, y todo lo demás cae del lado que pregunta. Un
    falso negativo cuesta un «sí»; un falso positivo dejaría borrar sin avisar.
    """
    texto = comando.strip()
    if not texto:
        return False
    # Una redirección escribe un fichero, y una subexpresión o un acento grave
    # esconden otro comando dentro. Nada de eso pasa por aquí.
    if any(c in texto for c in ">`$"):
        return False

    trozos = [texto]
    for separador in _SEPARADORES:
        trozos = [parte for trozo in trozos for parte in trozo.split(separador)]

    for trozo in trozos:
        palabras = trozo.strip().split()
        if not palabras:
            continue
        verbo = palabras[0].strip("(").lower()
        if verbo in _POWERSHELL_LECTURA:
            continue
        # `Get-`, `Show-`, `Find-` y `Test-` son los verbos de lectura de
        # PowerShell; `Get-Credential` es la excepción y se queda fuera.
        if verbo.startswith(("get-", "show-", "find-", "test-")) and verbo != "get-credential":
            continue
        return False
    return True


# -- El agente --------------------------------------------------------------- #


@registrar("mcp")
async def _mcp(trabajo: dict[str, Any]) -> dict[str, Any]:
    """Lista los servidores y sus herramientas, o llama a una herramienta."""
    peticion = trabajo.get("peticion") or {}
    accion = str(peticion.get("accion", "")).strip().lower()

    if accion == "servidores":
        return await _listar()

    if accion == "llamar":
        return await _llamar(peticion)

    raise ValueError(f"Acción desconocida para mcp: {accion!r}. Válidas: servidores, llamar.")


async def _listar() -> dict[str, Any]:
    if not definiciones:
        return {
            "texto": "No hay servidores MCP configurados.",
            "titular": None,
            "servidores": [],
        }

    lineas: list[str] = []
    resumen: list[dict[str, Any]] = []
    for nombre in sorted(definiciones):
        try:
            servidor = _activos.get(nombre)
            if servidor is None or not servidor.vivo:
                servidor = abrir_servidor(nombre, definiciones[nombre])
                await servidor.arrancar()
                _activos[nombre] = servidor
        except (ErrorMcp, OSError) as e:
            logger.warning("MCP '%s' no arrancó: %s", nombre, e)
            lineas.append(f"- {nombre}: no arrancó ({recortar(str(e), 120)})")
            resumen.append({"nombre": nombre, "error": recortar(str(e), 160)})
            continue

        # En el catálogo solo se enseñan las permitidas: que el modelo vea
        # `write_file` detrás de una lista que lo veta es invitarle a probar.
        permitidas = servidor._permitidas()
        visibles = [h for h in servidor.herramientas if not permitidas or str(h.get("name")) in permitidas]
        if visibles:
            # Con la firma, y no solo el nombre. Sin esto el modelo tiene que
            # adivinar cómo se llaman los parámetros —y adivina en español,
            # que es como se pierde una llamada por escribir `comando` donde
            # ponía `command`.
            lineas.append(f"- {nombre}:")
            for h in visibles:
                lineas.append(f"    {_firma(h)}")
        else:
            lineas.append(f"- {nombre}: sin herramientas")
        resumen.append(
            {
                "nombre": nombre,
                "nivel": str(definiciones[nombre]["nivel"]),
                "herramientas": [
                    {
                        "nombre": str(h.get("name")),
                        "descripcion": recortar(str(h.get("description") or ""), 200),
                        "parametros": _parametros(h),
                    }
                    for h in visibles
                ],
            }
        )

    texto = "Servidores MCP disponibles:\n" + "\n".join(lineas)
    return {"texto": texto, "titular": f"{len(resumen)} servidor(es) MCP", "servidores": resumen}


async def _llamar(peticion: dict[str, Any]) -> dict[str, Any]:
    nombre = str(peticion.get("servidor") or "").strip()
    herramienta = str(peticion.get("herramienta") or "").strip()
    argumentos = peticion.get("argumentos")
    if not nombre or not herramienta:
        raise ValueError("Para llamar a una herramienta hacen falta 'servidor' y 'herramienta'.")
    if argumentos is None:
        argumentos = {}
    if isinstance(argumentos, str):
        # Un modelo de voz manda a veces el objeto ya escrito como texto. Es
        # JSON válido: leerlo cuesta una línea y salva la llamada entera.
        try:
            argumentos = json.loads(argumentos or "{}")
        except json.JSONDecodeError:
            raise ValueError(
                "'argumentos' vino como texto y no es JSON: manda un objeto con los parámetros."
            ) from None
    if not isinstance(argumentos, dict):
        raise ValueError("'argumentos' debe ser un objeto con los parámetros de la herramienta.")

    servidor = _servidor(nombre)
    if not servidor.vivo:
        await servidor.arrancar()
        _activos[nombre] = servidor

    respuesta = await servidor.llamar(herramienta, argumentos)
    logger.info("MCP %s.%s contestó %d carácter(es).", nombre, herramienta, len(respuesta))
    titular = recortar(f"{nombre}.{herramienta}()", 120)
    return {
        "texto": respuesta or "(la herramienta no devolvió texto)",
        "titular": titular,
        "servidor": nombre,
        "herramienta": herramienta,
    }
