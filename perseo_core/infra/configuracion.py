"""La configuración del núcleo: de dónde sale cada ajuste y con qué manda.

Salió de `almacen.py` el 2026-09-12, cuando el fichero pasaba de mil doscientas
líneas. La costura estaba clara: una cosa es **la cola y su base de datos** y
otra **de qué está configurado el sistema**, y solo compartían vivir juntas.

El orden de precedencia de un ajuste, que es lo que hay que saber antes de tocar
nada: variable de entorno primero, luego `datos/entorno.json`, luego el fichero
suelto de la clave, y al final lo de fábrica. Está así para que probar algo con
una variable no obligue a editar ficheros, y para que lo editado sobreviva a un
reinicio.

Aquí vive también la resolución del tailnet. No es un detalle de red: es lo que
decide en qué interfaces escucha el núcleo, y **nunca es `0.0.0.0`**.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import secrets
import socket
import subprocess
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

#: La raíz del paquete: un escalón por encima de `infra/`. De aquí cuelgan
#: `datos/` y, un escalón más arriba, el vault por defecto.
RAIZ = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Configuracion:
    """Ajustes del núcleo, resueltos una vez al arrancar."""

    #: Interfaces en las que se escucha. Es una lista porque en la Fase B el
    #: núcleo atiende a la vez al PC (bucle local) y al móvil (Tailscale), y
    #: enumerarlas es lo que permite no caer nunca en `0.0.0.0`.
    hosts: tuple[str, ...]
    puerto: int
    token: str
    ruta_db: Path
    directorio_datos: Path
    url_ollama: str
    modelo_router: str
    #: Credenciales del bot. Vacías significa "sin Telegram", y el núcleo
    #: arranca igual: es un canal más, no una pieza de la que dependa nada.
    telegram_token: str
    telegram_chat: str
    #: Se puede apuntar a otro sitio para probar sin tocar Telegram de verdad.
    telegram_api: str
    #: La dirección que se pone en el enlace "ver detalle" de las
    #: notificaciones. Va a un mensaje que sale de la máquina, así que apunta al
    #: tailnet y no al bucle local.
    url_base: str
    #: Disparadores que se ponen en marcha al arrancar (Fase D). Vacío es un
    #: estado válido: el núcleo funciona igual, solo que nadie empieza nada solo.
    disparadores: tuple[str, ...]
    #: Cada cuántos segundos le toca a cada disparador. Lo que no esté aquí usa
    #: el valor con el que se registró.
    intervalos: dict[str, float]
    #: De dónde salen los correos: `falso` (fichero, para verificar) o vacío.
    #: Cuando haya credenciales OAuth se añadirá `gmail`.
    correo_buzon: str
    #: Ruta del JSON que hace de buzón cuando `correo_buzon` es `falso`.
    correo_falso: str
    #: Motor del agente `dev`: vacío (Claude Code por línea de comandos) o
    #: `falso`, que simula sin gastar suscripción.
    dev_motor: str
    #: Qué se ejecuta cuando el motor es el de verdad.
    dev_ejecutable: str
    #: Desde dónde trabaja un encargo de código, y de dónde no puede salir.
    #: Por defecto la CARPETA DEL USUARIO, no este repositorio: los proyectos
    #: del señor Persus están interconectados y un agente encerrado en uno solo
    #: no puede leer el de al lado (2026-08-26). Se estrecha con
    #: `PERSEO_DEV_RAIZ` si alguna vez hace falta.
    dev_raiz: str
    #: Segundos que se le dan a un encargo antes de cortarlo.
    dev_tope: float
    #: Lo que tarda el motor falso, para poder comprobar que un encargo largo no
    #: deja al resto de la cola esperando.
    dev_tardanza_falsa: float
    #: Fichero con las credenciales de Google (Gmail y Calendar). Va en el
    #: directorio de datos, que está fuera de git: lleva un `refresh_token`.
    google_credenciales: str
    #: Navegador del agente `web`: vacío (HTTP de verdad) o `falso`.
    web_navegador: str
    #: Cuánto se descarga como mucho de una página.
    web_tope_bytes: int
    #: Cuánto se espera a una página.
    web_tope_segundos: float
    #: **Solo para las verificaciones.** Deja alcanzar el bucle local, que en
    #: producción está prohibido: sin esto no se podría comprobar el camino real
    #: contra un servidor de prueba. Ver `web.comprobar_url`.
    web_local: bool
    #: De dónde salen los eventos: `falso` (fichero, para verificar) o vacío.
    agenda_origen: str
    #: Ruta del JSON que hace de calendario cuando `agenda_origen` es `falso`.
    agenda_falsa: str
    #: Con cuántos minutos de antelación se avisa de un evento.
    agenda_antelacion: int
    #: Raíz del vault de Obsidian. Se sigue leyendo de `OBSIDIAN_VAULT_PATH`,
    #: que es la variable que ya usaba el indexador de v1: quien la tuviera
    #: puesta no tiene que cambiar nada. Que las rutas del vault no coincidieran
    #: entre módulos fue.
    vault: str
    #: Qué hay detrás del puerto del vault: vacío (ficheros, como hasta ahora) o
    #: `rest`, el plugin Local REST API de Obsidian. Con `rest` la ruta del
    #: vault deja de usarse: quien sabe dónde están las notas es Obsidian.
    vault_respaldo: str
    #: Dónde escucha el plugin. Por defecto su HTTPS del bucle local, que es lo
    #: que trae encendido de fábrica.
    vault_rest_url: str
    #: La clave del plugin, que sale en sus ajustes. Si no está en la variable
    #: se lee de `<datos>/obsidian.txt`, igual que el token de Telegram: el
    #: directorio de datos está fuera de git.
    vault_rest_clave: str
    #: Modelo de fuera que responde cuando Ollama no está: `gemma-4-31b-it`, por
    #: ejemplo. **Vacío = apagado**, y es lo que viene de fábrica: es la única
    #: pieza que manda a un tercero el texto que se está clasificando.
    modelo_suplente: str
    #: Clave de la API de Gemini, para el suplente. Se comparte con la que usa la
    #: app para la voz: `GEMINI_API_KEY`, o `<datos>/gemini.txt`.
    gemini_clave: str
    #: Certificado y clave para servir por HTTPS. Existen por el micrófono: el
    #: navegador solo deja grabar en un contexto seguro, y `http://` por el
    #: tailnet no lo es —el bucle local sí, por eso en el PC se puede probar sin
    #: esto—. Los da `tailscale cert`. Vacíos = HTTP de siempre.
    tls_certificado: str
    tls_clave: str
    #: Puerto del HTTPS. **Aparte del de siempre y no en su lugar**: un socket
    #: que habla TLS no contesta a quien llega en claro, así que servir HTTPS en
    #: el puerto de siempre no cambia la dirección, la rompe — y con ella los
    #: accesos directos que ya hay guardados. Ver `_donde_escuchar`.
    tls_puerto: int

    @property
    def telegram_configurado(self) -> bool:
        return bool(self.telegram_token and self.telegram_chat)

    @property
    def tls_listo(self) -> bool:
        """Si hay con qué servir HTTPS. Que los ficheros existan se mira aquí:
        una ruta escrita a mano que ya no apunta a nada dejaría al núcleo sin
        arrancar, y prefiero HTTP a nada."""
        if not (self.tls_certificado and self.tls_clave):
            return False
        return Path(self.tls_certificado).is_file() and Path(self.tls_clave).is_file()

    @property
    def url_base_alcanzable(self) -> bool:
        """Si el enlace que sale por Telegram sirve desde fuera de esta máquina.

        `127.0.0.1` en el móvil es **el móvil**: el enlace abre una página en
        blanco y nadie sabe por qué. Pasa siempre que se arranca sin
        `PERSEO_CORE_HOST=tailscale`, porque entonces la única interfaz es la
        local y `_url_por_defecto` no tiene otra cosa que ofrecer.
        """
        anfitrion = urllib.parse.urlsplit(self.url_base).hostname or ""
        return anfitrion not in LOCALES


#: Rango que Tailscale reparte entre los nodos del tailnet (CGNAT).
_RED_TAILSCALE = ipaddress.ip_network("100.64.0.0/10")

#: Sitios donde suele estar la herramienta de Tailscale en Windows y en Linux.
_RUTAS_TAILSCALE = (
    Path(r"C:\Program Files\Tailscale\tailscale.exe"),
    Path("/usr/bin/tailscale"),
    Path("/usr/local/bin/tailscale"),
)

LOCALES = ("127.0.0.1", "localhost", "::1")

#: Que preguntarle su IP a Tailscale no abra una consola. El núcleo arranca sin
#: ventana —lo lanza `pythonw` desde el vigilante—, y `tailscale.exe` es un
#: programa de consola: sin esto parpadeaba una caja negra en cada arranque
#: La salida se captura, así que nadie se pierde nada.
_SIN_VENTANA = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def direcciones_tailscale() -> tuple[str, ...]:
    """Las direcciones de esta máquina en el tailnet: la IPv4 y la IPv6.

    **Las dos, y esto costó una mañana el 2026-08-17.** MagicDNS publica para
    cada nodo un registro `A` y otro `AAAA` —la `fd7a:…`—, y un iPhone que
    resuelve por el túnel prefiere la IPv6. Escuchando solo en la IPv4, entrar
    por la dirección numérica funcionaba y entrar por el nombre no: el móvil
    llamaba a una puerta donde no había nadie. Desde el PC no se veía, porque
    ahí el nombre resolvía a la IPv4.

    La IPv4 va primero: es la que se pone en los enlaces (`_url_por_defecto`),
    donde una IPv6 entre corchetes solo estorba.
    """
    for ruta in _RUTAS_TAILSCALE:
        if not ruta.exists():
            continue
        try:
            salida = subprocess.run(
                [str(ruta), "ip"],
                capture_output=True,
                creationflags=_SIN_VENTANA,
                text=True,
                timeout=10,
                check=False,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            continue
        encontradas = tuple(linea.strip() for linea in salida.splitlines() if linea.strip())
        if encontradas:
            return encontradas

    suelta = _direccion_tailscale_por_interfaz()
    return (suelta,) if suelta else ()


def direccion_tailscale() -> str | None:
    """La IPv4 del tailnet. Se conserva porque es la que va en los enlaces."""
    for direccion in direcciones_tailscale():
        if ":" not in direccion:
            return direccion
    return None


def _direccion_tailscale_por_interfaz() -> str | None:
    """Sin la herramienta de Tailscale, se busca una interfaz del rango CGNAT.

    Funciona, pero conviene saber que ese rango también lo usan algunos
    operadores en la interfaz de salida: por eso es el segundo intento.
    """

    try:
        vistas = {
            info[4][0]
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
        }
    except OSError:
        return None
    for direccion in sorted(vistas):
        if ipaddress.ip_address(direccion) in _RED_TAILSCALE:
            return direccion
    return None


def _resolver_hosts(crudo: str) -> tuple[str, ...]:
    """Convierte `PERSEO_CORE_HOST` en la lista de interfaces donde escuchar.

    Acepta varias direcciones separadas por comas y la palabra `tailscale`, que
    se sustituye por la dirección del tailnet. `tailscale` **no** se resuelve a
    `0.0.0.0` cuando falla: si Tailscale no está levantado, se avisa y se queda
    solo en local. Abrir todas las interfaces por no encontrar una es
    exactamente el fallo que la Fase 3 quería evitar.
    """
    resueltos: list[str] = []
    for parte in crudo.split(","):
        pieza = parte.strip()
        if not pieza:
            continue
        if pieza.lower() != "tailscale":
            resueltos.append(pieza)
            continue

        direcciones = direcciones_tailscale()
        if not direcciones:
            logger.error(
                "Se pidió escuchar en Tailscale pero no se encontró la dirección del "
                "tailnet. ¿Está Tailscale conectado? Se sigue solo en local."
            )
            continue
        # El bucle local va siempre con Tailscale: si no, la app del PC y las
        # pruebas dejarían de poder hablar con el núcleo.
        resueltos.append("127.0.0.1")
        # Las dos del tailnet, IPv4 e IPv6: MagicDNS publica un registro de cada
        # tipo y un iPhone resuelve la IPv6 primero. Con solo la IPv4, entrar
        # por el nombre no llegaba a ninguna parte.
        resueltos.extend(direcciones)

    if not resueltos:
        resueltos.append("127.0.0.1")

    # Sin duplicados y en orden estable: aiohttp falla si se le repite una.
    return tuple(dict.fromkeys(resueltos))


def _directorio_datos() -> Path:
    ruta = Path(os.environ.get("PERSEO_CORE_DATOS", RAIZ / "datos"))
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta


def _resolver_token(directorio: Path) -> str:
    """Devuelve el token de acceso, generándolo la primera vez.

    La variable de entorno manda. Si no está, se usa (o se crea) un fichero en
    el directorio de datos, que está fuera de git.
    """
    del_entorno = os.environ.get("PERSEO_TOKEN", "").strip()
    if del_entorno:
        return del_entorno

    fichero = directorio / "token.txt"
    if fichero.exists():
        guardado = fichero.read_text(encoding="utf-8").strip()
        if guardado:
            return guardado

    nuevo = secrets.token_urlsafe(32)
    fichero.write_text(nuevo, encoding="utf-8")
    logger.warning(
        "Token de acceso generado en %s. Guárdalo: lo necesitas para hablar con el núcleo.",
        fichero,
    )
    return nuevo


def ajustes_guardados(directorio: Path) -> dict[str, str]:
    """Lo que hay en `<datos>/entorno.json`, o nada si no existe.

    **Por qué existe este fichero.** Una entrada del registro de Windows arranca
    un proceso sin las variables de entorno que uno escribe en su terminal. Sin
    esto, el núcleo que arranca con el PC es otro núcleo: sin Gmail, sin agenda,
    sin el vault por Obsidian, y con el enlace de Telegram apuntando al bucle
    local. Arranca, no falla, y hace la mitad — que es peor que no arrancar.

    Un JSON plano de `VARIABLE: valor`. La variable de entorno manda sobre él:
    esto son los valores por defecto de esta instalación, no una orden.
    """
    fichero = directorio / "entorno.json"
    if not fichero.is_file():
        return {}
    try:
        # `utf-8-sig` y no `utf-8`: este fichero se edita a mano, y el Bloc de
        # notas, `Set-Content -Encoding utf8` de PowerShell 5.1 y media Windows
        # le ponen un BOM delante. Con `utf-8` eso es un JSONDecodeError, y el
        # resultado es un núcleo sin correo, sin agenda y sin tailnet que
        # arranca igual y no se queja. Pasó el 2026-08-17.
        crudo = json.loads(fichero.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("No se pudo leer %s (%s); se sigue solo con el entorno.", fichero, e)
        return {}
    if not isinstance(crudo, dict):
        logger.warning("%s no es un objeto JSON; se ignora.", fichero)
        return {}
    return {str(c): str(v) for c, v in crudo.items()}


def cargar_configuracion() -> Configuracion:
    directorio = _directorio_datos()
    guardados = ajustes_guardados(directorio)

    def var(nombre: str, por_defecto: str) -> str:
        """El entorno primero, luego el fichero, luego lo de fábrica."""
        del_entorno = os.environ.get(nombre)
        if del_entorno is not None:
            return del_entorno
        return guardados.get(nombre, por_defecto)

    # Por defecto solo el bucle local. Para que entre el móvil se pone
    # `PERSEO_CORE_HOST=tailscale`, que añade la dirección del tailnet **sin**
    # quitar la local y sin pasar por `0.0.0.0`.
    hosts = _resolver_hosts(var("PERSEO_CORE_HOST", "127.0.0.1"))
    puerto = int(var("PERSEO_CORE_PUERTO", "8787"))

    # Por defecto se registran todos los disparadores conocidos: los que no
    # tengan de dónde tirar se retiran solos al arrancar, igual que Telegram sin
    # token. Se puede acotar la lista, o vaciarla, con `PERSEO_DISPARADORES=`.
    disparadores = tuple(
        pieza.strip()
        for pieza in var("PERSEO_DISPARADORES", "correo,agenda").split(",")
        if pieza.strip()
    )

    return Configuracion(
        hosts=hosts,
        puerto=puerto,
        token=_resolver_token(directorio),
        ruta_db=Path(var("PERSEO_CORE_DB", directorio / "estado.sqlite3")),
        directorio_datos=directorio,
        url_ollama=var("PERSEO_OLLAMA", "http://127.0.0.1:11434"),
        modelo_router=var("PERSEO_MODELO_ROUTER", "qwen3:4b"),
        telegram_token=_de_entorno_o_fichero("PERSEO_TELEGRAM_TOKEN", directorio / "telegram.txt", guardados),
        telegram_chat=_de_entorno_o_fichero("PERSEO_TELEGRAM_CHAT", directorio / "telegram_chat.txt", guardados),
        telegram_api=var("PERSEO_TELEGRAM_API", "https://api.telegram.org").rstrip("/"),
        url_base=var("PERSEO_URL_BASE", "").strip() or _url_por_defecto(hosts, puerto),
        disparadores=disparadores,
        intervalos={
            "correo": float(var("PERSEO_CORREO_INTERVALO", "300")),
            "agenda": float(var("PERSEO_AGENDA_INTERVALO", "600")),
        },
        correo_buzon=var("PERSEO_CORREO", "").strip().lower(),
        correo_falso=var("PERSEO_CORREO_FALSO", str(directorio / "buzon.json")),
        dev_motor=var("PERSEO_DEV_MOTOR", "").strip().lower(),
        dev_ejecutable=var("PERSEO_DEV_CLAUDE", "claude"),
        dev_raiz=var("PERSEO_DEV_RAIZ", str(Path.home())),
        dev_tope=float(var("PERSEO_DEV_TOPE", "900")),
        dev_tardanza_falsa=float(var("PERSEO_DEV_TARDANZA", "0")),
        google_credenciales=var("PERSEO_GOOGLE_CREDENCIALES", str(directorio / "google.json")),
        web_navegador=var("PERSEO_WEB", "").strip().lower(),
        web_tope_bytes=int(var("PERSEO_WEB_TOPE_BYTES", str(2 * 1024 * 1024))),
        web_tope_segundos=float(var("PERSEO_WEB_TOPE_SEGUNDOS", "20")),
        web_local=var("PERSEO_WEB_LOCAL", "").strip() == "1",
        agenda_origen=var("PERSEO_AGENDA", "").strip().lower(),
        agenda_falsa=var("PERSEO_AGENDA_FALSA", str(directorio / "agenda.json")),
        agenda_antelacion=int(var("PERSEO_AGENDA_ANTELACION", "60")),
        vault=var("OBSIDIAN_VAULT_PATH", str(RAIZ.parent / "obsidian_vault")),
        vault_respaldo=var("PERSEO_VAULT", "").strip().lower(),
        vault_rest_url=var("PERSEO_VAULT_REST", "https://127.0.0.1:27124").rstrip("/"),
        vault_rest_clave=_de_entorno_o_fichero("PERSEO_VAULT_CLAVE", directorio / "obsidian.txt", guardados),
        modelo_suplente=var("PERSEO_MODELO_SUPLENTE", "").strip(),
        gemini_clave=_de_entorno_o_fichero("GEMINI_API_KEY", directorio / "gemini.txt", guardados),
        tls_certificado=var("PERSEO_TLS_CERT", ""),
        tls_clave=var("PERSEO_TLS_CLAVE", ""),
        tls_puerto=int(var("PERSEO_TLS_PUERTO", str(puerto + 1))),
    )


def _de_entorno_o_fichero(variable: str, fichero: Path, guardados: dict[str, str] | None = None) -> str:
    """Lee un secreto de la variable de entorno o, si no está, de un fichero.

    El fichero vive en el directorio de datos, que está fuera de git. Es más
    cómodo que exportar la variable en cada arranque, y no deja el token del bot
    en el historial del terminal.
    """
    del_entorno = os.environ.get(variable, "").strip()
    if del_entorno:
        return del_entorno
    guardado = (guardados or {}).get(variable, "").strip()
    if guardado:
        return guardado
    if fichero.exists():
        return fichero.read_text(encoding="utf-8").strip()
    return ""


def _url_por_defecto(hosts: tuple[str, ...], puerto: int) -> str:
    """Dirección para los enlaces que salen de la máquina.

    Se prefiere una interfaz no local: el enlace lo abre el móvil desde el
    tailnet, y `127.0.0.1` allí apunta al propio teléfono.

    Y entre las no locales, **la IPv4 antes que la IPv6**. Desde que se escucha
    también en la `fd7a:…`, la primera de la lista podría ser una IPv6, y
    un enlace con una IPv6 dentro se lee fatal y encima hay que acordarse de los
    corchetes. Si solo hubiera IPv6, se pone con sus corchetes y se manda.
    """
    externos = [host for host in hosts if host not in LOCALES]
    for host in externos:
        if ":" not in host:
            return f"http://{host}:{puerto}"
    if externos:
        return f"http://[{externos[0]}]:{puerto}"
    return f"http://{hosts[0]}:{puerto}"
