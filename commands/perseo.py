"""`perseo` — encender y apagar Perseo entero desde una terminal.

Perseo son cinco cosas y hasta ahora había que saberse las cinco:

    pythonw commands/vigilante.py        el núcleo, con quien lo revive
    pythonw commands/clap_detector.py    el detector de aplausos
    RealTime\\...\\perseo.exe              la app de voz
    ollama app.exe                       sin él no hay triaje
    Obsidian.exe                         sin él la memoria falla con el plugin

**Nada arranca al encender el PC** (decisión del 2026-08-22): Perseo se abre
con `perseo on` o despertado por el detector de aplausos. Lo único externo es
la tarea `PerseoRevivir` (`manage_startup.py`), que cada diez minutos levanta
el núcleo y el detector si se han caído, sin abrir ventanas. Este comando es
para el resto de los casos: después de matar algo, después de un `git pull`, o
cuando quieres mirar si está todo en pie sin acordarte de las cinco rutas.

    perseo on         enciende lo que falte y abre la app
    perseo off        apaga Perseo entero, el detector incluido
    perseo estado     dice qué hay vivo, sin tocar nada
    perseo nucleo     solo el núcleo
    perseo parar      apaga, pero **deja el detector**: se despierta aplaudiendo
    perseo actualizar construye la app después de tocar la interfaz, y la sella
    perseo comprobar  pasa todo lo que tiene que estar verde antes de un commit
    perseo cuentas    los números que cita la documentación, medidos

`perseo` a secas sigue siendo `perseo on`, que es como se ha escrito siempre en
esta bitácora.

**Nada de esto arranca lo que ya está corriendo.** Se comprueba antes: dos
núcleos peleándose por el puerto 8787 es un `OSError 10048` que no se parece en
nada al problema real, y dos detectores escuchando el mismo micrófono responden
a la vez.

Ollama y Obsidian **se encienden pero no se apagan**. Son programas del señor
Persus, no piezas de Perseo: cerrarle Obsidian con lo que estuviera escribiendo
dentro, porque ha apagado el asistente, sería un mal negocio. `off` lo dice.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from datetime import datetime
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent

sys.path.insert(0, str(AQUI))

import manage_startup  # noqa: E402
import presencia  # noqa: E402

#: Dónde puede estar la app construida, de la más buena a la menos.
#: `release` es la de verdad; `debug` la deja `npm run tauri dev` y sirve igual
#: para abrirla, solo que arranca más despacio.
APPS = (
    RAIZ / "RealTime" / "src-tauri" / "target" / "release" / "perseo.exe",
    RAIZ / "RealTime" / "src-tauri" / "target" / "debug" / "perseo.exe",
)

#: Dónde se instala Ollama. `ollama app.exe` es la bandeja, que a su vez levanta
#: el servidor; `ollama.exe serve` es el servidor a pelo, y es el respaldo para
#: una instalación donde no esté la ventana. Lo que hace falta es que conteste el
#: 11434: la bandeja es un medio, no el fin.
OLLAMA_BANDEJA = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama app.exe"
OLLAMA_SERVIDOR = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"

#: Dónde se instala Obsidian. Si no está en ninguna, queda el URI `obsidian:`,
#: que es lo que ya usa la lista blanca del agente `pc`.
OBSIDIANES = (
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Obsidian" / "Obsidian.exe",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Obsidian" / "Obsidian.exe",
    Path(os.environ.get("PROGRAMFILES", "")) / "Obsidian" / "Obsidian.exe",
)


def _pythonw() -> str:
    """El intérprete sin consola. Con `python.exe` a secas, cada proceso abriría
    una ventana negra que además lo mata al cerrarla — que es exactamente cómo se
    murió el vigilante el 2026-08-16."""
    ejecutable = sys.executable
    candidato = ejecutable.replace("python.exe", "pythonw.exe")
    return candidato if candidato != ejecutable and os.path.isfile(candidato) else ejecutable


#: Escapar del *job* de quien nos lanzó. Python no lo expone con nombre.
CREATE_BREAKAWAY_FROM_JOB = 0x01000000

#: Que un proceso auxiliar NO abra ventana. Se le pone a todo lo que consulta
#: algo del sistema —«¿está vivo el núcleo?», «¿qué IP tiene el tailnet?»— y a
#: todo lo que mata procesos, porque nadie lee esa salida: va a `capture_output`.
#:
#: Sin esto, la tarea `PerseoRevivir` hacía parpadear **dos** consolas de
#: PowerShell cada diez minutos encima de lo que estuvieras haciendo —una por
#: `arrancar_nucleo` y otra por `arrancar_detector`, las dos preguntando lo
#: mismo— y desde fuera parecía que algo iba mal. Un proceso de fondo que
#: se ve trabajar es un proceso de fondo mal hecho.
SIN_VENTANA = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def _sin_consola(argumentos: list[str]) -> None:
    """Lanza algo y se desentiende: ni consola, ni esperar, ni morir con esta.

    **`DETACHED_PROCESS` no basta en Windows**, y esto costó tres días de Perseo
    apagado. Las terminales modernas y los agentes meten lo que ejecutan
    en un *job object* con `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`: cuando esa
    sesión termina, **Windows mata a todos los descendientes**, estén detached o
    no, sin avisar y sin código que lo explique. Medido en esta máquina el
    2026-08-21: la terminal corría dentro de un job con esa bandera puesta.

    Se pide `CREATE_BREAKAWAY_FROM_JOB`, que es lo que existe para salirse. **No
    siempre sirve**, y aquí se comprobó que no basta: los hijos acaban en un job
    igualmente, así que un Perseo lanzado desde una terminal puede seguir
    muriéndose con ella. Se pide de todas formas porque es gratis y en una
    terminal normal sí funciona; lo que de verdad garantiza que Perseo vuelva es
    la tarea `PerseoRevivir` (`manage_startup.py`), que mira cada diez minutos
    desde fuera de cualquier sesión.

    Si el job no admite el intento, `Popen` falla con un `OSError` y se lanza
    como antes: arrancar y quizá morir con la terminal es mejor que no arrancar.
    """
    if os.name != "nt":
        subprocess.Popen(argumentos, cwd=str(RAIZ), close_fds=True)
        return

    banderas = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
    try:
        subprocess.Popen(
            argumentos,
            cwd=str(RAIZ),
            creationflags=banderas | CREATE_BREAKAWAY_FROM_JOB,
            close_fds=True,
        )
    except OSError:
        subprocess.Popen(argumentos, cwd=str(RAIZ), creationflags=banderas, close_fds=True)


def _corriendo(fragmento: str) -> bool:
    """Si hay algún proceso de Python cuya línea de comandos lleve eso dentro.

    Se pregunta a Windows por WMI en vez de mirar un fichero de PID porque el
    detector no deja ninguno. Si la consulta falla, se contesta que **no** está:
    el peor caso es arrancar algo dos veces, y eso se nota; dar por vivo algo
    muerto deja a Perseo apagado sin que nadie lo sepa.
    """
    if os.name != "nt":
        return False
    try:
        salida = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" "
                "| Select-Object -ExpandProperty CommandLine",
            ],
            capture_output=True,
            creationflags=SIN_VENTANA,
            text=True,
            # PowerShell contesta en el código de página de la consola, no en
            # UTF-8: sin este colchón, una línea de comandos con tilde mata al
            # hilo lector y «¿está corriendo?» se queda sin respuesta.
            errors="replace",
            timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return any(fragmento in linea for linea in salida.splitlines())


def _ollama() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _obsidian() -> bool:
    """Si el plugin de Obsidian contesta. Sin clave da 401, que también vale:
    lo que se quiere saber es si Obsidian está abierto."""
    import ssl

    contexto = ssl.create_default_context()
    contexto.check_hostname = False
    contexto.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(
            "https://127.0.0.1:27124/", timeout=3, context=contexto
        ) as r:
            return r.status in (200, 401)
    except (urllib.error.URLError, OSError):
        return False


def _app() -> Path | None:
    for ruta in APPS:
        if ruta.is_file():
            return ruta
    return None


def _exe_vivo(nombre: str) -> bool:
    """Si hay algún proceso con ese nombre de ejecutable.

    Es el hermano de `_corriendo`, que solo sabe de procesos de Python: Obsidian
    no deja PID en ningún sitio y su plugin puede estar apagado, así que la
    pregunta «¿está abierto?» no se puede hacer por HTTP.
    """
    if os.name != "nt":
        return False
    try:
        salida = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {nombre}", "/NH"],
            capture_output=True,
            creationflags=SIN_VENTANA,
            text=True,
            # tasklist escribe en el código de página OEM (cp850 en un Windows
            # español) y con `-X utf8` Python intenta leerlo como UTF-8: el
            # primer byte raro reventaba la lectura entera. `estado` tiene que
            # contestar SIEMPRE, aunque tasklist tenga un mal día; los nombres
            # de ejecutable son ASCII y sobreviven a cualquier recodificación.
            errors="replace",
            timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        # Mismo criterio que `_corriendo`: si no se sabe, se contesta que no.
        # Abrir dos veces algo que ya está abierto se nota; darlo por vivo
        # cuando está muerto deja a Perseo capado sin que nadie lo sepa.
        return False
    return nombre.lower() in salida.lower()


def _primera_que_exista(rutas: tuple[Path, ...]) -> Path | None:
    for ruta in rutas:
        if ruta.is_file():
            return ruta
    return None


# --------------------------------------------------------------------------- #
# Lo que hace cada orden
# --------------------------------------------------------------------------- #


def arrancar_nucleo() -> bool:
    """Levanta el vigilante, que a su vez levanta el núcleo. Devuelve si hizo algo."""
    if manage_startup.nucleo_responde():
        print("  [ya estaba]  El núcleo responde en", manage_startup.url_salud())
        return False

    print("  [arrancando] El núcleo, con su vigilante")
    _sin_consola([_pythonw(), str(AQUI / "vigilante.py")])

    # El vigilante espera un momento antes del primer intento, así que no se
    # pregunta de inmediato: se pregunta hasta que conteste o se acabe la
    # paciencia. Diez segundos son de sobra en frío.
    for _ in range(20):
        time.sleep(0.5)
        if manage_startup.nucleo_responde(espera=1.0):
            print("  [listo]      El núcleo responde")
            return True
    print("  [ojo]        El núcleo no ha contestado todavía; mira <datos>/vigilante.log")
    return True


def arrancar_detector() -> bool:
    if _corriendo("clap_detector.py"):
        print("  [ya estaba]  El detector de aplausos")
        return False
    print("  [arrancando] El detector de aplausos")
    _sin_consola([_pythonw(), str(AQUI / "clap_detector.py")])
    return True


def arrancar_ollama() -> bool:
    """Levanta Ollama y espera a que conteste. Devuelve si hizo algo.

    Se espera al 11434 y no a que aparezca la ventana: el triaje llama por HTTP,
    y una bandeja abierta con el servidor todavía cargando falla igual que si no
    estuviera. Son unos segundos en frío.
    """
    if _ollama():
        print("  [ya estaba]  Ollama responde en el 11434")
        return False

    binario = OLLAMA_BANDEJA if OLLAMA_BANDEJA.is_file() else None
    argumentos: list[str] | None = None
    if binario is not None:
        argumentos = [str(binario)]
    elif OLLAMA_SERVIDOR.is_file():
        argumentos = [str(OLLAMA_SERVIDOR), "serve"]

    if argumentos is None:
        print("  [falta]      Ollama no está instalado donde se le busca:")
        print(f"               {OLLAMA_BANDEJA.parent}")
        return False

    print("  [arrancando] Ollama")
    _sin_consola(argumentos)

    for _ in range(30):
        time.sleep(0.5)
        if _ollama():
            print("  [listo]      Ollama responde")
            return True
    print("  [ojo]        Ollama no contesta todavía en el 11434; sin él no hay triaje")
    return True


def arrancar_obsidian() -> bool:
    """Abre Obsidian, con el vault que se quedara abierto. Devuelve si hizo algo.

    Se mira **el proceso**, no el plugin: `_obsidian()` pregunta al Local REST
    API, que puede estar apagado en un Obsidian perfectamente abierto. Preguntar
    por ahí abriría una segunda ventana cada vez que el plugin estuviera caído.
    """
    if _exe_vivo("Obsidian.exe"):
        print("  [ya estaba]  Obsidian")
        return False

    binario = _primera_que_exista(OBSIDIANES)
    if binario is not None:
        print("  [arrancando] Obsidian")
        _sin_consola([str(binario)])
        return True

    # Sin ruta conocida queda el URI, que es lo que ya usa la lista blanca del
    # agente `pc`. Abre el mismo programa por el registro de Windows.
    print("  [arrancando] Obsidian (por el enlace obsidian:)")
    webbrowser.open("obsidian://")
    return True


#: Fichero que Rust vigila para sacar la ventana del escondite. Es distinto del
#: de la palabra clave (`.perseo-autollamada`), que además entra en llamada.
MARCADOR_MOSTRAR = RAIZ / ".perseo-mostrar"


def arrancar_app() -> bool:
    if presencia.app_viva():
        # Cerrar la ventana no cierra Perseo: la esconde en la bandeja. Así que
        # "ya estaba" era verdad y a la vez inútil — el usuario escribía `perseo`
        # y no pasaba nada en pantalla, que desde fuera se ve igual que una app
        # rota. Se pide que vuelva.
        try:
            MARCADOR_MOSTRAR.write_text("", encoding="utf-8")
            print("  [al frente]  La app de voz ya estaba; se saca de la bandeja")
        except OSError:
            print("  [ya estaba]  La app de voz (escondida en la bandeja)")
        return False

    binario = _app()
    if binario is None:
        print("  [falta]      La app no está construida. Constrúyela una vez:")
        print("               cd RealTime && npm install && npm run tauri build")
        return False

    if app_caducada(binario):
        # El binario lleva el `dist` dentro: si es más viejo que las fuentes, lo
        # que se va a abrir NO es lo último que se tocó. Se abre igual —vale más
        # una versión vieja que nada— pero se dice, y se dice fuerte.
        print("  [OJO]        Esta app es más vieja que la interfaz que hay en el código.")
        print("               Vas a abrir la construcción anterior. Para verla al día:")
        print("               perseo actualizar")

    print(f"  [arrancando] La app de voz ({binario.parent.name})")
    _sin_consola([str(binario)])
    return True


# ── Construir la app: lo que evita abrir una versión vieja ───────────────────
#
# La app de escritorio **no lee `RealTime/src`**: abre un binario que lleva el
# `dist` dentro, incrustado cuando se construyó. Así que tocar la interfaz y
# lanzar `perseo on` abre exactamente lo mismo de antes, sin un solo error que
# lo explique. Ha pasado varias veces y siempre se descubre mirando la pantalla
# y discutiendo si el cambio se hizo o no.
#
# De aquí salen las tres cosas que lo cierran:
#   1. `perseo actualizar`, que construye y sella.
#   2. Un aviso en `perseo on` y en `perseo estado` cuando el binario es más
#      viejo que las fuentes.
#   3. Una marca de construcción que se ve en las dos interfaces —la de la app y
#      la del móvil—, para que «qué versión estoy viendo» se conteste mirando.

#: Lo que, al cambiar, obliga a volver a construir. El móvil no está aquí a
#: propósito: `perseo_core/interfaz/index.html` lo sirve el núcleo tal cual está
#: en el disco, y por eso el móvil siempre va al día y la app no.
FUENTES_APP = (
    RAIZ / "RealTime" / "src",
    RAIZ / "RealTime" / "public",
    RAIZ / "RealTime" / "index.html",
    RAIZ / "RealTime" / "vite.config.ts",
    RAIZ / "RealTime" / "package.json",
    RAIZ / "RealTime" / "src-tauri" / "src",
    RAIZ / "RealTime" / "src-tauri" / "Cargo.toml",
    RAIZ / "RealTime" / "src-tauri" / "tauri.conf.json",
)


def _directorio_datos() -> Path:
    """El mismo directorio que usa el núcleo, con la misma variable de entorno.

    Se repite aquí en vez de importar `perseo_core` porque este comando tiene
    que funcionar sin las dependencias del núcleo instaladas.
    """
    return Path(os.environ.get("PERSEO_CORE_DATOS", RAIZ / "perseo_core" / "datos"))


def _ficheros_de_fuentes() -> list[Path]:
    """Todos los ficheros que entran en la construcción, en orden estable."""
    encontrados: list[Path] = []
    for ruta in FUENTES_APP:
        if ruta.is_file():
            encontrados.append(ruta)
        elif ruta.is_dir():
            encontrados.extend(h for h in ruta.rglob("*") if h.is_file())
    return sorted(encontrados)


def huella_de_fuentes() -> str:
    """Un resumen del **contenido** de la interfaz, no de sus fechas.

    Se compara con el que guardó la última construcción, y así «¿está la app al
    día?» se contesta sin depender de relojes. Por fechas no valía: la
    construcción tarda dos minutos y medio, así que un binario recién hecho
    parece más nuevo que un fichero tocado *mientras* se construía —que es justo
    el caso que hay que cazar—, y un `git checkout` reescribe fechas sin cambiar
    una línea.
    """
    resumen = hashlib.sha1()
    for fichero in _ficheros_de_fuentes():
        resumen.update(str(fichero.relative_to(RAIZ)).replace("\\", "/").encode("utf-8"))
        resumen.update(fichero.read_bytes())
    return resumen.hexdigest()[:12]


def _sello_guardado() -> dict:
    """Lo que dejó la última construcción, o vacío si no hay ninguna."""
    try:
        datos = json.loads((_directorio_datos() / "version.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return datos if isinstance(datos, dict) else {}


def app_caducada(binario: Path | None = None) -> bool:
    """¿Hay cambios en la interfaz que ese binario todavía no lleva dentro?

    Sin sello no se puede saber —es una construcción de antes de que esto
    existiera— y se contesta que no: un aviso que sale siempre se deja de leer.
    """
    binario = binario or _app()
    if binario is None:
        return False
    guardada = _sello_guardado().get("huella")
    if not guardada:
        return False
    return guardada != huella_de_fuentes()


def _marca_de_construccion() -> str:
    """`AAAAMMDD-HHMM` y, si se puede, la revisión de git.

    La fecha va primero porque es lo que se compara de un vistazo con «lo he
    tocado hace un minuto»; el `sha` está para poder volver al código exacto.
    """
    sello = time.strftime("%Y%m%d-%H%M")
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(RAIZ),
            capture_output=True,
            creationflags=SIN_VENTANA,
            text=True,
            errors="replace",
            timeout=10,
        )
        if revision.returncode == 0 and revision.stdout.strip():
            return f"{sello}+{revision.stdout.strip()}"
    except (OSError, subprocess.SubprocessError):
        pass
    return sello


def _sellar(marca: str, huella: str) -> Path:
    """Deja la marca donde el núcleo pueda servírsela al móvil.

    Es el mismo valor que Vite acaba de incrustar en el binario, y ahí está la
    gracia: la app enseña el suyo, el móvil enseña este, y si no coinciden es
    que una de las dos pantallas se quedó en una versión vieja.
    """
    directorio = _directorio_datos()
    directorio.mkdir(parents=True, exist_ok=True)
    fichero = directorio / "version.json"
    fichero.write_text(
        json.dumps(
            {
                "marca": marca,
                "construido": datetime.now().astimezone().replace(microsecond=0).isoformat(),
                # La huella es de cuando **empezó** la construcción: si algo se
                # tocó mientras compilaba, el binario no lo lleva y esto lo dice.
                "huella": huella,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return fichero


def _limpiar_cache_webview() -> None:
    """Vacía la caché HTTP de WebView2 tras una construcción.

    La interfaz va incrustada en el binario, pero WebView2 la sirve a través de
    su propia caché HTTP (`EBWebView\\Default\\Cache`), y esa caché no distingue
    un binario nuevo de uno viejo: tras actualizar, la ventana puede enseñar el
    bundle ANTERIOR sin un solo error que lo delate. Pasó el 2026-08-24 — la
    pestaña Proyectos reconstruida seguía mostrando la versión anterior con el
    exe nuevo ya corriendo. Borrarla es barato y no toca los ajustes: esos
    viven en Local Storage, que aquí no se mira.
    """
    raiz = (
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "com.perseo.app"
        / "EBWebView"
        / "Default"
    )
    borradas: list[str] = []
    for nombre in ("Cache", "Code Cache"):
        carpeta = raiz / nombre
        try:
            if carpeta.is_dir():
                shutil.rmtree(carpeta)
                borradas.append(nombre)
        except OSError:
            # Con la app viva los ficheros están bloqueados; no pasa nada,
            # porque en el flujo normal `actualizar` la cerró al empezar.
            print("  [aviso]      No se pudo vaciar la caché de WebView2; "
                  "si la ventana enseña la interfaz vieja, ciérra del todo y ábrela otra vez")
    if borradas:
        print("  [caché]      Vaciada la caché de WebView2 "
              f"({', '.join(borradas)}): la ventana nace del binario nuevo")


def actualizar() -> None:
    """`perseo actualizar`: construir la app y sellar las dos interfaces.

    Tarda un par de minutos y hace falta cada vez que se toca `RealTime`. Cierra
    la app primero porque Windows no deja sobrescribir un `.exe` en marcha: sin
    eso, la construcción falla al enlazar y el fallo no dice de qué va.
    """
    print("Actualizando Perseo:\n")

    marca = _marca_de_construccion()
    # Se toma antes de construir, no después: lo que va a quedar dentro del
    # binario es el código que hay ahora mismo.
    huella = huella_de_fuentes()
    estaba_abierta = presencia.app_viva()
    if estaba_abierta:
        # Con la app abierta el enlazador no puede escribir el .exe, así que se
        # cierra y se vuelve a abrir al final. La llamada en curso se pierde, y
        # es el precio de construir.
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Stop-Process -Name perseo -Force -ErrorAction SilentlyContinue",
            ],
            capture_output=True,
            creationflags=SIN_VENTANA,
        )
        print("  [cerrada]    La app de voz, para poder sobrescribirla")

    print(f"  [marca]      {marca}")
    print("  [construye]  npm run tauri build — esto tarda un par de minutos\n")

    entorno = {**os.environ, "PERSEO_BUILD": marca}
    npm = "npm.cmd" if os.name == "nt" else "npm"
    try:
        construccion = subprocess.run(
            [npm, "run", "tauri", "build", "--", "--no-bundle"],
            cwd=str(RAIZ / "RealTime"),
            env=entorno,
        )
    except OSError as error:
        print(f"\n  [ERROR]      No se pudo lanzar npm: {error}")
        print("               ¿Está Node instalado y en el PATH?")
        return

    if construccion.returncode != 0:
        print("\n  [ERROR]      La construcción falló; se deja el binario anterior.")
        print("               Lee el error de arriba: casi siempre es TypeScript.")
        if estaba_abierta:
            arrancar_app()
        return

    fichero = _sellar(marca, huella)
    print(f"\n  [sellado]    {fichero}")
    print("  [listo]      La app y el móvil enseñan ya la misma marca.")

    _limpiar_cache_webview()

    if estaba_abierta:
        arrancar_app()
    else:
        print("               Ábrela con: perseo on")


def estado() -> None:
    manage_startup.estado()
    print()
    print("Las dependencias, que `perseo on` enciende:")
    # De Obsidian se dicen las dos cosas: si el programa está abierto y si su
    # plugin contesta. Un Obsidian abierto con el plugin apagado deja la memoria
    # igual de rota que un Obsidian cerrado, y son dos arreglos distintos.
    obsidian_abierto = _exe_vivo("Obsidian.exe")
    obsidian_plugin = _obsidian()
    for nombre, vivo, sin_el in (
        ("Ollama", _ollama(), "sin él no hay triaje de correo"),
        (
            "Obsidian",
            obsidian_abierto,
            "sin él la memoria falla si el vault va por el plugin"
            if obsidian_plugin or not obsidian_abierto
            else "abierto, pero su Local REST API no contesta: mira el plugin",
        ),
    ):
        marca = "[activo]  " if vivo else "[PARADO]  "
        print(f"  {marca}   {nombre} — {sin_el}")

    print()
    detector = "[activo]  " if _corriendo("clap_detector.py") else "[PARADO]  "
    app = "[activo]  " if presencia.app_viva() else "[PARADO]  "
    print(f"  {detector}   Detector de aplausos")
    print(f"  {app}   App de voz")

    binario = _app()
    if binario is None:
        print("  [FALTA]      La app no está construida: perseo actualizar")
    elif app_caducada(binario):
        print("  [VIEJA]      La app construida no lleva los últimos cambios de la")
        print("               interfaz. Para ponerla al día: perseo actualizar")


def parar(avisar_del_detector: bool = True) -> None:
    """Cierra la app y el núcleo. **Al vigilante primero**, o resucita el núcleo.

    No toca el detector de aplausos: es lo que despierta a Perseo, y pararlo sin
    querer deja el sistema mudo de una forma que no se nota hasta que aplaudes.
    Quien sí lo mata es `apagar`, y por eso puede callar ese aviso.
    """
    if os.name != "nt":
        print("Esto solo sabe parar procesos en Windows.")
        return

    for descripcion, patron in (
        ("el vigilante", "vigilante.py"),
        ("el núcleo", "-m perseo_core"),
    ):
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" "
                f"| Where-Object {{ $_.CommandLine -like '*{patron}*' }} "
                "| ForEach-Object { Stop-Process -Id $_.ProcessId -Force }",
            ],
            capture_output=True,
            creationflags=SIN_VENTANA,
        )
        print(f"  [parado]     {descripcion}")

    subprocess.run(
        ["powershell", "-NoProfile", "-Command", "Stop-Process -Name perseo -Force -ErrorAction SilentlyContinue"],
        capture_output=True,
        creationflags=SIN_VENTANA,
    )
    print("  [parado]     La app de voz")
    if avisar_del_detector:
        print("\n  El detector de aplausos sigue en pie: es lo que despierta a Perseo.")


def parar_detector() -> None:
    """Mata el detector de aplausos. Lo que separa `off` de `parar`."""
    if os.name != "nt":
        print("Esto solo sabe parar procesos en Windows.")
        return
    subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" "
            "| Where-Object { $_.CommandLine -like '*clap_detector.py*' } "
            "| ForEach-Object { Stop-Process -Id $_.ProcessId -Force }",
        ],
        capture_output=True,
        creationflags=SIN_VENTANA,
    )
    print("  [parado]     El detector de aplausos")


def apagar() -> None:
    """`perseo off`: Perseo entero, el detector incluido.

    Es lo que `parar` no hace, y la diferencia importa: con el detector vivo,
    Perseo sigue escuchando el micrófono y dos palmadas lo encienden otra vez.
    Eso está bien para reiniciar el núcleo y mal para apagarlo de verdad.

    Ollama y Obsidian se quedan abiertos: son programas del señor Persus, no
    piezas de Perseo.
    """
    print("Apagando Perseo:\n")
    parar(avisar_del_detector=False)
    parar_detector()
    print("\n  Ollama y Obsidian siguen abiertos: no son de Perseo.")
    print("  Para volver: perseo on")


def todo() -> None:
    """`perseo on`: las cinco piezas, y las dependencias antes que nada.

    Ollama y Obsidian van primero **a propósito**: el núcleo arranca igual sin
    ellos, pero el primer triaje de correo y la primera nota que se guarde caen
    en el hueco. Ollama, además, tarda en contestar, así que lo que se gana es
    ese arranque mientras suben el núcleo y la app.
    """
    print("Encendiendo Perseo:\n")
    arrancar_ollama()
    arrancar_obsidian()
    arrancar_nucleo()
    arrancar_detector()
    arrancar_app()
    print("\n  Panel y cola: botón de cuadrícula en la app, o http://127.0.0.1:8787")


def _comprobar() -> None:
    """Las comprobaciones de antes de un commit. Viven en su propio módulo.

    El importe es perezoso por lo mismo que el resto de este fichero es rápido:
    `perseo on` se escribe cien veces más que `perseo comprobar`, y no tiene por
    qué pagar el análisis del árbol de importaciones.
    """
    import comprobar as modulo

    raise SystemExit(modulo.comprobar(sys.argv[2:]))


def _cuentas() -> None:
    import comprobar as modulo

    raise SystemExit(modulo.imprimir_cuentas(sys.argv[2:]))


#: Las órdenes, con sus sinónimos. `on` y `off` son las que pidió el señor
#: Persus el 2026-08-21; `perseo` a secas se queda como `on` porque es lo que
#: dice la bitácora entera, y `parar` porque apagar dejando el detector vivo
#: sigue siendo útil para reiniciar el núcleo sin quedarse sordo.
ORDENES = {
    "": todo,
    "on": todo,
    "encender": todo,
    "off": apagar,
    "apagar": apagar,
    "estado": estado,
    "nucleo": lambda: arrancar_nucleo(),
    "núcleo": lambda: arrancar_nucleo(),
    "parar": parar,
    "actualizar": actualizar,
    "construir": actualizar,
    "comprobar": _comprobar,
    "cuentas": _cuentas,
}


def main() -> None:
    orden = (sys.argv[1] if len(sys.argv) > 1 else "").strip().lower()
    if orden in ("-h", "--help", "ayuda"):
        print(__doc__)
        return
    hacer = ORDENES.get(orden)
    if hacer is None:
        print(f"No sé qué es «{orden}». Órdenes: {', '.join(o for o in ORDENES if o)}")
        sys.exit(2)
    hacer()


if __name__ == "__main__":
    main()
