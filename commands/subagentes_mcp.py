"""Servidor MCP de subagentes: Perseo reparte trabajo entre CLIs de agente.

La pieza que convierte al modo live en un jefe de equipo: `encargar_tarea`
arranca un agente de código (opencode por defecto) en un directorio y devuelve
al momento un identificador; `consultar_tarea` y `listar_tareas` siguen su
curso. Varios encargos corren A LA VEZ — cada uno es un hilo con su proceso,
y el canal JSON-RPC solo viaja respuestas cortas.

Motores, en orden de preferencia salvo que `PERSEO_SUBAGENTE_MOTOR` diga otra
cosa: `opencode run` (con el modelo de `PERSEO_SUBAGENTE_MODELO`) y, si no está
el ejecutable, `claude -p`. Los dos se hablan sin interactivo y aceptan ediciones
— son subagentes de trabajo, no consultores.

LA LLAMADA DE AVISO Y EL TELEGRAM
---------------------------------
Cuando un encargo termina y NADIE ha preguntado por él pasado un plazo de gracia,
pasan dos cosas:

1. Se deja el marcador `.perseo-autollamada` en la raíz del repo con el motivo
   dentro: la app lo vigila, saca la ventana, entra en llamada y Perseo cuenta
   qué acabó ("estoy viendo una serie y el agente de mi web terminó").
2. Se manda el mismo aviso POR TELEGRAM, leyendo el token y el chat de los
   ficheros de siempre en `datos/`: llega al móvil aunque la app esté apagada.

Si alguien consultó el resultado, no hay ni marcador ni Telegram: ya lo sabe
quien preguntó.

Seguridad, la de siempre: los directorios permitidos están escritos aquí abajo
(persona delante de la máquina); nada de shell; lo que llega del modelo viaja
como argumento del CLI y como directorio validado, nunca interpolado en una
línea de intérprete.

    Perseo Core lo lanza desde <datos>/mcp.json:
    ["python", "-X", "utf8", "...\\commands\\subagentes_mcp.py"]
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

VERSION_PROTOCOLO = "2024-11-05"

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "perseo_core" / "datos"

#: Directorios donde un subagente puede trabajar. Escritos por una persona;
#: ampliarlos es decisión suya, igual que la lista blanca de `pc.py`.
RAICES_PERMITIDAS = tuple(
    Path(r).resolve()
    for r in os.environ.get("PERSEO_SUBAGENTE_RAICES", str(Path.home())).split(";")
    if r.strip()
)

#: Cuánto corre un encargo como mucho. Una tarea de código puede irse media
#: hora; más que esto, mejor partirla.
TOPE_SEGUNDOS = float(os.environ.get("PERSEO_SUBAGENTE_TOPE", "1800"))

#: Minutos de gracia antes de llamar por un resultado que nadie reclamó. Si el
#: señor Persus está en llamada, el modelo consulta en cuanto acaba; el aviso
#: solo sale cuando de verdad nadie está mirando. Ajustable para pruebas.
GRACIA_SEGUNDOS = float(os.environ.get("PERSEO_SUBAGENTE_GRACIA", "120"))

#: Cuánto texto conserva cada tarea para devolverlo.
TOPE_SALIDA = 4000

MARCA_AUTOLLAMADA = RAIZ / ".perseo-autollamada"


# --------------------------------------------------------------------------- #
# Motores
# --------------------------------------------------------------------------- #


#: Los modelos GRATIS de opencode Zen, de más contexto a menos. Salen de
#: `opencode models opencode --verbose` filtrando los que tienen `cost.input` y
#: `cost.output` a cero — no de lo que suene conocido. Los seis saben usar
#: herramientas y razonar, que es lo que necesita un agente de código.
#:
#: Es la MISMA lista que ofrece la pestaña de encargos del panel
#: (`RealTime/src/components/Panel.tsx`); si cambia una, cambia la otra.
MODELOS_GRATIS = (
    "opencode/nemotron-3-ultra-free",
    "opencode/nemotron-3.5-lightning-free",
    "opencode/hy3-free",
    "opencode/big-pickle",
    "opencode/mimo-v2.5-free",
    # Pide ser contribuidor de opencode: puede no estar disponible en esta
    # cuenta, y por eso va el último pese a su millón de contexto.
    "opencode/muse-spark-1.2-contributor-free",
)

#: Con qué modelo trabaja opencode si nadie dice otra cosa. Explícito y
#: gratuito a propósito: sin `-m`, opencode usa el que tenga configurado, que
#: puede ser de pago — y el señor Persus pidió el 2026-08-26 la opción que no
#: gasta suscripción, no la que se la gasta por defecto.
MODELO_POR_DEFECTO = MODELOS_GRATIS[0]


def _motor(pedido: str = "") -> str:
    """Qué CLI trabaja. Manda lo que pida el encargo; luego el entorno; luego
    el que esté instalado.

    Por defecto manda **opencode**, que es el que no gasta suscripción: el
    señor Persus lo pidió con todas las letras el 2026-08-26 —«usar Claude es
    secundario, quiero la opción gratuita»—. La razón por la que antes mandaba
    Claude (el endpoint gratuito se cae a ratos y sale con código 0 sin haber
    hecho nada) ya no exige cambiar de motor: `_fracaso_encubierto` lo detecta
    y `_ejecutar` reintenta. Claude sigue a un `motor='claude'` de distancia.
    """
    pedido = (pedido or os.environ.get("PERSEO_SUBAGENTE_MOTOR", "")).strip().lower()
    if pedido == "claude":
        orden = ("claude", "opencode")
    elif pedido:
        orden = (pedido, "opencode", "claude")
    else:
        orden = ("opencode", "claude")
    for nombre in orden:
        if shutil.which(nombre):
            return nombre
    raise RuntimeError(
        "No hay ningún motor de subagentes instalado. Instala opencode "
        "(npm i -g opencode-ai) o Claude Code."
    )


#: Lo que un subagente no ejecuta ni aunque se lo pidan. Copia deliberada de
#: `perseo_core/dev.py`: los dos reparten trabajo a un CLI de agente, y el cerco
#: tiene que ser el mismo se entre por donde se entre.
DENEGADAS = (
    "Bash(git push*)",
    "Bash(git reset --hard*)",
    "Bash(git clean*)",
    "Bash(rm *)",
    "Bash(rmdir *)",
    "Bash(del *)",
    "Bash(format*)",
)


def _modelo_de(motor: str, pedido: str = "") -> str:
    """Con qué modelo trabaja este encargo. Vacío para Claude, que usa el suyo.

    Para opencode NUNCA se devuelve vacío: sin `-m` trabaja con el modelo que
    tenga configurado, y ese puede ser de pago.
    """
    if motor != "opencode":
        return (pedido or os.environ.get("PERSEO_SUBAGENTE_MODELO", "")).strip()
    elegido = (pedido or os.environ.get("PERSEO_SUBAGENTE_MODELO", "")).strip()
    if not elegido:
        return MODELO_POR_DEFECTO
    # Un nombre corto («hy3-free») se completa con el proveedor: es lo que
    # escribe un modelo de voz cuando le dictan el nombre a medias.
    return elegido if "/" in elegido else f"opencode/{elegido}"


def _comando(motor: str, tarea: str, directorio: Path | None = None, modelo: str = "") -> list[str]:
    """La línea completa del encargo, como lista de argumentos y sin shell."""
    if motor == "opencode":
        # `--auto` es obligatorio aquí: sin él, `opencode run` PIDE permiso para
        # escribir fuera del proyecto, nadie contesta —esto no es interactivo— y
        # el propio programa se lo deniega («auto-rejecting») y sale con código
        # 0. El encargo se daba por hecho con el disco intacto.
        comando = [shutil.which("opencode") or "opencode", "run", "--auto"]
        # `--dir` NO es redundante con el `cwd` del proceso, y en el agente
        # `dev` costó un fichero escrito dos carpetas más arriba para verlo
        # (2026-08-26): `opencode run` levanta su propio servidor y resuelve el
        # proyecto por su cuenta, así que hereda el `cwd` y luego lo ignora.
        # La raíz hay que decírsela, no dársela por supuesta.
        if directorio is not None:
            comando += ["--dir", str(directorio)]
        comando += ["-m", _modelo_de(motor, modelo)]
        return [*comando, tarea]
    linea = [
        shutil.which("claude") or "claude",
        "-p",
        tarea,
        "--output-format",
        "text",
        # `acceptEdits` acepta escribir ficheros y NADA MÁS: cualquier comando
        # se queda esperando una aprobación que aquí no puede dar nadie, y el
        # subagente contesta «necesita tu aprobación» y se va. Comprobado el
        # 2026-08-24 pidiéndole que ejecutara `python -c "print(6*7)"`. Un
        # subagente que no puede correr nada no sirve para lo que se le pide
        # —instalar, probar, construir—, así que trabaja sin pedir permiso.
        "--permission-mode",
        "bypassPermissions",
        # Lo que no hace ni con eso. Es la misma lista del agente `dev`, y por
        # la misma razón: publicar es del usuario y borrar no tiene vuelta
        # atrás. La lista de denegados manda sobre el modo de permisos.
        "--disallowed-tools",
        *DENEGADAS,
        "--max-turns",
        "40",
    ]
    # Claude sí sabe salir de su carpeta de arranque, pero decírsela es gratis
    # y quita ambigüedad cuando el encargo nombra rutas relativas.
    if directorio is not None:
        linea += ["--add-dir", str(directorio)]
    elegido = _modelo_de("claude", modelo)
    if elegido:
        linea += ["--model", elegido]
    return linea


# --------------------------------------------------------------------------- #
# Las tareas
# --------------------------------------------------------------------------- #

_tareas: dict[str, dict] = {}
_cerrojo = threading.Lock()
_contador = 0

#: El seguimiento, persistido: al reiniciar Perseo los encargos siguen ahí
#: (los que estaban en marcha pasan a 'perdido', porque nadie puede saber su
#: final). Sin esto, «¿y el encargo de antes?» era un misterio tras cada
#: reinicio — y una espiral de errores para Perseo.
RUTA_ESTADO = DATOS / "subagentes_estado.json"

_CLAVES_PERSISTIDAS = ("estado", "salida", "error", "motor", "modelo", "entregado")


def _persistir() -> None:
    try:
        with _cerrojo:
            datos = {
                i: {k: t.get(k) for k in _CLAVES_PERSISTIDAS} for i, t in _tareas.items()
            }
        RUTA_ESTADO.write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")
    except (OSError, TypeError):
        pass


def _cargar() -> None:
    global _contador
    try:
        crudo = json.loads(RUTA_ESTADO.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(crudo, dict):
        return
    with _cerrojo:
        for id_tarea, entrada in crudo.items():
            if not isinstance(entrada, dict):
                continue
            if entrada.get("estado") == "en_curso":
                # El proceso murió con él en marcha: el resultado no está.
                entrada["estado"] = "perdido"
                entrada["salida"] = ""
                entrada["error"] = (
                    "Perseo se apagó mientras este encargo trabajaba; su "
                    "resultado se perdió. Vuelve a lanzarlo si hacía falta."
                )
            _tareas[str(id_tarea)] = {
                "estado": str(entrada.get("estado") or "perdido"),
                "salida": str(entrada.get("salida") or ""),
                "error": str(entrada.get("error") or ""),
                "motor": str(entrada.get("motor") or "?"),
                "modelo": str(entrada.get("modelo") or ""),
                # Lo cargado del disco NUNCA avisa: su ventana de gracia murió
                # con el servidor anterior, y un aviso por algo de hace horas
                # es ruido, no información (H-62).
                "entregado": True,
                "inicio": 0.0,
                "fin": None,
            }
        for id_tarea in _tareas:
            try:
                numero = int(id_tarea.lstrip("s"))
                _contador = max(_contador, numero)
            except ValueError:
                pass


#: Colores y negritas del CLI: en la salida van bien, de viva voz no se leen.
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _recortar(texto: str, tope: int = TOPE_SALIDA) -> str:
    limpio = _ANSI.sub("", (texto or "")).strip()
    return limpio if len(limpio) <= tope else limpio[: tope - 1].rstrip() + "…"


def _primera_linea(texto: str, tope: int = 160) -> str:
    linea = next((x.strip() for x in (texto or "").splitlines() if x.strip()), "")
    return linea[:tope]


def _quien(tarea: dict) -> str:
    """Con qué trabajó un encargo: el motor y, si lo hubo, el modelo.

    Perseo lo cuenta de viva voz, y «opencode» a secas no distingue el modelo
    gratuito del de pago — que es justo lo que el señor Persus quiere saber.
    """
    modelo = str(tarea.get("modelo") or "").split("/")[-1]
    return f"{tarea.get('motor', '?')} · {modelo}" if modelo else str(tarea.get("motor", "?"))


def _validar_directorio(directorio: str) -> Path:
    """El encargo trabaja aquí dentro, o no trabaja.

    El error es un manual en miniatura, no un muro: las tres veces seguidas
    que el modelo pasó como directorio la CARPETA A CREAR (que por definición
    aún no existe), el mensaje genérico «no existe» no le enseñó la salida.
    """
    # Sin directorio, la carpeta del usuario y no la de Perseo: los proyectos
    # del señor Persus cuelgan de ahí y se llaman unos a otros, así que un
    # subagente arrancado dentro de este repositorio nacía mirando a la pared
    # (es la misma decisión que tomó `dev.py` el 2026-08-26).
    destino = Path(directorio or str(Path.home())).expanduser().resolve()
    if destino.exists() and not destino.is_dir():
        raise ValueError(f"{destino} es un fichero, no un directorio.")
    if not destino.is_dir():
        raise ValueError(
            f"El directorio {destino} no existe. 'directorio' es donde ARRANCA "
            "el subagente y tiene que existir ya; la carpeta nueva la crea él "
            "dentro de la tarea. Usa una carpeta real —por ejemplo "
            f"{Path.home() / 'Desktop'} para cosas del escritorio— o no pases "
            f"directorio y arrancará en {Path.home()}, desde donde ve todos "
            "los proyectos."
        )
    if not any(destino == raiz or raiz in destino.parents for raiz in RAICES_PERMITIDAS):
        fuera = ", ".join(str(r) for r in RAICES_PERMITIDAS)
        raise ValueError(f"{destino} cae fuera de los directorios permitidos ({fuera}).")
    return destino


def _telegram_de_aviso(motivo: str) -> None:
    """Avisa al móvil por Telegram, sin depender de que la app viva.

    La llamada de aviso (el marcador de abajo) necesita la app encendida; si
    está apagada, sin esto el resultado se perdería hasta que a alguien se le
    ocurriera preguntar. El token y el chat se leen de los mismos ficheros que
    usa el núcleo (`datos/telegram*.txt`), así que no hay secretos duplicados.
    Nunca lanza: un aviso que falla no puede tumbar el servidor.
    """
    try:
        token = (DATOS / "telegram.txt").read_text(encoding="utf-8").strip()
        chat = (DATOS / "telegram_chat.txt").read_text(encoding="utf-8").strip()
        if not token or not chat:
            return
        carga = json.dumps({"chat_id": chat, "text": motivo[:500]}).encode("utf-8")
        peticion = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=carga,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(peticion, timeout=10).read()
    except OSError:
        pass


def _avisar_si_nadie_pregunto(id_tarea: str) -> None:
    """Pasado el plazo de gracia sin consulta, avisa por dos caminos.

    El marcador hace que la app LLAME cuando esté viva; el Telegram llega al
    móvil aunque esté apagada. Los dos dicen lo mismo: qué terminó y cómo.

    Avisar ES entregar: quien tiene que saberlo, ya lo sabe — por oído o por
    Telegram. Sin marcar `entregado` aquí, una tarea vieja seguía «pendiente»
    para siempre y cada camino que la revisaba volvía a contarla (H-62).
    """
    with _cerrojo:
        tarea = _tareas.get(id_tarea)
        if tarea is None or tarea["entregado"]:
            return
        if tarea["estado"] == "en_curso":
            return
        resumen = tarea.get("error") or tarea.get("salida") or ""
        legible = _ESTADOS_LEGIBLES.get(tarea["estado"], tarea["estado"])
        motivo = (
            f"El subagente ({_quien(tarea)}) {legible} en la tarea "
            f"{id_tarea}: {_primera_linea(resumen)}"
        )
        tarea["entregado"] = True
        try:
            MARCA_AUTOLLAMADA.write_text(motivo, encoding="utf-8")
        except OSError:
            pass
    _persistir()
    _telegram_de_aviso(motivo)


#: Si el primer intento falló habiendo durado MENOS de esto, se repite una vez.
#: Los picos de red de los modelos gratuitos matan el encargo en los primeros
#: segundos; una tarea que llevaba veinte minutos trabajando de verdad no se
#: reejecuta — repetirla sería tirar el trabajo hecho.
REINTENTO_SI_MENOS_DE = 120.0


#: Lo que un CLI de agente escribe cuando ha fracasado PERO sale con código 0.
#: Los dos casos vistos el 2026-08-24: opencode denegándose a sí mismo el
#: permiso de escribir (sin `--auto`, ya arreglado arriba) y el proveedor del
#: modelo gratuito cayéndose a media petición. Los dos dejaban el encargo en
#: "hecho" con el disco intacto, que es la peor forma de fallar: la que no se ve.
_SENALES_DE_FRACASO = (
    "auto-rejecting",
    "rejected permission",
    "error from provider",
    "endpoint is unavailable",
    "no such model",
    # Y cuando el que se queda esperando un sí es Claude: sin nadie al otro
    # lado, contesta esto y se va con código 0.
    "needs your approval",
    "requires approval",
    "necesita tu aprobación",
    "requiere tu aprobación",
)


def _fracaso_encubierto(texto: str) -> str:
    """El motivo, si la salida delata un fracaso con código de éxito. O ''."""
    bajo = (texto or "").lower()
    for senal in _SENALES_DE_FRACASO:
        if senal in bajo:
            for linea in reversed((texto or "").splitlines()):
                if senal in linea.lower():
                    return _primera_linea(linea)
            return senal
    return ""


def _una_vez(comando: list[str], directorio: Path):
    banderas = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.run(
        comando,
        cwd=str(directorio),
        # El hijo NO hereda nuestro stdin: ese tubo es el canal JSON-RPC
        # con el núcleo, y un agente que lo leyera se comería las
        # peticiones — visto en la práctica, un "tools/call" perdido cada
        # tantos encargos. Devnull y todos contentos.
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=TOPE_SEGUNDOS,
        creationflags=banderas,
    )


def _ejecutar(id_tarea: str, motor: str, directorio: Path, tarea: str, modelo: str = "") -> None:
    def linea() -> list[str]:
        return _comando(motor, tarea, directorio, modelo)

    try:
        comienzo = time.time()
        hecho = _una_vez(linea(), directorio)
        # Hasta dos reintentos contra los picos de red del proveedor, siempre
        # que el fallo llegue rápido: una tarea que llevaba minutos trabajando
        # de verdad no se reejecuta — repetirla sería tirar lo hecho.
        vueltas = 0
        while (
            hecho.returncode == 0
            and (
                not (hecho.stdout or "").strip()
                # Un «Endpoint is unavailable» del proveedor es exactamente el
                # pico de red contra el que existe este reintento: se ve en la
                # salida, no en el código de salida.
                or _fracaso_encubierto(hecho.stdout or "")
            )
            and time.time() - comienzo < REINTENTO_SI_MENOS_DE
            and vueltas < 1
        ):
            # Salir 0 SIN decir nada es la otra forma de fallar del modelo
            # gratuito: responde solo con controles ANSI o se calla. Un reintento
            # rápido cuesta poco y suele traer el resultado de verdad.
            vueltas += 1
            hecho = _una_vez(linea(), directorio)
        while hecho.returncode != 0 and time.time() - comienzo < REINTENTO_SI_MENOS_DE and vueltas < 2:
            vueltas += 1
            hecho = _una_vez(linea(), directorio)
        salida = _recortar(hecho.stdout or "")
        if not salida and (hecho.stderr or "").strip():
            salida = _recortar(hecho.stderr)
        if hecho.returncode == 0 and not salida:
            salida = "(el agente terminó sin devolver texto; comprueba el resultado en el disco)"
        error = "" if hecho.returncode == 0 else (
            _recortar((hecho.stderr or "")[:400], 400)
            or f"el proceso terminó con código {hecho.returncode}"
        )
        estado = "hecho" if hecho.returncode == 0 else "fallido"
        if estado == "hecho":
            # Código 0 no basta: hay que leer lo que dijo. Un encargo dado por
            # bueno sin haber tocado nada es lo que hacía que Perseo contara
            # como terminado algo que no existía.
            motivo = _fracaso_encubierto(
                (hecho.stdout or "") + "\n" + (hecho.stderr or "")
            )
            if motivo:
                estado, error = "fallido", _recortar(motivo, 400)
    except subprocess.TimeoutExpired:
        estado, salida, error = "fallido", "", f"pasó de {TOPE_SEGUNDOS:.0f} s y se cortó"
    except OSError as e:
        estado, salida, error = "fallido", "", f"no se pudo arrancar {motor}: {e}"

    with _cerrojo:
        registro = _tareas[id_tarea]
        registro.update(estado=estado, salida=salida, error=error, fin=time.time())

    _persistir()
    threading.Timer(GRACIA_SEGUNDOS, _avisar_si_nadie_pregunto, args=(id_tarea,)).start()


def encargar_tarea(tarea: str, directorio: str = "", motor: str = "", modelo: str = "") -> str:
    """Lanza un subagente y devuelve su identificador al momento.

    `motor` y `modelo` los puede elegir Perseo en voz alta —«mándalo con
    Claude», «usa el nemotron»— y, si no dice nada, sale opencode con un
    modelo gratuito: es lo que pidió el señor Persus el 2026-08-26.
    """
    global _contador
    tarea = (tarea or "").strip()
    if not tarea:
        raise ValueError("Falta la descripción de la tarea.")

    destino = _validar_directorio(directorio)
    motor = _motor(motor)
    modelo = _modelo_de(motor, modelo)

    with _cerrojo:
        _contador += 1
        id_tarea = f"s{_contador}"
        _tareas[id_tarea] = {
            "estado": "en_curso",
            "salida": "",
            "error": "",
            "motor": motor,
            "modelo": modelo,
            "entregado": False,
            "inicio": time.time(),
        }
    _persistir()

    threading.Thread(
        target=_ejecutar,
        args=(id_tarea, motor, destino, tarea, modelo),
        daemon=True,
        name=f"subagente-{id_tarea}",
    ).start()

    detalle = f" ({modelo})" if modelo else ""
    return (
        f"Encargo {id_tarea} en marcha con {motor}{detalle} en {destino}. "
        "Sigue con otra cosa y consúltalo con consultar_tarea cuando toque."
    )


_ESTADOS_LEGIBLES = {
    "hecho": "terminó con éxito",
    "fallido": "FALLÓ",
    "perdido": "SE PERDIÓ (Perseo se apagó mientras trabajaba)",
}


def consultar_tarea(id_tarea) -> str:
    """El estado de un encargo. Un id desconocido NO es un error: es información.

    Tratarlo como fallo hacía que Perseo entrara en espiral («ha habido un
    error…») por lo que solo es que esa tarea no existe aquí — se lanzó en una
    sesión anterior o el identificador no es de este servidor.

    Y ojo con el nombre del parámetro: se llama `encargo` y no `id` a propósito
    — las llamadas de Gemini llevan SU propio `id` numérico interno, y el modelo
    copiaba ese número aquí en vez del identificador que él mismo recibió.
    """
    if isinstance(id_tarea, (int, float)):
        id_tarea = f"{id_tarea:.0f}"
    id_tarea = str(id_tarea or "").strip()
    with _cerrojo:
        registro = _tareas.get(id_tarea)
        if registro is not None:
            registro["entregado"] = True
            datos = dict(registro)
    if registro is None:
        pista = (
            " (los identificadores son los que devolvió encargar_tarea: s1, s2…)"
            if id_tarea.replace(".0", "").isdigit()
            else ""
        )
        return (
            f"No hay ninguna tarea llamada '{id_tarea}'{pista}. O el "
            "identificador no es de aquí, o el encargo es de antes de un "
            "reinicio y su seguimiento se perdió. Lo que sí hay ahora mismo lo "
            "dice listar_tareas."
        )
    quien = _quien(datos)
    if datos["estado"] == "en_curso":
        return f"Tarea {id_tarea}: sigue en marcha ({quien})."
    cuerpo = datos["salida"] or datos["error"]
    legible = _ESTADOS_LEGIBLES.get(datos["estado"], datos["estado"])
    return f"Tarea {id_tarea} ({quien}) {legible}:\n{cuerpo}"


def listar_tareas() -> str:
    with _cerrojo:
        registros = [(i, dict(t)) for i, t in sorted(_tareas.items())]
    if not registros:
        return (
            "No hay ningún encargo de subagente lanzado todavía. Se lanzan con "
            "encargar_tarea."
        )
    lineas = [
        f"- {i}: {t['estado']} ({_quien(t)}) — {_primera_linea(t['salida'] or t['error']) or 'sin salida aún'}"
        for i, t in registros
    ]
    return "Encargos de subagentes:\n" + "\n".join(lineas)


# --------------------------------------------------------------------------- #
# Servidor JSON-RPC sobre stdio, una línea por mensaje
# --------------------------------------------------------------------------- #

HERRAMIENTAS = [
    {
        "name": "encargar_tarea",
        "description": (
            "Lanza un subagente de programación —opencode GRATIS de serie, Claude "
            "si se pide— que trabaja "
            "SOLo en una tarea sobre un directorio local: arreglar fallos, añadir "
            "funciones, refactorizar, escribir scripts. Devuelve un identificador al "
            "momento; el trabajo sigue aunque hables de otra cosa o cuelgues. Lanza "
            "tantos en paralelo como haga falta y consúltalos con consultar_tarea."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tarea": {
                    "type": "string",
                    "description": "La instrucción completa y autocontenida: qué hacer, con qué criterio.",
                },
                "directorio": {
                    "type": "string",
                    "description": (
                        "Carpeta EXISTENTE desde la que arranca el agente: la raíz "
                        "del proyecto, o el escritorio del señor Persus para tareas "
                        "suyas (C:\\Users\\<usuario>\\Desktop). NO es la carpeta a "
                        "crear — eso lo hace el subagente dentro de la tarea. Si no "
                        "se pasa, arranca en la carpeta del usuario "
                        "(C:\\Users\\<usuario>), desde donde ve todos los "
                        "proyectos y puede crear carpetas nuevas."
                    ),
                },
                "motor": {
                    "type": "string",
                    "enum": ["opencode", "claude"],
                    "description": (
                        "Con qué CLI trabaja. Por defecto 'opencode', que es "
                        "GRATIS y es lo que el señor Persus quiere de serie. "
                        "Usa 'claude' solo si él lo pide o si el encargo es "
                        "grande y delicado."
                    ),
                },
                "modelo": {
                    "type": "string",
                    "enum": list(MODELOS_GRATIS),
                    "description": (
                        "Modelo de opencode, todos gratuitos. Por defecto "
                        f"{MODELO_POR_DEFECTO} (1M de contexto). Pásalo solo "
                        "si el señor Persus nombra uno."
                    ),
                },
            },
            "required": ["tarea"],
        },
    },
    {
        "name": "consultar_tarea",
        "description": (
            "Cómo va un encargo lanzado con encargar_tarea. Si terminó, trae su "
            "resultado entero; consultarlo CANCELA la llamada de aviso, así que "
            "pregunta solo cuando puedas contárselo al señor Persus."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "encargo": {
                    "type": "string",
                    "description": (
                        "El identificador LITERAL que devolvió encargar_tarea: "
                        "'s1', 's2'… Nunca un número largo ni el id interno de la "
                        "llamada."
                    ),
                }
            },
            "required": ["encargo"],
        },
    },
    {
        "name": "listar_tareas",
        "description": "Lista de los encargos de subagentes lanzados en esta sesión, con su estado.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
]

_ACCIONES = {
    "encargar_tarea": lambda a: encargar_tarea(
        str(a.get("tarea", "")),
        str(a.get("directorio", "") or ""),
        str(a.get("motor", "") or ""),
        str(a.get("modelo", "") or ""),
    ),
    # Se acepta el 'id' viejo por compatibilidad con sesiones ya arrancadas.
    "consultar_tarea": lambda a: consultar_tarea(
        a.get("encargo") if a.get("encargo") not in (None, "") else a.get("id")
    ),
    "listar_tareas": lambda a: listar_tareas(),
}


def _responder(identificador, resultado) -> None:
    print(json.dumps({"jsonrpc": "2.0", "id": identificador, "result": resultado}), flush=True)


def main() -> None:
    _cargar()
    for linea in sys.stdin:
        linea = linea.strip()
        if not linea:
            continue
        try:
            mensaje = json.loads(linea)
        except json.JSONDecodeError:
            continue
        metodo = str(mensaje.get("method", ""))
        identificador = mensaje.get("id")
        if identificador is None:
            continue  # notificación: nada que contestar

        if metodo == "initialize":
            _responder(
                identificador,
                {
                    "protocolVersion": VERSION_PROTOCOLO,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "subagentes", "version": "1.0"},
                },
            )
        elif metodo == "tools/list":
            _responder(identificador, {"tools": HERRAMIENTAS})
        elif metodo == "tools/call":
            parametros = mensaje.get("params") or {}
            nombre = str((parametros.get("name") or ""))
            argumentos = parametros.get("arguments") or {}
            accion = _ACCIONES.get(nombre)
            if accion is None:
                _responder(
                    identificador,
                    {
                        "content": [{"type": "text", "text": f"herramienta desconocida: {nombre}"}],
                        "isError": True,
                    },
                )
                continue
            try:
                _responder(identificador, {"content": [{"type": "text", "text": accion(argumentos)}]})
            except Exception as e:  # noqa: BLE001 — el error viaja al modelo, no tumba el servidor
                _responder(
                    identificador,
                    {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True},
                )
        else:
            print(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": identificador,
                        "error": {"code": -32601, "message": f"no sé hacer {metodo}"},
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
