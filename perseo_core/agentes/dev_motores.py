"""Los motores del agente `dev`: con qué CLI o SDK se hace el encargo.

Salieron de `dev.py` el 2026-09-12 porque el fichero pasaba de mil quinientas
líneas y la costura estaba a la vista: por un lado **con qué** se trabaja —cuatro
motores intercambiables detrás del mismo `Protocol`— y por otro **qué** se le
encarga, que es el agente y se queda allí.

Aquí vive también la lista de lo que un encargo puede ejecutar. No es papeleo:
es el cerco del agente, y en Claude Code lo denegado gana siempre sobre lo
permitido.

El motor sobre el SDK está aparte, en `dev_sdk.py`: es el único que sabe contar
por dónde va, y eso le cuesta doscientas líneas que no tienen que ver con los
otros tres.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any, Protocol


logger = logging.getLogger(__name__)


#: Lo que `dev` puede ejecutar sin preguntar. Es corta a propósito: lo justo para
#: que compruebe lo que acaba de escribir. Todo lo demás se deniega.
HERRAMIENTAS_PERMITIDAS = (
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
    "TodoWrite",
    "Bash(python -m pytest*)",
    "Bash(python -m perseo_core*)",
    "Bash(python perseo_core/verificar*)",
    "Bash(npx tsc*)",
    "Bash(cargo check*)",
    "Bash(git status*)",
    "Bash(git diff*)",
    "Bash(git log*)",
)

#: Lo que no se ejecuta ni aunque se permitiera por otro lado. Publicar es del
#: usuario, y borrar no tiene vuelta atrás: las dos cosas están fuera del nivel
#: "reversible" de §7.
HERRAMIENTAS_DENEGADAS = (
    "Bash(git push*)",
    "Bash(git reset --hard*)",
    "Bash(git clean*)",
    "Bash(rm *)",
    "Bash(rmdir *)",
    "Bash(del *)",
    "Bash(format*)",
    "WebFetch",
    "WebSearch",
)

#: Lo que puede un encargo que ha pedido el señor Persus con el dedo —desde el
#: panel, desde el móvil o hablando—. Es la lista de arriba MÁS `Bash` a secas
#: y `Task`, y existe por el fallo del 2026-08-25: «Abre la app de armario»
#: terminó en verde dos veces seguidas SIN abrir nada, porque el agente no
#: tenía con qué arrancar un proceso y se limitó a leer ficheros y a contarlo
#: bien. Un agente que no puede hacer el encargo no debe poder decir que lo
#: hizo, y la forma de arreglarlo es darle las manos, no bajar el listón.
#:
#: `Bash` abierto NO relaja lo denegado: en Claude Code lo denegado gana
#: siempre sobre lo permitido, así que publicar, borrar y formatear siguen
#: fuera con esta lista igual que con la corta.
#:
#: Lo que **no** se amplía es el encargo que nace solo —un correo triado, un
#: disparador—: eso sigue con la lista corta, que es la razón de que exista.
#: Quién lo pidió ya lo sabe la cola: `trabajo.origen`.
HERRAMIENTAS_PERMITIDAS_AMPLIAS = HERRAMIENTAS_PERMITIDAS + (
    "Bash",
    "Task",
    "NotebookEdit",
)

#: Los orígenes que son el señor Persus en persona. Lo demás —`disparador`— es
#: trabajo que nació de algo que Perseo leyó, y ese va con la lista corta.
ORIGENES_DE_CONFIANZA = ("texto", "voz")

#: Tope de vueltas del bucle de agente. Sin él, un encargo mal entendido puede
#: dar vueltas sin fin contra la cuota de la suscripción.
MAX_VUELTAS = 40


@dataclass(frozen=True)
class Resultado:
    """Lo que devuelve un motor. `sesion` permite continuar el encargo después."""

    texto: str
    ok: bool = True
    vueltas: int = 0
    sesion: str = ""


@dataclass(frozen=True)
class Encargo:
    """Todo lo que un motor necesita para una vuelta de trabajo.

    Va en un objeto y no en seis argumentos sueltos porque lo que un encargo
    lleva encima ha ido creciendo —el modelo, el permiso, las carpetas de al
    lado— y una firma de seis posiciones es donde empiezan los errores de
    llamar en el orden equivocado.

    `contexto` se antepone a la instrucción al hablar con el modelo, pero
    **no** forma parte de ella: lo que el señor Persus escribió se guarda tal
    cual, que es lo que luego se lee en el panel.
    """

    instruccion: str
    raiz: Path
    tope: float
    sesion: str = ""
    #: El modelo pedido para ESTE encargo (`sonnet`, `opus`, `haiku`, o un id
    #: entero). Vacío: el que traiga el motor por defecto.
    modelo: str = ""
    contexto: str = ""
    permitidas: tuple[str, ...] = HERRAMIENTAS_PERMITIDAS
    #: Otras carpetas que el agente puede tocar además de `raiz`. Sin esto, un
    #: encargo con carpeta explícita se queda ciego para el resto del perfil, y
    #: los proyectos del señor Persus se llaman unos a otros.
    carpetas_extra: tuple[Path, ...] = ()

    @property
    def prompt(self) -> str:
        """Lo que se le dice al modelo: el contexto y luego el encargo."""
        if not self.contexto:
            return self.instruccion
        return f"{self.contexto}\n\n{self.instruccion}"


@dataclass(frozen=True)
class Paso:
    """Una línea de la bitácora de un encargo: qué hizo y quién lo hizo.

    Es lo que la pestaña de agentes enseña para poder depurar sin abrir el
    registro del núcleo. `agente` vale `principal` o el identificador del
    subagente, que es lo que permite meterse dentro de uno y ver SU trabajo.
    """

    tipo: str
    titulo: str
    detalle: str = ""
    agente: str = "principal"
    ok: bool = True
    momento: str = ""

    def a_dict(self) -> dict[str, Any]:
        return {
            "tipo": self.tipo,
            "titulo": self.titulo,
            "detalle": self.detalle,
            "agente": self.agente,
            "ok": self.ok,
            "momento": self.momento,
        }


#: Cómo avisa un motor de por dónde va. Solo el motor sobre el SDK sabe
#: rellenarlo —los que hablan por línea de órdenes no dicen nada hasta el
#: final—, y quien no lo use no paga nada por tenerlo.
Aviso = Callable[[Paso], None]


class Motor(Protocol):
    """Quién ejecuta de verdad el encargo."""

    async def ejecutar(self, encargo: Encargo, avisar: Aviso | None = None) -> Resultado: ...


class MotorClaude:
    """Claude Code en modo no interactivo (`claude -p`).

    Se habla con él por línea de comandos y no por biblioteca porque es lo que
    está instalado. La salida se pide en JSON, que trae el texto final, el número
    de vueltas y el identificador de sesión para poder retomar el encargo.
    """

    def __init__(self, ejecutable: str) -> None:
        # El envoltorio `.cmd` de npm corta la orden en el primer salto de
        # línea, y el prompt de un encargo lleva dos. Ver `ejecutable_real`.
        self._ejecutable = ejecutable_real(ejecutable)

    async def ejecutar(self, encargo: Encargo, avisar: Aviso | None = None) -> Resultado:
        argumentos = [
            self._ejecutable,
            "-p",
            encargo.prompt,
            "--output-format",
            "json",
            # Las ediciones se aceptan solas; lo que no está en la lista de
            # permitidas se deniega en vez de quedarse esperando una respuesta
            # que aquí no puede dar nadie.
            "--permission-mode",
            "acceptEdits",
            "--max-turns",
            str(MAX_VUELTAS),
            "--allowed-tools",
            *encargo.permitidas,
            "--disallowed-tools",
            *HERRAMIENTAS_DENEGADAS,
        ]
        for carpeta in encargo.carpetas_extra:
            argumentos += ["--add-dir", str(carpeta)]
        if encargo.modelo:
            argumentos += ["--model", encargo.modelo]
        if encargo.sesion:
            argumentos += ["--resume", encargo.sesion]

        proceso = await asyncio.create_subprocess_exec(
            *argumentos,
            cwd=str(encargo.raiz),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            salida, error = await asyncio.wait_for(
                proceso.communicate(), timeout=encargo.tope
            )
        except asyncio.TimeoutError:
            proceso.kill()
            await proceso.wait()
            raise TimeoutError(f"El encargo pasó de {encargo.tope:.0f} s y se cortó.") from None

        texto = salida.decode("utf-8", "replace").strip()
        if proceso.returncode != 0:
            detalle = error.decode("utf-8", "replace").strip()[:500] or texto[:500]
            return Resultado(texto=detalle or "Claude terminó con error.", ok=False)

        try:
            datos = json.loads(texto)
        except json.JSONDecodeError:
            # Sin JSON no hay metadatos, pero el texto sigue valiendo.
            return Resultado(texto=texto[:4000])

        return Resultado(
            texto=str(datos.get("result", ""))[:4000],
            ok=not datos.get("is_error", False),
            vueltas=int(datos.get("num_turns", 0) or 0),
            sesion=str(datos.get("session_id", "")),
        )


#: Lo que un CLI de agente escribe cuando ha fracasado PERO sale con código 0.
#: Dos casos vistos el 2026-08-24: opencode denegándose a sí mismo el permiso de
#: escribir (sin `--auto`) y el proveedor del modelo gratuito cayéndose a media
#: petición. Los dos daban el encargo por bueno con el disco intacto.
SENALES_DE_FRACASO = (
    "auto-rejecting",
    "rejected permission",
    "error from provider",
    "endpoint is unavailable",
    "no such model",
)


def fracaso_encubierto(texto: str) -> str:
    """El motivo, si la salida delata un fracaso con código de éxito. O ''."""
    bajo = (texto or "").lower()
    for senal in SENALES_DE_FRACASO:
        if senal in bajo:
            for linea in reversed((texto or "").splitlines()):
                if senal in linea.lower():
                    return linea.strip()[:400]
            return senal
    return ""


#: Los modelos GRATIS de opencode Zen que **contestan**, de más rápido a menos.
#: Salen de `opencode models opencode` y de probarlos uno a uno: la lista de
#: antes se ordenaba por contexto y encabezaba con dos que ya no responden.
#:
#: Medido el 2026-08-28, un «di HOLA» por modelo:
#:   big-pickle 10 s · hy3-free 9 s · muse-spark 9 s · ling-3.0-flash 7 s
#:   mimo-v2.5 27 s · nemotron-3-ultra NADA en 100 s · nemotron-3.5 NADA en 100 s
#:
#: Los dos nemotron se quedan al final y no de adorno: si vuelven, ahí están;
#: mientras no vuelvan, no los coge nadie por defecto. Un modelo que no
#: contesta no falla — se cuelga hasta `dev_tope` (900 s), y desde el panel eso
#: se ve como un encargo que no termina nunca.
#:
#: Es la MISMA lista que ofrecen la pestaña de encargos del panel
#: (`RealTime/src/components/Panel.tsx`, `perseo_core/caras/interfaz/index.html`) y el
#: servidor MCP de subagentes (`commands/subagentes_mcp.py`). Si cambia una,
#: cambian todas: se vuelven a sacar del mismo comando y se vuelven a probar.
MODELOS_GRATIS_OPENCODE = (
    "opencode/big-pickle",
    "opencode/hy3-free",
    "opencode/muse-spark-1.2-contributor-free",
    "opencode/ling-3.0-flash-fin-free",
    "opencode/mimo-v2.5-free",
    # No contestaban el 2026-08-28: cien segundos sin una sola línea. Van al
    # final para que nadie los coja sin pedirlos.
    "opencode/nemotron-3-ultra-free",
    "opencode/nemotron-3.5-lightning-free",
)

#: Con qué trabaja opencode si nadie elige. El primero de los que contestan.
MODELO_OPENCODE_POR_DEFECTO = MODELOS_GRATIS_OPENCODE[0]


def ejecutable_real(ruta: str) -> str:
    """El binario de verdad detrás de un envoltorio `.cmd` de npm.

    En Windows `shutil.which("opencode")` devuelve `opencode.CMD`, y lanzar un
    `.cmd` pasa por `cmd.exe`, que **corta la línea de órdenes en el primer
    salto de línea**. El prompt de un encargo es `contexto + "\n\n" + encargo`:
    al modelo le llegaba el contexto y **nunca el encargo**. Contestaba «¿cuál
    es la tarea?», el trabajo se apuntaba como HECHO, y el disco intacto.
    Medido el 2026-08-28 con dos modelos distintos, los dos igual.

    El envoltorio de npm no hace nada más que llamar al `.exe` con `%*`, así
    que se llama a ese directamente y el salto de línea sobrevive. Si el
    envoltorio tiene otra forma —dos rutas entrecomilladas, como los que llaman
    a `node.exe script.js`— se deja como estaba: mejor el fallo conocido que
    una orden mal montada.

    Hay una **copia deliberada** en `commands/subagentes_mcp.py`, que corre
    como proceso suelto y no importa el núcleo. Si tocas esta, toca la otra:
    `docs/adr/0004-una-copia-deliberada-de-ejecutable-real.md`.
    """
    if os.name != "nt" or not ruta.lower().endswith((".cmd", ".bat")):
        return ruta
    try:
        texto = Path(ruta).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ruta
    base = Path(ruta).parent
    for linea in texto.splitlines():
        if "%*" not in linea:
            continue
        entrecomillados = re.findall(r'"([^"]+)"', linea)
        if len(entrecomillados) != 1:
            return ruta
        destino = entrecomillados[0]
        for marca in ("%~dp0", "%dp0%"):
            destino = destino.replace(marca, "")
        candidato = base / destino.lstrip("\\/")
        if candidato.suffix.lower() == ".exe" and candidato.exists():
            return str(candidato)
    return ruta


def modelo_opencode(pedido: str = "") -> str:
    """El modelo de un encargo de opencode. Nunca vacío, y gratis por defecto.

    Un nombre a medias («hy3-free») se completa con el proveedor: es lo que
    escribe un modelo de voz cuando le dictan el nombre sin la barra.
    """
    elegido = (pedido or os.environ.get("PERSEO_DEV_MODELO", "")).strip()
    if not elegido:
        return MODELO_OPENCODE_POR_DEFECTO
    return elegido if "/" in elegido else f"opencode/{elegido}"


class MotorOpencode:
    """opencode en modo no interactivo (`opencode run`).

    El segundo motor, para que el señor Persus ELIJA con quién trabaja cada
    encargo (2026-08-24): Claude de suscripción u opencode gratuito. Los
    permisos van en `--auto`, porque sin él este programa se deniega a sí mismo
    lo que necesita para trabajar y sale con código 0 igualmente. El modelo, en
    `PERSEO_DEV_MODELO` si se quiere uno concreto. La salida llega formateada y
    con controles ANSI; se limpian aquí una vez, en un sitio.

    **Es el motor de serie desde el 2026-08-26**, y no por ser mejor: es el que
    no gasta suscripción, y el señor Persus lo pidió con esas palabras. Lo que
    antes lo dejaba fuera —su endpoint gratuito se cae a ratos y sale con
    código 0 sin haber hecho nada— ya no lo decide todo, porque
    `fracaso_encubierto` lee la salida y lo cuenta como el fallo que es.
    """

    _ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

    def __init__(self, ejecutable: str) -> None:
        # El envoltorio `.cmd` de npm corta la orden en el primer salto de
        # línea, y el prompt de un encargo lleva dos. Ver `ejecutable_real`.
        self._ejecutable = ejecutable_real(ejecutable)

    async def ejecutar(self, encargo: Encargo, avisar: Aviso | None = None) -> Resultado:
        # `--auto` no es una comodidad: sin él, `opencode run` pide permiso para
        # escribir, nadie contesta porque esto no es interactivo, y el propio
        # programa se lo deniega («auto-rejecting») saliendo con código 0. El
        # encargo se apuntaba como hecho sin haber tocado un fichero (2026-08-24).
        # `--dir` NO es redundante con el `cwd` del proceso, y costó un fichero
        # escrito en la raíz del repositorio para verlo (2026-08-26): `opencode
        # run` levanta su propio servidor y resuelve el proyecto por su cuenta,
        # así que hereda el `cwd` y luego lo ignora. El encargo decía «ok» y
        # había creado el fichero DOS CARPETAS más arriba. Es el mismo fallo, con
        # otro motor: la raíz hay que decírsela, no dársela por supuesta.
        argumentos = [
            self._ejecutable,
            "run",
            "--auto",
            "--dir",
            str(encargo.raiz),
            # Los eventos en crudo, uno por línea. No es una preferencia de
            # formato: es lo que permite contar POR DÓNDE VA el encargo. Con la
            # salida bonita, opencode no dice nada hasta el final y la pestaña
            # de actividad se quedaba en blanco justo con el motor que el señor
            # Persus quiere usar a diario (2026-08-26).
            "--format",
            "json",
        ]
        # El modelo del encargo manda sobre el del entorno: elegirlo en la
        # pantalla no sirve de nada si una variable lo pisa. Y si no lo dice
        # nadie, uno GRATIS por su nombre — sin `-m`, opencode trabaja con el
        # que tenga configurado, que puede ser de pago.
        argumentos += ["-m", modelo_opencode(encargo.modelo)]
        if encargo.sesion:
            argumentos += ["--session", encargo.sesion]
        argumentos.append(encargo.prompt)

        proceso = await asyncio.create_subprocess_exec(
            *argumentos,
            cwd=str(encargo.raiz),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # `stderr` se vacía en paralelo y no al final: un error largo llena su
        # tubería, el proceso se bloquea escribiendo y el encargo se queda
        # colgado hasta el tope sin que nadie sepa por qué.
        async def tragar_error() -> bytes:
            assert proceso.stderr is not None
            return await proceso.stderr.read()

        tarea_error = asyncio.create_task(tragar_error())
        try:
            async with asyncio.timeout(encargo.tope):
                dichos, sesion = await self._leer_eventos(proceso, avisar)
                await proceso.wait()
                error = await tarea_error
        except TimeoutError:
            tarea_error.cancel()
            proceso.kill()
            await proceso.wait()
            raise TimeoutError(f"El encargo pasó de {encargo.tope:.0f} s y se cortó.") from None

        texto = "\n".join(dichos).strip()
        if proceso.returncode != 0:
            detalle = self._ANSI.sub("", error.decode("utf-8", "replace")).strip()[:500]
            if avisar is not None:
                avisar(Paso(tipo="error", titulo=(detalle or "opencode falló")[:120], ok=False))
            return Resultado(texto=detalle or texto[:500] or "opencode terminó con error.", ok=False)

        # Código 0 tampoco basta aquí: el modelo gratuito devuelve «Endpoint is
        # unavailable» y sale bien. Un encargo que no hizo nada tiene que
        # contarse como fallo, o el panel enseña éxitos que no existieron.
        motivo = fracaso_encubierto(texto)
        if motivo:
            if avisar is not None:
                avisar(Paso(tipo="error", titulo=motivo[:120], ok=False))
            return Resultado(texto=motivo, ok=False, sesion=sesion)

        if avisar is not None:
            avisar(Paso(tipo="fin", titulo="Terminado", detalle=_recortar(texto, 2000)))
        return Resultado(texto=texto[:4000], sesion=sesion)

    async def _leer_eventos(
        self, proceso: Any, avisar: Aviso | None
    ) -> tuple[list[str], str]:
        """Va leyendo los eventos según salen y los apunta en la bitácora.

        Devuelve lo que dijo el agente y el identificador de sesión, que es lo
        que permite continuar un encargo donde se quedó. Antes se perdía: con la
        salida bonita no venía por ninguna parte, así que `peticion.sesion` no
        servía de nada con este motor.

        Una línea que no sea JSON no se descarta como ruido: `--format json`
        manda eventos, pero un aviso del propio programa puede colarse, y en un
        fallo suele ser justo lo único que explica algo.
        """
        dichos: list[str] = []
        sesion = ""
        assert proceso.stdout is not None
        async for cruda in proceso.stdout:
            linea = self._ANSI.sub("", cruda.decode("utf-8", "replace")).strip()
            if not linea:
                continue
            try:
                evento = json.loads(linea)
            except json.JSONDecodeError:
                dichos.append(linea)
                continue
            if not isinstance(evento, dict):
                continue
            sesion = str(evento.get("sessionID") or sesion)
            paso = self._paso_del_evento(evento, dichos)
            if paso is not None and avisar is not None:
                avisar(paso)
        return dichos, sesion

    @staticmethod
    def _paso_del_evento(evento: dict[str, Any], dichos: list[str]) -> Paso | None:
        """Un evento de opencode traducido a línea de bitácora, o nada.

        Los `step_start` se tiran: son el latido del bucle y no cuentan nada que
        no cuente ya la herramienta que viene detrás.
        """
        tipo = str(evento.get("type") or "")
        parte = evento.get("part") or {}
        if not isinstance(parte, dict):
            return None

        if tipo == "text":
            texto = str(parte.get("text") or "")
            if not texto.strip():
                return None
            dichos.append(texto)
            return Paso(tipo="dice", titulo=_recortar(texto, 200), detalle=_recortar(texto, 2000))

        if tipo == "tool_use":
            estado = parte.get("state") or {}
            herramienta = str(parte.get("tool") or "")
            # opencode nombra sus herramientas en minúscula («write», «bash»);
            # se traducen a las mismas palabras que las de Claude para que la
            # pestaña se lea igual con los dos motores.
            nombre = _NOMBRES_OPENCODE.get(herramienta, herramienta)
            entrada = estado.get("input") if isinstance(estado, dict) else None
            fallo = str((estado or {}).get("status", "")) == "error"
            return Paso(
                tipo="herramienta",
                titulo=_contar_herramienta(nombre, _entrada_de_opencode(entrada)),
                detalle=_recortar(str((estado or {}).get("output") or _texto_de_entrada(entrada)), 2000),
                ok=not fallo,
            )

        if tipo == "step_finish":
            razon = str(parte.get("reason") or "")
            if razon and razon != "stop":
                return Paso(tipo="fin", titulo=f"Paso terminado: {razon}"[:120])
        return None


#: Cómo se cuenta cada herramienta mientras el encargo corre. Se dice qué está
#: pasando, no el JSON de la llamada: quien mira quiere saber si avanza.
_COMO_SE_CUENTA = {
    "Read": "Leyendo",
    "Write": "Escribiendo",
    "Edit": "Editando",
    "Glob": "Buscando ficheros",
    "Grep": "Buscando",
    "Bash": "Ejecutando",
    "TodoWrite": "Ordenando el trabajo",
    "Task": "Repartiendo a un subagente",
}


# opencode nombra sus herramientas en minúscula y con otras palabras. Se
# traducen a los nombres de Claude para que `_COMO_SE_CUENTA` valga para los dos
# motores y la pestaña de actividad se lea igual con cualquiera de ellos.
_NOMBRES_OPENCODE = {
    "read": "Read",
    "write": "Write",
    "edit": "Edit",
    "patch": "Edit",
    "glob": "Glob",
    "list": "Glob",
    "grep": "Grep",
    "bash": "Bash",
    "todowrite": "TodoWrite",
    "todoread": "TodoWrite",
    "task": "Task",
}

# La misma traducción para las claves de la entrada: opencode escribe
# `filePath` donde Claude escribe `file_path`. Sin traducirlas, la línea saldría
# con el verbo y sin el detalle —«Escribiendo» a secas—, que es justo lo que
# hacía falta saber.
_CLAVES_OPENCODE = {
    "filePath": "file_path",
    "filepath": "file_path",
    "oldString": "old_string",
    "newString": "new_string",
}


def _entrada_de_opencode(entrada: Any) -> dict[str, Any]:
    """La entrada de una herramienta de opencode con las claves de Claude."""
    if not isinstance(entrada, dict):
        return {}
    return {_CLAVES_OPENCODE.get(clave, clave): valor for clave, valor in entrada.items()}


def _nombre_de_ruta(ruta: str) -> str:
    """El nombre del fichero, venga la ruta con barras de Windows o de Unix.

    `Path(...).name` solo entiende el separador del sistema donde corre, y
    el agente puede estar en el otro: en Linux, `C:\\Users\\x\\api.py` es un
    nombre de fichero entero, y la línea del progreso salía con la ruta
    completa en vez de con `api.py`. Se parten los dos separadores.
    """
    return re.split(r"[\\/]", ruta.rstrip("\\/"))[-1] or ruta


def _contar_herramienta(nombre: str, entrada: dict[str, Any]) -> str:
    """Una línea corta y en cristiano de lo que el agente acaba de hacer."""
    verbo = _COMO_SE_CUENTA.get(nombre, nombre)
    detalle = ""
    for clave in ("file_path", "path", "pattern", "command", "description"):
        valor = entrada.get(clave) if isinstance(entrada, dict) else None
        if valor:
            detalle = (
                _nombre_de_ruta(str(valor))
                if clave in ("file_path", "path")
                else str(valor)
            )
            break
    return f"{verbo} {detalle}".strip()[:120]


def _recortar(texto: str, tope: int) -> str:
    """El texto, cortado con aviso. Una bitácora que se come la memoria del
    núcleo por guardar la salida entera de un `pytest` no es una bitácora."""
    limpio = (texto or "").strip()
    if len(limpio) <= tope:
        return limpio
    return limpio[:tope] + f"… (+{len(limpio) - tope} caracteres)"


def _texto_de_entrada(entrada: Any) -> str:
    """Lo que se le pasó a una herramienta, legible. El JSON solo si hace falta."""
    if isinstance(entrada, str):
        return entrada
    if not isinstance(entrada, dict):
        return str(entrada)
    for clave in ("command", "prompt", "content", "new_string", "pattern", "file_path"):
        if entrada.get(clave):
            return str(entrada[clave])
    try:
        return json.dumps(entrada, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(entrada)


def _texto_de_resultado(contenido: Any) -> str:
    """Lo que devolvió una herramienta, en texto. El SDK lo manda de tres formas
    —cadena, lista de bloques o diccionario— y aquí se unifican."""
    if contenido is None:
        return ""
    if isinstance(contenido, str):
        return contenido
    if isinstance(contenido, list):
        trozos: list[str] = []
        for bloque in contenido:
            if isinstance(bloque, dict):
                trozos.append(str(bloque.get("text") or bloque.get("content") or ""))
            else:
                trozos.append(str(getattr(bloque, "text", bloque)))
        return "\n".join(t for t in trozos if t)
    if isinstance(contenido, dict):
        return str(contenido.get("text") or contenido.get("content") or contenido)
    return str(contenido)


def _nombre_de_subagente(entrada: Any) -> str:
    """Cómo se llama el subagente que acaba de arrancar, para la pestaña."""
    if not isinstance(entrada, dict):
        return "Subagente"
    tipo = str(entrada.get("subagent_type") or "").strip()
    descripcion = str(entrada.get("description") or "").strip()
    if tipo and descripcion:
        return f"{tipo}: {descripcion}"[:120]
    return (tipo or descripcion or "Subagente")[:120]


def _paso_de_sistema(mensaje: Any, nombres: dict[str, str]) -> Paso | None:
    """Los avisos del propio Claude Code sobre las tareas que lanza.

    El SDK manda `TaskStarted`, `TaskProgress` y `TaskNotification` con el
    identificador de la tarea; se traducen a pasos para que un subagente tenga
    principio y final en la pantalla, y no solo un montón de herramientas.
    Cualquier otro mensaje de sistema se descarta: es ruido de protocolo.
    """
    subtipo = str(getattr(mensaje, "subtype", "") or "")
    tarea = getattr(mensaje, "tool_use_id", None) or getattr(mensaje, "task_id", None)
    agente = str(tarea) if tarea else "principal"
    descripcion = str(getattr(mensaje, "description", "") or "")
    if descripcion and agente != "principal":
        nombres.setdefault(agente, descripcion[:120])

    if subtipo == "task_started":
        # El título es la descripción a secas: es el nombre con el que este
        # subagente aparece en la pantalla, y «Empieza:» delante lo estropea.
        return Paso(tipo="subagente", titulo=(descripcion or "Subagente")[:120], agente=agente)
    if subtipo == "task_progress":
        # No se apunta: con `forward_subagent_text` las herramientas del
        # subagente ya llegan enteras, y esto repetiría la última sin detalle.
        return None
    if subtipo == "task_notification":
        estado = str(getattr(mensaje, "status", "") or "")
        return Paso(
            tipo="subagente",
            titulo=f"Termina ({estado or 'sin estado'})"[:120],
            agente=agente,
            ok=estado == "completed",
        )
    return None


class MotorFalso:
    """Motor de mentira, para verificar el circuito sin gastar suscripción.

    Existe por lo mismo que `BuzonFalso`: el camino que va de la cola al agente y
    vuelta —incluido lo que pasa cuando un encargo tarda— se puede comprobar sin
    depender de un servicio de fuera.
    """

    def __init__(self, tardanza: float = 0.0) -> None:
        self.tardanza = tardanza
        self.encargos: list[str] = []

    async def ejecutar(self, encargo: Encargo, avisar: Aviso | None = None) -> Resultado:
        self.encargos.append(encargo.instruccion)
        if avisar is not None:
            avisar(Paso(tipo="herramienta", titulo="Simulando el encargo"))
        if self.tardanza:
            await asyncio.sleep(min(self.tardanza, encargo.tope))
        if avisar is not None:
            avisar(Paso(tipo="fin", titulo="Simulado"))
        return Resultado(
            texto=f"(simulado) {encargo.instruccion}", vueltas=1, sesion="falsa"
        )
