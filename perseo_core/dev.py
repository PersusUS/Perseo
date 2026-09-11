"""Agente `dev`: escribir, programar y probar, con el Claude Agent SDK.

Este agente no se escribe, se envuelve. El bucle de agente, las herramientas, el
manejo de contexto y los subagentes ya existen en Claude Code y en su SDK;
reimplementarlos sería tirar meses para tener algo peor. Lo que hay aquí es el
puente entre la cola del núcleo y ese bucle, con la parte que sí es decisión
nuestra: **qué puede tocar y qué no**.

Sobre el coste: el Agent SDK y `claude -p` entran en la suscripción (§3 del
plan), así que este agente no gasta dinero nuevo. La restricción que sí aplica es
que la autenticación por suscripción está prohibida en productos de terceros —
Perseo es una herramienta personal y queda dentro.

**El motor es un puerto**, como el buzón y el vault. Hoy detrás está la línea de
comandos de Claude Code, que es lo que hay instalado; el día que el paquete
`claude-agent-sdk` esté en el entorno, es escribir otra clase con el mismo
método. El agente no cambia.

Tres decisiones que gobiernan el módulo:

1. **Los encargos trabajan desde la carpeta del usuario, y de ahí no salen.** La
   raíz configurada (`PERSEO_DEV_RAIZ`, por defecto `C:\\Users\\<quien>`) más el
   **Escritorio** —que puede estar redirigido fuera del perfil por OneDrive—.
   Se pueden añadir más por entorno sin tocar código:
   `PERSEO_DEV_RAICES_EXTRA="C:\\una\\ruta;C:\\otra"`. Una ruta que se sale de
   todas se rechaza antes de arrancar nada. El motivo es el de siempre: lo que
   Perseo lee viene de correos y de pantallas, y desde la Fase D un correo puede
   acabar convertido en trabajo.

   Hasta el 2026-08-26 la raíz era este repositorio, y el proyecto nombrado en
   el encargo —«en Armario, arregla X»— **encerraba** al agente en la carpeta de
   ese proyecto. El señor Persus lo tachó: sus proyectos se llaman unos a otros
   y un agente encerrado en uno no puede leer el de al lado. Ahora el proyecto
   nombrado se le **cuenta** (`_contexto_del_encargo`) en vez de servirle de
   jaula, y la carpeta solo se estrecha si el encargo la pide expresamente.
2. **Editar sí, destruir no.** Se aceptan las ediciones sin preguntar —son
   reversibles, están en git, y es lo que dice §7 del plan— pero hay una lista de
   lo que no se ejecuta ni preguntando: `git push`, borrados, formateos.

   Lo que sí puede ejecutar depende de **quién pidió el encargo**: el señor
   Persus con el dedo (`origen` `texto` o `voz`) trabaja con
   `HERRAMIENTAS_PERMITIDAS_AMPLIAS` —`Bash` y subagentes incluidos, porque un
   agente que no puede arrancar un proceso no puede «abrir la app» y decía que
   sí—; lo que nace solo, de un correo o un disparador, se queda con la
   lista corta.
3. **`dev` tiene su propio carril.** Un encargo de código tarda minutos; el
   correo y la memoria tardan milisegundos. Los atiende un trabajador aparte para
   que uno no deje al otro esperando. Ver `Trabajador` en `agentes.py`.
4. **Todo lo que hace queda apuntado.** Cada herramienta, cada resultado y cada
   subagente van a la bitácora (`Paso`, `actividad_de`), en memoria y en
   `<datos>/actividad/<id>.jsonl`. No se borra al terminar el encargo: la
   pregunta «dice HECHO, ¿pero qué hizo?» solo se hace después.


"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from collections.abc import Callable
from typing import Any, Protocol

from . import almacen, proyectos
from .agentes import registrar

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
#: (`RealTime/src/components/Panel.tsx`, `perseo_core/interfaz/index.html`) y el
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


class MotorSdk:
    """Claude por el **Agent SDK oficial** (`claude-agent-sdk`), no por la consola.

    Es el mismo bucle de agente que `MotorClaude`, con la diferencia que se
    nota usándolo: los mensajes llegan **según pasan**, así que se puede contar
    por dónde va el encargo en vez de enseñar una barra girando durante seis
    minutos. Lo que se cuenta es la herramienta que acaba de usar —«Editando
    api.py», «Ejecutando pytest»—, que es la pregunta que uno se hace mirando.

    Lo demás es lo mismo y a propósito: las mismas listas de herramientas
    permitidas y denegadas, el mismo tope de vueltas y el mismo cerco de
    directorios resuelto antes de arrancar. El SDK no relaja ninguna decisión
    de seguridad; solo cambia por dónde se habla con el agente.

    `setting_sources=["project"]`: el encargo hereda el `CLAUDE.md` del
    proyecto donde trabaja —que es contexto útil— pero **no** los ajustes ni
    los hooks globales del usuario. Un agente que corre solo, de madrugada y
    sin nadie mirando no debe arrastrar la configuración de una sesión humana.
    """

    def __init__(self) -> None:
        # Se importa aquí y no arriba: el paquete es opcional (§18) y sin él
        # el núcleo tiene que arrancar igual, con el motor de consola.
        from claude_agent_sdk import ClaudeAgentOptions, query

        self._query = query
        self._Opciones = ClaudeAgentOptions

    async def ejecutar(self, encargo: Encargo, avisar: Aviso | None = None) -> Resultado:
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            SystemMessage,
            TextBlock,
            ThinkingBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
        )

        opciones = self._Opciones(
            cwd=str(encargo.raiz),
            permission_mode="acceptEdits",
            max_turns=MAX_VUELTAS,
            allowed_tools=list(encargo.permitidas),
            disallowed_tools=list(HERRAMIENTAS_DENEGADAS),
            setting_sources=["project"],
            resume=encargo.sesion or None,
            model=encargo.modelo or None,
            add_dirs=[str(c) for c in encargo.carpetas_extra],
            # Sin esto, lo que dice un subagente se queda dentro del subagente y
            # el panel enseña «Repartiendo a un subagente» durante cinco minutos
            # sin más. Con esto se puede entrar a ver qué hace cada uno, que es
            # lo que el señor Persus pidió para poder depurar (2026-08-26).
            forward_subagent_text=True,
            # NO es un adorno. Sin el preset, el SDK arranca al agente SIN el
            # preámbulo de Claude Code —el que le dice en qué directorio está
            # trabajando— y el modelo se inventa rutas absolutas: el mismo
            # encargo escribió en la carpeta del usuario, en la de otro usuario y en la
            # raíz del repositorio, tres veces seguidas y ninguna donde tocaba.
            # Con el preset, el fichero cae exactamente en `cwd`.
            system_prompt={"type": "preset", "preset": "claude_code"},
        )

        def contar(paso: Paso) -> None:
            if avisar is not None:
                avisar(paso)

        # Quién es cada subagente. La clave es el `tool_use_id` del `Task` que
        # lo lanzó, que es lo que luego llega como `parent_tool_use_id` en todo
        # lo que ese subagente hace: así se le pone nombre a la columna en vez
        # de un identificador de veinte letras.
        nombres: dict[str, str] = {}

        def quien(mensaje: Any) -> str:
            padre = getattr(mensaje, "parent_tool_use_id", None)
            return str(padre) if padre else "principal"

        texto_suelto: list[str] = []
        final: Any = None
        try:
            async with asyncio.timeout(encargo.tope):
                async for mensaje in self._query(prompt=encargo.prompt, options=opciones):
                    if isinstance(mensaje, AssistantMessage):
                        agente = quien(mensaje)
                        for bloque in mensaje.content:
                            if isinstance(bloque, ToolUseBlock):
                                if bloque.name == "Task":
                                    nombres[str(bloque.id)] = _nombre_de_subagente(bloque.input)
                                    contar(Paso(
                                        tipo="subagente",
                                        titulo=nombres[str(bloque.id)],
                                        detalle=_recortar(_texto_de_entrada(bloque.input), 1500),
                                        agente=str(bloque.id),
                                    ))
                                contar(Paso(
                                    tipo="herramienta",
                                    titulo=_contar_herramienta(bloque.name, bloque.input),
                                    detalle=_recortar(_texto_de_entrada(bloque.input), 1500),
                                    agente=agente,
                                ))
                            elif isinstance(bloque, TextBlock):
                                if agente == "principal":
                                    texto_suelto.append(bloque.text)
                                contar(Paso(
                                    tipo="dice",
                                    titulo=_recortar(bloque.text, 200),
                                    detalle=_recortar(bloque.text, 2000),
                                    agente=agente,
                                ))
                            elif isinstance(bloque, ThinkingBlock):
                                # Un pensamiento sin texto es la firma cifrada y
                                # nada más: apuntarlo llena la bitácora de líneas
                                # en blanco, que es peor que no apuntarlo.
                                pensado = str(getattr(bloque, "thinking", "") or "")
                                if pensado.strip():
                                    contar(Paso(
                                        tipo="piensa",
                                        titulo=_recortar(pensado, 200),
                                        detalle=_recortar(pensado, 2000),
                                        agente=agente,
                                    ))
                    elif isinstance(mensaje, UserMessage):
                        # Lo que CONTESTÓ cada herramienta. Es la mitad que
                        # faltaba para depurar: un encargo que acaba en verde
                        # habiendo fallado seis órdenes lo dice aquí y en
                        # ningún otro sitio.
                        agente = quien(mensaje)
                        contenido = mensaje.content
                        if isinstance(contenido, list):
                            for bloque in contenido:
                                if isinstance(bloque, ToolResultBlock):
                                    fallo = bool(getattr(bloque, "is_error", False))
                                    salida = _texto_de_resultado(bloque.content)
                                    contar(Paso(
                                        tipo="resultado",
                                        titulo=_recortar(salida, 200) or ("Error" if fallo else "ok"),
                                        detalle=_recortar(salida, 2000),
                                        agente=agente,
                                        ok=not fallo,
                                    ))
                    elif isinstance(mensaje, SystemMessage):
                        paso = _paso_de_sistema(mensaje, nombres)
                        if paso is not None:
                            contar(paso)
                    elif isinstance(mensaje, ResultMessage):
                        final = mensaje
        except TimeoutError:
            contar(Paso(tipo="error", titulo=f"Cortado a los {encargo.tope:.0f} s", ok=False))
            raise TimeoutError(f"El encargo pasó de {encargo.tope:.0f} s y se cortó.") from None

        if final is None:
            # El SDK terminó sin dar resultado: pasa si el proceso muere solo.
            unido = "\n".join(texto_suelto).strip()
            contar(Paso(
                tipo="fin",
                titulo="Terminó sin resultado del SDK",
                detalle=_recortar(unido, 2000),
                ok=bool(unido),
            ))
            return Resultado(texto=unido or "El agente terminó sin decir nada.", ok=bool(unido))

        contar(Paso(
            tipo="fin",
            titulo=f"Terminado en {int(final.num_turns or 0)} vueltas",
            detalle=_recortar(str(final.result or ""), 2000),
            ok=not final.is_error,
        ))
        return Resultado(
            texto=str(final.result or "\n".join(texto_suelto))[:4000],
            ok=not final.is_error,
            vueltas=int(final.num_turns or 0),
            sesion=str(final.session_id or ""),
        )


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


def _contar_herramienta(nombre: str, entrada: dict[str, Any]) -> str:
    """Una línea corta y en cristiano de lo que el agente acaba de hacer."""
    verbo = _COMO_SE_CUENTA.get(nombre, nombre)
    detalle = ""
    for clave in ("file_path", "path", "pattern", "command", "description"):
        valor = entrada.get(clave) if isinstance(entrada, dict) else None
        if valor:
            detalle = Path(str(valor)).name if clave in ("file_path", "path") else str(valor)
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


def hay_sdk() -> bool:
    """¿Está instalado el Agent SDK? Es opcional: sin él se habla por consola."""
    return importlib.util.find_spec("claude_agent_sdk") is not None


def abrir_motor(cfg: almacen.Configuracion) -> Motor | None:
    """Devuelve el motor configurado, o `None` si no hay ninguno utilizable.

    Sin `PERSEO_DEV_MOTOR` manda **opencode**: es el que no gasta suscripción,
    y el señor Persus lo pidió con todas las letras el 2026-08-26 —«que de una
    puñetera vez se use opencode de principal»—. Sabe contar por dónde va el
    encargo igual que el SDK (`--format json`), así que la razón por la que
    antes ganaba el SDK ya no existe.

    Si opencode no está instalado, el SDK; y si tampoco, la consola de Claude.
    """
    if cfg.dev_motor == "falso":
        return MotorFalso(tardanza=float(cfg.dev_tardanza_falsa))

    if cfg.dev_motor == "sdk":
        if not hay_sdk():
            logger.warning(
                "PERSEO_DEV_MOTOR=sdk pero no está `claude-agent-sdk` en el "
                "entorno; se sigue con la consola de Claude. `pip install "
                "claude-agent-sdk`."
            )
        else:
            return MotorSdk()

    if cfg.dev_motor == "opencode":
        ejecutable = shutil.which("opencode")
        if ejecutable is None:
            logger.warning(
                "PERSEO_DEV_MOTOR=opencode pero no se encuentra %r en el PATH; "
                "el agente `dev` fallará hasta que esté.",
                "opencode",
            )
            return None
        return MotorOpencode(ejecutable)

    if not cfg.dev_motor:
        ejecutable = shutil.which("opencode")
        if ejecutable is not None:
            return MotorOpencode(ejecutable)
        if hay_sdk():
            return MotorSdk()

    ejecutable = shutil.which(cfg.dev_ejecutable)
    if ejecutable is None:
        logger.warning(
            "No se encuentra %r en el PATH; el agente `dev` fallará hasta que esté.",
            cfg.dev_ejecutable,
        )
        return None
    return MotorClaude(ejecutable)


def _escritorio() -> Path | None:
    """La carpeta de Escritorio DE VERDAD, preguntándoselo a Windows.

    «~/Desktop» se queda corto: un Windows español con OneDrive redirige la
    carpeta y la llama «Escritorio», y lo que vale no es la adivinanza sino la
    API de carpetas conocidas. Si algo falla, `None` y punto: es un extra del
    cerco, nunca una pieza de la que dependa nada.
    """
    if os.name != "nt":
        return None
    try:
        import ctypes

        # FOLDERID_Desktop {B4BFCC3A-DB2C-424C-B029-7FE99A87C641}. A mano
        # porque `ctypes.wintypes` no trae GUID hasta Python 3.12.
        class _GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", ctypes.c_ulong),
                ("Data2", ctypes.c_ushort),
                ("Data3", ctypes.c_ushort),
                ("Data4", ctypes.c_ubyte * 8),
            ]

        identificador = _GUID(
            0xB4BFCC3A,
            0xDB2C,
            0x424C,
            (ctypes.c_ubyte * 8)(0xB0, 0x29, 0x7F, 0xE9, 0x9A, 0x87, 0xC6, 0x41),
        )
        salida = ctypes.c_wchar_p()
        shell32 = ctypes.WinDLL("shell32", use_last_error=True)
        shell32.SHGetKnownFolderPath.argtypes = (
            ctypes.POINTER(_GUID),
            ctypes.c_ulong,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_wchar_p),
        )
        if shell32.SHGetKnownFolderPath(
            ctypes.byref(identificador), 0, None, ctypes.byref(salida)
        ) != 0:
            return None
        ruta = Path(str(salida.value))
        memoria = ctypes.cast(salida, ctypes.c_void_p)
        if memoria.value:
            ctypes.WinDLL("ole32").CoTaskMemFree(memoria)
        return ruta
    except Exception:  # noqa: BLE001 — sin Escritorio conocido, se vive sin él
        return None


def raices_permitidas(raiz: Path) -> tuple[Path, ...]:
    """El cerco entero: la raíz configurada, el PERFIL DEL USUARIO y los extras.

    El perfil entero (`C:\\Users\\<quien>`) lo mandó el señor Persus el
    2026-08-24 —*«que tenga permiso para trabajar en todo usuario»—: sus
    proyectos viven repartidos por la carpeta personal (MAGI, armario…) y
    andar listándolos a mano era el impuesto de cada encargo. El Escritorio
    sigue añadiéndose aparte porque OneDrive puede redirigirlo FUERA del
    perfil. Los extras siguen por entorno (`PERSEO_DEV_RAICES_EXTRA`,
    separada por `;`) para lo que esté fuera de `C:\\Users`.
    """
    rutas: list[Path] = [raiz, Path.home()]
    for crudo in os.environ.get("PERSEO_DEV_RAICES_EXTRA", "").split(";"):
        if crudo.strip():
            rutas.append(Path(crudo.strip()).expanduser())
    escritorio = _escritorio()
    if escritorio is not None:
        rutas.append(escritorio)

    resueltas: list[Path] = []
    for ruta in rutas:
        resuelta = ruta.resolve()
        if resuelta not in resueltas:
            resueltas.append(resuelta)
    return tuple(resueltas)


# --------------------------------------------------------------------------- #
# El encargo en lenguaje natural
# --------------------------------------------------------------------------- #

#: El motor y el proyecto se DICEN en el propio encargo — «en Armario, añade
#: un README, con opencode» — y el núcleo los entiende aquí. La cara manda el
#: texto tal cual: las caras no piensan, que para eso está el núcleo.
_MOTOR_DEL_TEXTO = re.compile(
    r"\b(?:con|usando|usa|vía|via)\s+(opencode|claude)\b", re.IGNORECASE
)


def _motor_del_texto(texto: str) -> str:
    """El motor pedido en el encargo, o vacío si no se dijo ninguno."""
    coincidencia = _MOTOR_DEL_TEXTO.search(texto)
    return coincidencia.group(1).lower() if coincidencia else ""


def _parece_carpeta(candidato: str) -> bool:
    """¿Esto es una carpeta, o una dirección web disfrazada?

    Existe por lo que le pasó al señor Persus el 2026-08-25 estando fuera de
    casa: la lista de proyectos del móvil ofrecía «Armario · App» y mandaba su
    `destino`, que en modo servicio es `http://127.0.0.1:8000`. El encargo
    moría con «'http://127.0.0.1:8000' no es un directorio» antes de arrancar
    nada, y en la pantalla solo se veía FALLIDO sin decir por qué.
    """
    texto = (candidato or "").strip()
    return bool(texto) and "://" not in texto


def _carpeta_del_proyecto(proyecto: proyectos.Proyecto) -> str:
    """Dónde trabaja un agente en cada modo: la carpeta, nunca la URL.

    Se prueban los candidatos en orden y se devuelve el primero que sea una
    ruta de verdad. Un proyecto que solo sabe decir dónde MIRARSE —una URL— no
    tiene carpeta, y eso es un vacío, no un error.
    """
    if proyecto.modo == "arranque":
        candidatos = [proyecto.carpeta]
    elif proyecto.modo == "servicio":
        candidatos = [str(s.get("carpeta", "")) for s in proyecto.servidores]
        candidatos.append(proyecto.carpeta)
    else:
        candidatos = [proyecto.destino, proyecto.carpeta]
    for candidato in candidatos:
        if _parece_carpeta(candidato):
            return candidato
    return ""


def _contexto_del_encargo(instruccion: str, raiz: Path) -> str:
    """Lo que el agente tiene que saber antes de leer el encargo.

    Tres cosas, y las tres salieron de fallos reales:

    1. **Desde dónde trabaja y que puede moverse.** La carpeta del usuario
       entera, porque los proyectos del señor Persus se llaman unos a otros y
       encerrar al agente en uno era el impuesto de cada encargo (2026-08-26).
    2. **Dónde vive el proyecto que se nombra**, si se nombra alguno. Es la
       ruta absoluta, que es lo que evita que el modelo se la invente.
    3. **Que no dé por hecho lo que no hizo.** «Abre la app de armario» acabó
       en verde dos veces sin abrir nada (2026-08-25): un encargo que no se
       puede cumplir se dice, no se aprueba.
    """
    lineas = [
        f"Trabajas desde {raiz}, la carpeta personal del señor Persus. Sus "
        "proyectos cuelgan de ahí y se llaman unos a otros, así que puedes "
        "entrar en cualquiera de ellos; usa rutas absolutas.",
    ]
    if _datos is not None:
        carpeta = _proyecto_del_texto(instruccion, _datos)
        if carpeta:
            lineas.append(f"El proyecto que nombra el encargo está en {carpeta}.")
    lineas.append(
        "Si no puedes cumplir el encargo con las herramientas que tienes, dilo "
        "claramente y explica qué te falta. No lo des por hecho."
    )
    return " ".join(lineas)


def _proyecto_del_texto(texto: str, datos: Path) -> str:
    """La carpeta del proyecto nombrado en el encargo, o vacío.

    Se busca el nombre de ficha («Armario», «CVScraper») como palabra entera
    — que «perseo» no salte dentro de «perseverar». El primero que aparezca
    en la lista gana; nombrar dos proyectos en un encargo es dos encargos.
    """
    minusculas = texto.lower()
    for proyecto in proyectos.listar(datos):
        ficha = proyecto.nombre.lower()
        token = re.split(r"[·—–-]", ficha, maxsplit=1)[0].strip()
        if token and re.search(rf"(?<!\w){re.escape(token)}(?!\w)", minusculas):
            return _carpeta_del_proyecto(proyecto)
    return ""


# --------------------------------------------------------------------------- #
# El agente
# --------------------------------------------------------------------------- #

_motor: Motor | None = None
_raiz: Path | None = None
_raices: tuple[Path, ...] = ()
_tope: float = 900.0
#: Dónde viven `proyectos.json` y compañía: hace falta para entender el
#: proyecto nombrado en un encago en lenguaje natural.
_datos: Path | None = None
#: El ejecutable de Claude, por si el entorno le cambió el nombre. Se fija en
#: `iniciar` para que la elección POR ENCARGO (peticion.motor) lo respete.
_ejecutable_claude: str = "claude"

#: Por dónde va cada encargo vivo, para que el panel enseñe algo mejor que una
#: barra girando. En memoria a propósito: un encargo en curso no sobrevive a un
#: reinicio del núcleo —`recuperar_huerfanos` lo devuelve a la cola—, así que
#: guardar esto en disco sería conservar una frase que ya no es verdad.
_progreso: dict[int, str] = {}

#: Cuántos pasos se guardan por encargo en memoria, y cuántos encargos se
#: recuerdan. El tope existe porque la salida de un `pytest` largo cabe entera
#: en un paso, y cuarenta encargos con seiscientos pasos ya son memoria del
#: núcleo que no se recupera hasta reiniciar.
TOPE_PASOS = 600
TOPE_ENCARGOS_RECORDADOS = 40

#: Cuántos ficheros de bitácora se dejan en disco. Es un registro de depuración,
#: no un archivo histórico.
TOPE_BITACORAS_EN_DISCO = 200

#: La bitácora de cada encargo: todo lo que hicieron el agente y sus subagentes.
#: A diferencia de `_progreso`, esto NO se borra al terminar: la pregunta que
#: hubo que contestar el 2026-08-25 —«dice HECHO, pero ¿qué hizo?»— solo se
#: contesta después, y con el paso a paso delante.
_bitacoras: dict[int, list[Paso]] = {}


def _carpeta_bitacoras() -> Path | None:
    """Dónde se escriben las bitácoras, o `None` si no hay dónde.

    En disco además de en memoria porque el núcleo se reinicia y la pregunta
    «¿qué hizo el encargo de anoche?» sigue en pie a la mañana siguiente.
    """
    if _datos is None:
        return None
    carpeta = Path(_datos) / "actividad"
    try:
        carpeta.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return carpeta


def _anotar(id_trabajo: int, paso: Paso) -> None:
    """Apunta un paso: en memoria para el panel, en disco para mañana.

    Un fallo escribiendo no puede tumbar el encargo: la bitácora es para mirar,
    no una pieza de la que dependa el trabajo.
    """
    id_trabajo = int(id_trabajo)
    if not id_trabajo:
        return
    sellado = replace(paso, momento=datetime.now().isoformat(timespec="seconds"))

    if paso.titulo:
        _progreso[id_trabajo] = paso.titulo

    pasos = _bitacoras.setdefault(id_trabajo, [])
    pasos.append(sellado)
    if len(pasos) > TOPE_PASOS:
        del pasos[: len(pasos) - TOPE_PASOS]
    while len(_bitacoras) > TOPE_ENCARGOS_RECORDADOS:
        _bitacoras.pop(next(iter(_bitacoras)))

    carpeta = _carpeta_bitacoras()
    if carpeta is None:
        return
    try:
        with (carpeta / f"{id_trabajo}.jsonl").open("a", encoding="utf-8") as fichero:
            fichero.write(json.dumps(sellado.a_dict(), ensure_ascii=False) + "\n")
    except OSError as e:  # noqa: PERF203 — apuntar no puede romper el encargo
        logger.debug("No se pudo escribir la bitácora de #%d: %s", id_trabajo, e)


def _leer_bitacora(id_trabajo: int) -> list[Paso]:
    """La bitácora de un encargo que ya no está en memoria, leída del disco."""
    carpeta = _carpeta_bitacoras()
    if carpeta is None:
        return []
    fichero = carpeta / f"{int(id_trabajo)}.jsonl"
    if not fichero.is_file():
        return []
    pasos: list[Paso] = []
    try:
        for linea in fichero.read_text(encoding="utf-8").splitlines():
            if not linea.strip():
                continue
            try:
                crudo = json.loads(linea)
            except json.JSONDecodeError:
                continue
            pasos.append(
                Paso(
                    tipo=str(crudo.get("tipo", "")),
                    titulo=str(crudo.get("titulo", "")),
                    detalle=str(crudo.get("detalle", "")),
                    agente=str(crudo.get("agente", "principal")),
                    ok=bool(crudo.get("ok", True)),
                    momento=str(crudo.get("momento", "")),
                )
            )
    except OSError:
        return []
    return pasos[-TOPE_PASOS:]


def _limpiar_bitacoras_viejas() -> None:
    """Deja en disco solo las últimas. Se llama al arrancar, una vez."""
    carpeta = _carpeta_bitacoras()
    if carpeta is None:
        return
    try:
        ficheros = sorted(carpeta.glob("*.jsonl"), key=lambda f: f.stat().st_mtime)
    except OSError:
        return
    for viejo in ficheros[:-TOPE_BITACORAS_EN_DISCO]:
        with contextlib.suppress(OSError):
            viejo.unlink()


def progreso_de(id_trabajo: int) -> str:
    """Lo último que se sabe de un encargo en curso. Vacío si no hay nada."""
    return _progreso.get(int(id_trabajo), "")


def actividad_de(id_trabajo: int) -> dict[str, Any]:
    """Todo lo que hizo un encargo, suyo y de sus subagentes.

    Devuelve los pasos en orden y, aparte, la lista de quiénes trabajaron —el
    principal y cada subagente— con su nombre y cuántos pasos dio cada uno.
    Con eso la pantalla puede enseñar el encargo entero o meterse dentro de un
    subagente concreto, que es de lo que se trata para depurar.
    """
    id_trabajo = int(id_trabajo)
    pasos = _bitacoras.get(id_trabajo)
    if pasos is None:
        pasos = _leer_bitacora(id_trabajo)

    agentes: dict[str, dict[str, Any]] = {}
    for paso in pasos:
        ficha = agentes.setdefault(
            paso.agente,
            {
                "id": paso.agente,
                "titulo": "Agente principal" if paso.agente == "principal" else "",
                "pasos": 0,
                "fallos": 0,
            },
        )
        ficha["pasos"] += 1
        if not paso.ok:
            ficha["fallos"] += 1
        if paso.tipo == "subagente" and not ficha["titulo"]:
            ficha["titulo"] = paso.titulo
    for identificador, ficha in agentes.items():
        if not ficha["titulo"]:
            ficha["titulo"] = f"Subagente {identificador[:8]}"

    return {
        "id": id_trabajo,
        "vivo": id_trabajo in _progreso,
        "pasos": [p.a_dict() for p in pasos],
        "agentes": list(agentes.values()),
    }


def iniciar(cfg: almacen.Configuracion) -> Motor | None:
    global _motor, _raiz, _raices, _tope, _datos, _ejecutable_claude
    _raiz = Path(cfg.dev_raiz).resolve()
    _raices = raices_permitidas(_raiz)
    _tope = float(cfg.dev_tope)
    _datos = Path(cfg.directorio_datos)
    _ejecutable_claude = str(cfg.dev_ejecutable or "claude")
    _limpiar_bitacoras_viejas()
    if _motor is None:
        _motor = abrir_motor(cfg)
        if _motor is not None:
            logger.info("Agente dev listo sobre %s (tope %.0f s).", list(map(str, _raices)), _tope)
    return _motor


def detener() -> None:
    global _motor, _raices, _datos
    _motor = None
    _raices = ()
    _datos = None
    _progreso.clear()


def _motor_de(nombre: str) -> Motor | None:
    """Un motor para UN encargo, por nombre. `None` si no está instalado.

    Los nombres son los de `PERSEO_DEV_MOTOR`: `claude`, `opencode` y `falso`.
    Vacío no llega aquí — lo filtra quien llama, que usa el motor por defecto.
    """
    nombre = nombre.strip().lower()
    if nombre == "falso":
        return MotorFalso()
    if nombre == "sdk":
        return MotorSdk() if hay_sdk() else None
    if nombre == "opencode":
        ejecutable = shutil.which("opencode")
        return MotorOpencode(ejecutable) if ejecutable else None
    if nombre in ("claude", "anthropic"):
        ejecutable = shutil.which(_ejecutable_claude)
        return MotorClaude(ejecutable) if ejecutable else None
    return None


def resolver_raiz(pedida: str) -> Path:
    """Comprueba que el encargo se queda dentro de las raíces permitidas.

    Se resuelve antes de comparar: comparar cadenas sin resolver es exactamente
    como se cuela un `..`. Es la misma regla que en `memoria.py`, y por el mismo
    motivo — lo que llega puede venir de un correo.
    """
    if _raiz is None or not _raices:
        raise RuntimeError("El agente dev no está iniciado; falta llamar a dev.iniciar().")
    if not pedida:
        return _raiz

    # Unir una ruta absoluta a `_raiz` da la absoluta tal cual (pathlib), así que
    # la misma comparación vale para relativas —caen dentro o fuera— y para el
    # Escritorio u otra raíz que llegue ya absoluta.
    destino = (_raiz / pedida).expanduser().resolve()
    if not any(destino == raiz or raiz in destino.parents for raiz in _raices):
        listadas = ", ".join(str(raiz) for raiz in _raices)
        raise ValueError(f"{pedida!r} cae fuera de las raíces permitidas ({listadas}).")
    if not destino.is_dir():
        raise ValueError(f"{pedida!r} no es un directorio.")
    return destino


@registrar("dev")
async def _dev(trabajo: dict[str, Any]) -> dict[str, Any]:
    """Le encarga a un agente de código una tarea y devuelve lo que contestó.

    El motor se elige por encargo (`peticion.motor`: `claude` u `opencode`);
    sin elegir, manda el configurado al arrancar. No pide confirmación: editar
    código es reversible y está en git, que es lo que dice §7 del plan. Lo
    irreversible —publicar, borrar— no está en la lista de lo que puede
    ejecutar.
    """
    peticion = trabajo.get("peticion") or {}
    instruccion = str(peticion.get("texto") or peticion.get("instruccion") or "").strip()
    if not instruccion:
        raise ValueError("Un encargo de `dev` necesita `texto`.")

    # El motor: explícito en la petición, dicho en el texto («con opencode»),
    # o el configurado al arrancar. En ese orden.
    pedido = str(peticion.get("motor") or "").strip().lower() or _motor_del_texto(instruccion)
    if pedido:
        motor = _motor_de(pedido)
        if motor is None:
            raise ValueError(
                f"El motor {pedido!r} no está disponible ahora mismo: o no está "
                "instalado su ejecutable en el PATH, o el nombre no es ninguno "
                "de los conocidos (sdk, claude, opencode, falso)."
            )
    elif _motor is not None:
        motor = _motor
    else:
        raise RuntimeError(
            "No hay motor para `dev`. Instala Claude Code u opencode y "
            "comprueba su ejecutable en el PATH, pon PERSEO_DEV_MOTOR=falso "
            "para probar el circuito, o elige motor en cada encargo."
        )

    # El directorio: la RAÍZ, salvo que el encargo pida otra cosa a propósito.
    #
    # Antes se adivinaba del texto —«en Armario…» encerraba al agente en la
    # carpeta de Armario— y el señor Persus lo tachó el 2026-08-26: sus
    # proyectos se llaman unos a otros, y un agente encerrado en uno no puede
    # mirar el de al lado. Ahora el proyecto nombrado se le CUENTA al agente
    # (ver `_contexto_del_encargo`) en vez de servirle de jaula.
    directorio_pedido = str(peticion.get("directorio", "")).strip()
    if not _parece_carpeta(directorio_pedido):
        # Una URL en `directorio` es lo que mandaba la lista del móvil: se
        # ignora en vez de tumbar el encargo con «no es un directorio».
        if directorio_pedido:
            logger.info("Se ignora %r como directorio: no es una carpeta.", directorio_pedido)
        directorio_pedido = ""
    raiz = resolver_raiz(directorio_pedido)
    sesion = str(peticion.get("sesion", ""))
    modelo = str(peticion.get("modelo", "")).strip()

    # Las manos que se le dan dependen de quién lo pidió: el señor Persus con
    # el dedo, o algo que Perseo leyó. Ver `HERRAMIENTAS_PERMITIDAS_AMPLIAS`.
    origen = str(trabajo.get("origen") or "texto")
    permitidas = (
        HERRAMIENTAS_PERMITIDAS_AMPLIAS
        if origen in ORIGENES_DE_CONFIANZA
        else HERRAMIENTAS_PERMITIDAS
    )

    logger.info(
        "Encargo de dev (%s%s) en %s: %.120s",
        type(motor).__name__.replace("Motor", "").lower() or pedido or "configurado",
        f", {modelo}" if modelo else "",
        raiz,
        instruccion,
    )
    id_trabajo = int(trabajo.get("id") or 0)

    def contar(paso: Paso) -> None:
        _anotar(id_trabajo, paso)

    encargo = Encargo(
        instruccion=instruccion,
        raiz=raiz,
        tope=_tope,
        sesion=sesion,
        modelo=modelo,
        contexto=_contexto_del_encargo(instruccion, raiz),
        permitidas=permitidas,
        carpetas_extra=tuple(r for r in _raices if r != raiz),
    )

    try:
        resultado = await motor.ejecutar(encargo, avisar=contar)
    except Exception as e:
        _anotar(id_trabajo, Paso(tipo="error", titulo=str(e)[:120], detalle=str(e), ok=False))
        raise
    finally:
        _progreso.pop(id_trabajo, None)

    if not resultado.ok:
        raise RuntimeError(resultado.texto or "El encargo falló.")

    return {
        "texto": resultado.texto,
        "vueltas": resultado.vueltas,
        # Cuántos pasos quedaron apuntados: es la pista de que hay una bitácora
        # que mirar, y de si el encargo hizo algo o solo habló.
        "pasos": len(_bitacoras.get(id_trabajo, ())),
        # Con esto, un encargo siguiente puede continuar donde se quedó en vez de
        # empezar de cero: `peticion.sesion` con este valor.
        "sesion": resultado.sesion,
        "directorio": str(raiz),
        # Por Telegram, el titular; el texto entero se lee por el tailnet.
        "titular": (
            f"Encargo de código terminado "
            f"({resultado.vueltas} vuelta{'s' if resultado.vueltas != 1 else ''})"
        ),
    }
