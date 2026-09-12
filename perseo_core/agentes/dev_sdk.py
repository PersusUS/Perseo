"""El motor del agente `dev` sobre el SDK oficial de Claude Code.

Es el único de los cuatro que **cuenta por dónde va**: los que hablan por línea
de órdenes no dicen nada hasta el final, y un encargo de diez minutos sin una
sola señal se vive como un encargo colgado. Por eso está aparte: esa capacidad
le cuesta doscientas líneas que no tienen que ver con los otros tres.

Lo que comparte con ellos —los tipos, el cerco de herramientas y la lectura de
lo que escupen— vive en `dev_motores.py`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .dev_motores import (
    Aviso,
    HERRAMIENTAS_DENEGADAS,
    MAX_VUELTAS,
    Encargo,
    Paso,
    Resultado,
    _contar_herramienta,
    _nombre_de_subagente,
    _paso_de_sistema,
    _recortar,
    _texto_de_entrada,
    _texto_de_resultado,
)

logger = logging.getLogger(__name__)


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
