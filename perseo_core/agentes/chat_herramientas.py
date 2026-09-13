"""Lo que el chat escrito hace cuando el modelo pide una herramienta.

Salió de `chat.py` el 2026-09-12, cuando el fichero pasaba de mil líneas y se le
veían dos trabajos: **sostener la conversación** —el bucle de turno, la llamada
al modelo, el historial— y **atender lo que el modelo pide**, que es esto.

La forma de atender es la misma que en la llamada de voz y a propósito: se
encola un trabajo y se espera su resultado. El chat no ejecuta nada por su
cuenta; quien decide qué se puede hacer es la política, y quien lo hace es el
agente que toque. Por eso casi todo aquí abajo termina en `_encolar_y_esperar`.

Qué herramientas existen no se decide aquí: se declara una sola vez en
`servicios/catalogo.py`, y hay una prueba que comprueba que lo declarado es
exactamente lo que este fichero sabe despachar.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..infra import almacen, politica
from ..infra.configuracion import Configuracion
from ..servicios import correo_lectura, habitos, tareas, triaje

logger = logging.getLogger(__name__)


class ErrorHerramienta(RuntimeError):
    """Una herramienta falló. El error viaja al modelo, que decide cómo contarlo."""


#: La configuración, puesta al arrancar. Este módulo tiene la suya en vez de
#: mirar la de `chat.py`: una capa no lee las globales de otra, y así el módulo
#: se puede probar solo.
_cfg: Configuracion | None = None


def iniciar(cfg: Configuracion) -> None:
    global _cfg
    _cfg = cfg


# --------------------------------------------------------------------------- #
# Ejecución de herramientas: encolar y esperar, como hace la voz
# --------------------------------------------------------------------------- #


async def _encolar_y_esperar(agente: str, peticion: dict[str, Any], espera: float = 30) -> str:
    """Encola el trabajo y espera su resultado, con el mismo criterio que la app
    de voz: hecho → resumen; esperando → se lo dices al modelo para que pregunte
    el sí; pasado el plazo → sigue en marcha, que no es un fallo."""
    trabajo = await asyncio.to_thread(almacen.encolar, agente, peticion, "texto")
    id_trabajo = int(trabajo["id"])
    limite = asyncio.get_running_loop().time() + espera
    while True:
        actual = await asyncio.to_thread(almacen.obtener, id_trabajo)
        if actual is not None:
            estado = str(actual.get("estado"))
            if estado == almacen.HECHO:
                return _resumir(actual.get("resultado"))
            if estado == almacen.ESPERANDO:
                pregunta = ((actual.get("confirmacion") or {}).get("resumen")) or "una confirmación"
                return (
                    f"PENDIENTE DE CONFIRMACIÓN (trabajo #{id_trabajo}): {pregunta} "
                    "Pregúntaselo al señor Persus por escrito y, con su respuesta literal, "
                    "llama a responder_confirmacion con ese número."
                )
            if estado in (almacen.FALLIDO, almacen.CANCELADO, almacen.RECHAZADO):
                raise ErrorHerramienta(
                    f"El trabajo #{id_trabajo} quedó {estado}: {actual.get('error') or 'sin detalle'}"
                )
        if asyncio.get_running_loop().time() >= limite:
            return (
                f"SIGUE EN MARCHA (trabajo #{id_trabajo}): se está trabajando en ello. "
                "Díselo tal cual, sin dar nada por hecho; cuando se pregunte de nuevo "
                "cómo va, mira el estado real con consultar_trabajo usando ese número."
            )
        await asyncio.sleep(0.4)


def _recortar(texto: str, tope: int) -> str:
    limpio = texto if len(texto) <= tope else texto[: tope - 1].rstrip() + "…"
    return limpio


def _resumir(resultado: Any) -> str:
    """El resultado de un agente, en texto que el modelo sepa contar. Las mismas
    ramas que `resumir` en Rust: si aquí y allí divergen, la voz y el chat
    contarán cosas distintas del mismo trabajo."""
    if resultado is None or resultado == {}:
        return "Hecho."
    if isinstance(resultado, str):
        return resultado

    if resultado.get("titulo") and resultado.get("texto"):
        return f"{resultado['titulo']}: {_recortar(str(resultado['texto']), 3500)}"
    if isinstance(resultado.get("notas"), list):
        notas = resultado["notas"]
        if not notas:
            return "No hay ninguna nota sobre eso en el vault."
        lineas = [
            f"- {n.get('titulo', '(sin título)')} (ruta: {n.get('ruta', '?')}): {n.get('extracto', '')}"
            for n in notas
        ]
        return f"{len(notas)} nota(s):\n" + "\n".join(lineas)
    if isinstance(resultado.get("eventos"), list):
        eventos = resultado["eventos"]
        if not eventos:
            return "No hay nada en la agenda para ese plazo."
        return "\n".join(f"- {e.get('titulo', '(sin título)')}: {str(e.get('inicio', ''))[:16]}" for e in eventos)
    if isinstance(resultado.get("resultados"), list):
        hallazgos = resultado["resultados"]
        if not hallazgos:
            return "La búsqueda no devolvió nada."
        lineas = [f"- {p.get('titulo', '(sin título)')} — {p.get('url', '')}" for p in hallazgos]
        return f"{len(hallazgos)} resultado(s):\n" + "\n".join(lineas)
    if isinstance(resultado.get("contenido"), str):
        return _recortar(resultado["contenido"], 4000)
    if resultado.get("texto"):
        return str(resultado["texto"])
    if resultado.get("titular"):
        return str(resultado["titular"])
    if resultado.get("ruta"):
        return f"Guardado en {resultado['ruta']}."
    pares = [
        f"{k}: {v}" for k, v in resultado.items()
        if v is not None and v != "" and not isinstance(v, (dict, list))
    ]
    return " · ".join(pares) if pares else "Hecho."


# -- Las herramientas que no pasan por la cola ------------------------------- #


def _situacion_actual() -> str:
    """El briefing de la voz, traducido a Python. Lee la cola y la presencia
    directamente: es el núcleo mirándose a sí mismo."""
    trabajos = almacen.listar(None, 15)
    partes: list[str] = []
    # Los turnos de ESTE chat son trabajos también, y el que está corriendo
    # ahora mismo aparece como «en_curso»: contarle al señor Persus que
    # «trabaja en "¿de verdad lo hiciste?"» no dice nada. Lo laboral es lo
    # demás.
    laborables = [t for t in trabajos if t["agente"] != "chat"]
    en_curso = next((t for t in laborables if t["estado"] == almacen.EN_CURSO), None)
    if en_curso is not None:
        peticion = en_curso.get("peticion") or {}
        que = peticion.get("texto") or peticion.get("consulta") or peticion.get("titulo") or ""
        partes.append(
            f"ahora mismo trabaja en '{_recortar(str(que), 80)}' ({en_curso['agente']})"
            if que else f"ahora mismo trabaja en un asunto de '{en_curso['agente']}'"
        )
    esperando = [t for t in laborables if t["estado"] == almacen.ESPERANDO]
    if esperando:
        preguntas = "; ".join(
            f"#{t['id']} {((t.get('confirmacion') or {}).get('resumen')) or 'una confirmación'}"
            for t in esperando
        )
        partes.append(f"esperan tu sí: {preguntas}")
    fallido = next((t for t in laborables if t["estado"] == almacen.FALLIDO), None)
    if fallido is not None:
        primera = str(fallido.get("error") or "sin detalle").splitlines()[0]
        partes.append(f"falló por última vez un asunto de '{fallido['agente']}': {_recortar(primera, 100)}")

    # El último encargo de código, aunque ya haya acabado. La cola solo cuenta
    # lo ABIERTO, así que un subagente terminado era invisible aquí: al señor
    # Persus se le contestó «sigue en curso» de un encargo llevaba minutos
    # hecho (2026-08-24). `listar` viene del más reciente hacia atrás.
    ultimo_dev = next((t for t in trabajos if t["agente"] == "dev"), None)
    if ultimo_dev is not None:
        if ultimo_dev["estado"] == almacen.HECHO:
            salida = _resumir(ultimo_dev.get("resultado")).replace("\n", " ")
            partes.append(
                f"tu último encargo de código (#{ultimo_dev['id']}) TERMINÓ: {_recortar(salida, 160)}"
                if salida != "Hecho."
                else f"tu último encargo de código (#{ultimo_dev['id']}) terminó."
            )
        elif ultimo_dev["estado"] in (almacen.EN_CURSO, almacen.PENDIENTE):
            partes.append(f"tu último encargo de código (#{ultimo_dev['id']}) sigue en marcha")

    presencia_correo = _correo_por_cajones(trabajos)
    if presencia_correo:
        nombres = {"requiere_accion": "piden acción", "no_seguro": "sin decidir", "interesante": "son interesantes"}
        for clase, cuantos in presencia_correo.items():
            partes.append(f"el buzón tiene {cuantos} correo(s) que {nombres.get(clase, clase)}")
    else:
        partes.append("el buzón está al día")

    try:
        import psutil  # noqa: PLC0415

        bateria = psutil.sensors_battery()
        if bateria is not None:
            partes.append(
                f"la batería va al {round(bateria.percent)}% "
                + ("(enchufada)" if bateria.power_plugged else "sin enchufar")
            )
    except Exception:  # noqa: BLE001 — la batería es un extra, nunca una pieza
        pass

    return ". ".join(partes) + "." if partes else "Todo tranquilo: nada en marcha y el buzón al día."


def _correo_por_cajones(trabajos: list[dict[str, Any]]) -> dict[str, int]:
    """El recuento del buzón sin resolver, con el criterio de `triaje`."""
    try:
        marcados = almacen.correos_marcados()
    except Exception:  # noqa: BLE001
        marcados = {}
    return triaje.pendientes_por_cajon(trabajos, marcados)


def _consultar_trabajo(id_crudo: Any) -> str:
    """El estado de un trabajo, o los últimos encargos si no dan id.

    Nace de una escena real (2026-08-24): un encargo de código terminó a los
    veinte segundos, pero ninguna herramienta sabía contarlo — `situacion_actual`
    solo mira lo abierto — y el modelo llevaba la razón del señor Persus
    contestando «sigue en curso» desde memoria. Ahora el que pregunta recibe lo
    que de verdad pasó."""
    trabajos = almacen.listar(None, 50)

    if id_crudo is not None and str(id_crudo).strip() != "":
        try:
            id_trabajo = int(id_crudo)
        except (TypeError, ValueError):
            raise ErrorHerramienta(f"Ese identificador no es un número de trabajo: {id_crudo!r}")
        actual = next((t for t in trabajos if t["id"] == id_trabajo), None)
        if actual is None:
            # Puede ser de antes del tope de 50 o de otra vida del núcleo; no es
            # un error: se dice y se ofrece lo que sí se ve.
            recientes = ", ".join(f"#{t['id']}" for t in trabajos[:8]) or "(ninguno)"
            return (
                f"No veo ningún trabajo #{id_trabajo} en los recientes ({recientes}). "
                "Si era de hace mucho, ya no está en la cola."
            )
        estado = str(actual.get("estado"))
        if estado == almacen.HECHO:
            salida = _resumir(actual.get("resultado"))
            return f"El trabajo #{id_trabajo} TERMINÓ ({actual['agente']}). Resultado: {salida}"
        if estado == almacen.FALLIDO:
            error = str(actual.get("error") or "sin detalle").splitlines()[0]
            return f"El trabajo #{id_trabajo} FALLÓ ({actual['agente']}): {_recortar(error, 300)}"
        if estado == almacen.ESPERANDO:
            pregunta = ((actual.get("confirmacion") or {}).get("resumen")) or "una confirmación"
            return (
                f"El trabajo #{id_trabajo} espera un sí tuyo: {pregunta} "
                "Pregúntaselo por escrito y, con su respuesta literal, llama a "
                "responder_confirmacion con ese número."
            )
        if estado == almacen.EN_CURSO:
            peticion = actual.get("peticion") or {}
            que = peticion.get("texto") or peticion.get("consulta") or ""
            detalle = f": {_recortar(str(que), 80)}" if que else ""
            return f"El trabajo #{id_trabajo} sigue EN CURSO ({actual['agente']}){detalle}. Avisa de que no ha terminado."
        return f"El trabajo #{id_trabajo} está {estado}."

    # Sin id: los últimos encargos, excluyendo los turnos de este mismo chat —
    # son ruido para quien pregunta por su equipo, no por la conversación.
    lineas: list[str] = []
    for t in trabajos:
        if len(lineas) >= 8:
            break
        if t["agente"] == "chat":
            continue
        peticion = t.get("peticion") or {}
        que = str(peticion.get("texto") or peticion.get("consulta") or peticion.get("accion") or "").strip()
        resumen = _recortar(que.replace("\n", " "), 60) if que else "(sin detalle)"
        extra = ""
        if t["estado"] == almacen.HECHO:
            resultado = t.get("resultado")
            titular = resultado.get("titular") if isinstance(resultado, dict) else None
            texto = resultado.get("texto") if isinstance(resultado, dict) else None
            salida = str(titular or texto or "").strip().replace("\n", " ")
            extra = f" → {_recortar(salida, 100)}" if salida else ""
        elif t["estado"] == almacen.FALLIDO:
            error = str(t.get("error") or "sin detalle").splitlines()[0]
            extra = f" → ERROR: {_recortar(error.replace(chr(10), ' '), 80)}"
        lineas.append(f"#{t['id']} [{t['estado']}] {t['agente']}: {resumen}{extra}")
    return "\n".join(lineas) if lineas else "No hay ningún encargo en la cola reciente."


async def _ejecutar_herramienta(nombre: str, argumentos: dict[str, Any]) -> str:
    """El despacho. Lo directo va directo (lecturas del propio núcleo); lo que
    tiene manos, encola y espera, con la política delante por el camino normal."""
    cfg = _cfg
    assert cfg is not None
    argumentos = argumentos or {}

    if nombre == "situacion_actual":
        return _situacion_actual()

    if nombre == "consultar_correo":
        limite = argumentos.get("limite") if isinstance(argumentos.get("limite"), (int, float)) else 15
        return correo_lectura.correos_triados(cfg.ruta_db, int(limite), str(argumentos.get("clase", "") or ""))

    if nombre == "detalle_correo":
        return correo_lectura.detalle_correo(cfg.ruta_db, str(argumentos.get("id_mensaje", "")))

    if nombre == "consultar_habitos":
        # Se lee del espejo en disco y no se encola: es un fichero de dos
        # kilobytes que ya está redactado. Ver `habitos.py`.
        return await asyncio.to_thread(habitos.resumen, cfg.directorio_datos)

    if nombre == "consultar_tareas":
        # Lo mismo, y por lo mismo. Ver `tareas.py`.
        return await asyncio.to_thread(tareas.resumen, cfg.directorio_datos)

    if nombre in ("crear_tarea", "mover_tarea"):
        # No se escribe el tablero desde aquí: se le pide a la ventana, que es
        # su único escritor. Ver el punto 3 de la cabecera de `tareas.py`.
        accion = "crear" if nombre == "crear_tarea" else "mover"
        try:
            orden = await asyncio.to_thread(
                tareas.encolar,
                cfg.directorio_datos,
                accion,
                str(argumentos.get("titulo", "") or ""),
                columna=str(argumentos.get("columna", "") or ""),
                detalle=str(argumentos.get("detalle", "") or ""),
            )
        except ValueError as error:
            return f"No se ha pedido nada: {error}."
        destino = orden.get("columna")
        if accion == "crear":
            return (
                f"Pedido a la app: clavar «{orden['titulo']}»"
                + (f" en {destino}" if destino else "")
                + ". Se hará en cuanto la ventana esté abierta; si lo está, en segundos."
            )
        return (
            f"Pedido a la app: mover «{orden['titulo']}» a {destino}. Se hará en cuanto "
            "la ventana esté abierta; si lo está, en segundos. Si hay más de una nota "
            "con ese nombre no moverá ninguna."
        )

    if nombre == "consultar_agenda":
        peticion: dict[str, Any] = {"accion": "proximos"}
        horas = argumentos.get("horas")
        if isinstance(horas, (int, float)) and horas > 0:
            peticion["horas"] = float(horas)
        return await _encolar_y_esperar("agenda", peticion, 20)

    if nombre == "buscar_en_memoria":
        peticion_mem: dict[str, Any] = {"accion": "buscar", "texto": str(argumentos.get("texto", "") or "")}
        carpeta = str(argumentos.get("carpeta", "") or "").strip()
        if carpeta:
            peticion_mem["carpeta"] = carpeta
        return await _encolar_y_esperar("memoria", peticion_mem)
    if nombre == "leer_nota":
        return await _encolar_y_esperar("memoria", {"accion": "leer", "ruta": str(argumentos.get("ruta", "") or "")})
    if nombre == "guardar_recuerdo":
        entidad = str(argumentos.get("entidad", "") or "").strip()
        if not entidad:
            raise ErrorHerramienta("guardar_recuerdo necesita 'entidad'.")
        cuerpo = str(argumentos.get("contexto", "") or "").strip()
        visual = str(argumentos.get("descripcion_visual", "") or "").strip()
        if not cuerpo and not visual:
            raise ErrorHerramienta("Un recuerdo necesita contexto o descripción.")
        texto = cuerpo + ("\n\nDescripción visual: " + visual if visual else "")
        return await _encolar_y_esperar("memoria", {"accion": "anotar", "titulo": entidad, "texto": texto})

    if nombre == "buscar_en_web":
        consulta = str(argumentos.get("consulta", "") or argumentos.get("texto", "") or "").strip()
        if not consulta:
            raise ErrorHerramienta("Falta la consulta.")
        return await _encolar_y_esperar("web", {"accion": "buscar", "texto": consulta}, 30)
    if nombre == "leer_pagina":
        url = str(argumentos.get("url", "") or "").strip()
        if not url:
            raise ErrorHerramienta("Falta la URL.")
        return await _encolar_y_esperar("web", {"accion": "leer", "url": url}, 30)

    if nombre == "controlar_pc":
        accion = str(argumentos.get("accion", "") or "")
        parametro = str(argumentos.get("parametro", "") or "")
        if not accion:
            raise ErrorHerramienta("controlar_pc necesita 'accion'.")
        return await _encolar_y_esperar("pc", {"accion": accion, "parametro": parametro}, 25)

    if nombre == "encargar_codigo":
        texto = str(argumentos.get("texto", "") or "").strip()
        if not texto:
            raise ErrorHerramienta("encargar_codigo necesita la descripción del trabajo.")
        peticion_dev: dict[str, Any] = {"texto": texto}
        directorio = str(argumentos.get("directorio", "") or "").strip()
        if directorio:
            peticion_dev["directorio"] = directorio
        # Un subagente tarda segundos incluso para lo trivial (arrancar `claude`
        # ya se los come): esperar menos era contestar «en marcha» siempre. Con
        # 25 s, los encargos cortos mueren dentro de la espera y los largos
        # quedan consultables con consultar_trabajo.
        return await _encolar_y_esperar("dev", peticion_dev, 25)

    if nombre == "consultar_trabajo":
        return _consultar_trabajo(argumentos.get("id"))

    if nombre == "listar_mcp":
        return await _encolar_y_esperar("mcp", {"accion": "servidores"}, 240)
    if nombre == "usar_mcp":
        servidor = str(argumentos.get("servidor", "") or "").strip()
        herramienta = str(argumentos.get("herramienta", "") or "").strip()
        if not servidor or not herramienta:
            raise ErrorHerramienta("usar_mcp necesita 'servidor' y 'herramienta'.")
        return await _encolar_y_esperar(
            "mcp",
            {
                "accion": "llamar",
                "servidor": servidor,
                "herramienta": herramienta,
                "argumentos": argumentos.get("argumentos") or {},
            },
        )

    if nombre == "responder_confirmacion":
        return await _resolver_confirmacion(argumentos)

    raise ErrorHerramienta(f"Herramienta desconocida: {nombre}")


async def _resolver_confirmacion(argumentos: dict[str, Any]) -> str:
    """El sí hablado de la voz, pero por escrito. Resuelve y espera el resultado,
    porque lo siguiente que dirá el señor Persus es «¿y?»."""
    crudo = argumentos.get("id")
    try:
        id_trabajo = int(crudo)
    except (TypeError, ValueError):
        raise ErrorHerramienta(f"Ese identificador no es un número de trabajo: {crudo!r}")
    decision = str(argumentos.get("decision", "") or "")
    if decision not in ("aprobar", "rechazar"):
        raise ErrorHerramienta(f"Decisión desconocida: {decision!r}")

    aprobado = decision == "aprobar"
    if aprobado:
        # Lo crítico no lo aprueba el modelo. Esta herramienta es el «sí» que
        # Perseo dice haber oído, y un modelo puede creer que lo oyó: el
        # 2026-08-27 anunció una confirmación que nadie le había pedido y dio
        # el comando por autorizado. Para lo que no se deshace, el sí lo pone
        # una persona en la tarjeta del panel.
        pendiente = await asyncio.to_thread(almacen.obtener, id_trabajo)
        if pendiente is not None and politica.nivel(
            str(pendiente.get("agente") or ""), pendiente.get("peticion")
        ) == politica.CRITICO:
            return (
                f"El trabajo #{id_trabajo} no se puede aprobar hablando: no se "
                "puede deshacer. Dile que lo confirme él mismo en la tarjeta del "
                "panel, y no lo des por hecho hasta verlo."
            )
    resuelto = await asyncio.to_thread(almacen.resolver_confirmacion, id_trabajo, aprobado)
    if resuelto is None:
        actual = await asyncio.to_thread(almacen.obtener, id_trabajo)
        estado = actual["estado"] if actual else "inexistente"
        return f"Ese trabajo ya no espera confirmación (está {estado}). Díselo con naturalidad."

    if not aprobado:
        return f"Hecho: el trabajo #{id_trabajo} queda rechazado y no se ejecuta."

    async def esperar() -> str:
        limite = asyncio.get_running_loop().time() + 30
        while True:
            actual = await asyncio.to_thread(almacen.obtener, id_trabajo)
            if actual is not None:
                if actual["estado"] == almacen.HECHO:
                    return f"Hecho. Resultado: {_resumir(actual.get('resultado'))}"
                if actual["estado"] in (almacen.FALLIDO, almacen.CANCELADO, almacen.RECHAZADO):
                    return f"El trabajo #{id_trabajo} acabó {actual['estado']}: {actual.get('error') or ''}"
                if actual["estado"] == almacen.ESPERANDO:
                    pregunta = ((actual.get("confirmacion") or {}).get("resumen")) or "una confirmación"
                    return f"Aprobado y vuelto a parar: {pregunta}. Vuelve a preguntárselo."
            if asyncio.get_running_loop().time() >= limite:
                return f"Aprobado; sigue en marcha (trabajo #{id_trabajo}). Dilo así."
            await asyncio.sleep(0.4)

    return await esperar()
