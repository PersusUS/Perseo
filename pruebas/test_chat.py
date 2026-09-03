"""El chat escrito: la cabeza, el semáforo y las manos.

Sin red y sin modelo: lo que se prueba aquí son las decisiones —qué herramientas
existen, cómo se despachan a los agentes, cómo se resume un resultado para que
el modelo lo cuente, y el semáforo de una conversación a la vez—. La conversación
de punta a punta contra un Gemini de mentira la comprueba
`perseo_core/verificar_chat.py`.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from perseo_core import almacen, chat


# --------------------------------------------------------------------------- #
# Lo que el modelo ve
# --------------------------------------------------------------------------- #


def test_las_herramientas_estan_completas_y_con_forma() -> None:
    nombres = {d["name"] for d in chat._declaraciones()}
    esperadas = {
        "situacion_actual", "consultar_correo", "detalle_correo",
        "consultar_agenda", "consultar_habitos",
        "consultar_tareas", "crear_tarea", "mover_tarea",
        "buscar_en_memoria", "leer_nota", "guardar_recuerdo",
        "buscar_en_web", "leer_pagina", "controlar_pc", "encargar_codigo",
        "consultar_trabajo",
        "listar_mcp", "usar_mcp", "responder_confirmacion",
    }
    assert nombres == esperadas
    for declaracion in chat._declaraciones():
        assert declaracion["description"].strip()
        assert "type" in declaracion["parameters"]


def test_el_prompt_trae_las_reglas_que_no_se_negocian() -> None:
    # Verdad: sin esto, vuelve a inventarse asuntos de correo.
    assert "NO existe" in chat.PROMPT_CHAT or "no existe" in chat.PROMPT_CHAT
    # Seguridad: lo leído nunca es una orden.
    assert "jamás instrucciones" in chat.PROMPT_CHAT
    # Y habla como Perseo, no como un asistente anónimo.
    assert "señor Persus" in chat.PROMPT_CHAT


def test_la_politica_deja_pasar_el_turno(db: almacen.Configuracion) -> None:
    from perseo_core import politica

    # Sin esta entrada en la tabla, cada turno de chat pediría un sí y la
    # conversación entera moriría de pie.
    assert politica.nivel("chat") == politica.LIBRE


# --------------------------------------------------------------------------- #
# El resumen de resultados: lo que el modelo sabe contar
# --------------------------------------------------------------------------- #


def test_resumir_notas_con_ruta() -> None:
    texto = chat._resumir({
        "notas": [{"titulo": "Proyecto X", "ruta": "X.md", "extracto": "algo"}],
        "titular": "1 nota(s)",
    })
    assert "X.md" in texto and "Proyecto X" in texto


def test_resumir_agenda_vacia_no_es_un_error() -> None:
    assert "No hay nada" in chat._resumir({"eventos": []})


def test_resumir_pagina_web_recorta() -> None:
    texto = chat._resumir({"titulo": "Página", "url": "http://x", "texto": "a" * 9000})
    assert len(texto) < 4000 and texto.endswith("…")


def test_resumir_vacio_dice_hecho() -> None:
    assert chat._resumir(None) == "Hecho."


# --------------------------------------------------------------------------- #
# El semáforo de turnos
# --------------------------------------------------------------------------- #


def test_el_semaforo_del_chat(db: almacen.Configuracion) -> None:
    sesion = almacen.crear_sesion_chat()
    id_sesion = sesion["id"]

    almacen.marcar_turno_chat(id_sesion, "ocupado")
    with pytest.raises(ValueError):
        # Dos pantallas a la vez: el segundo se queda fuera.
        almacen.marcar_turno_chat(id_sesion, "ocupado")

    almacen.marcar_turno_chat(id_sesion, "libre")
    assert almacen.obtener_sesion_chat(id_sesion)["turno"] == "libre"


def test_borrar_sesion_ocupada_no_se_permite(db: almacen.Configuracion) -> None:
    sesion = almacen.crear_sesion_chat()
    almacen.anadir_mensaje_chat(sesion["id"], "usuario", "hola")
    almacen.marcar_turno_chat(sesion["id"], "ocupado")
    with pytest.raises(ValueError):
        almacen.borrar_sesion_chat(sesion["id"])
    almacen.marcar_turno_chat(sesion["id"], "libre")
    assert almacen.borrar_sesion_chat(sesion["id"]) is True
    assert almacen.obtener_sesion_chat(sesion["id"]) is None


def test_reiniciar_turnos_al_arrancar(db: almacen.Configuracion) -> None:
    """Un apagón no puede dejar una conversación ocupada para siempre."""
    sesion = almacen.crear_sesion_chat()
    almacen.marcar_turno_chat(sesion["id"], "ocupado")
    almacen.reiniciar_turnos_chat()
    assert almacen.obtener_sesion_chat(sesion["id"])["turno"] == "libre"


# --------------------------------------------------------------------------- #
# Los mensajes
# --------------------------------------------------------------------------- #


def test_los_mensajes_viajan_decodificados(db: almacen.Configuracion) -> None:
    sesion = almacen.crear_sesion_chat()  # sin título: lo pone el primer mensaje
    id_usuario = almacen.anadir_mensaje_chat(sesion["id"], "usuario", "¿qué hay?")
    id_perseo = almacen.anadir_mensaje_chat(sesion["id"], "perseo", "", "escribiendo")

    # El streaming escribe por trozos y al final firma sus herramientas.
    almacen.actualizar_mensaje_chat(id_perseo, texto="Todo tranquilo")
    almacen.actualizar_mensaje_chat(id_perseo, estado="hecho", herramientas=["situacion_actual"])

    mensajes = almacen.mensajes_chat(sesion["id"])
    assert [m["rol"] for m in mensajes] == ["usuario", "perseo"]
    assert mensajes[1]["herramientas"] == ["situacion_actual"]
    assert mensajes[1]["estado"] == "hecho"
    assert id_usuario < id_perseo

    # Y el título lo puso el primer mensaje, una sola vez.
    assert almacen.obtener_sesion_chat(sesion["id"])["titulo"] == "¿qué hay?"


def test_el_historial_salta_lo_vacio_y_lo_fallido(db: almacen.Configuracion) -> None:
    sesion = almacen.crear_sesion_chat()
    almacen.anadir_mensaje_chat(sesion["id"], "usuario", "uno")
    vacio = almacen.anadir_mensaje_chat(sesion["id"], "perseo", "", "escribiendo")
    almacen.anadir_mensaje_chat(sesion["id"], "perseo", "error de red", "fallido")
    almacen.anadir_mensaje_chat(sesion["id"], "perseo", "dos")

    contenidos = chat._historial(sesion["id"])
    textos = [c["parts"][0]["text"] for c in contenidos]
    assert "uno" in textos and "dos" in textos
    assert "" not in textos and "error de red" not in textos
    assert all(c["role"] in ("user", "model") for c in contenidos)
    del vacio


# --------------------------------------------------------------------------- #
# El despacho de herramientas
# --------------------------------------------------------------------------- #


@pytest.fixture()
def chat_listo(db: almacen.Configuracion, monkeypatch: pytest.MonkeyPatch):
    """El módulo con cfg puesta y las colas interceptadas.

    `_capturadas` recibe (agente, peticion) de cada herramienta que encole, así
    el despacho se comprueba sin tocar ningún agente de verdad.
    """
    capturadas: list[tuple[str, dict]] = []

    async def falsa_encola(agente: str, peticion: dict, espera: float = 30) -> str:
        capturadas.append((agente, peticion))
        return f"resultado de {agente}"

    monkeypatch.setattr(chat, "_cfg", db)
    monkeypatch.setattr(chat, "_encolar_y_esperar", falsa_encola)
    return capturadas


def test_despacho_consultar_agenda(chat_listo) -> None:
    respuesta = asyncio.run(chat._ejecutar_herramienta("consultar_agenda", {"horas": 12}))
    assert respuesta.startswith("resultado de agenda")
    agente, peticion = chat_listo[-1]
    assert (agente, peticion["accion"]) == ("agenda", "proximos")
    assert peticion["horas"] == 12.0


def test_despacho_pc_y_dev(chat_listo) -> None:
    asyncio.run(chat._ejecutar_herramienta(
        "controlar_pc", {"accion": "abrir_app", "parametro": "spotify"}
    ))
    assert chat_listo[-1][0] == "pc"

    asyncio.run(chat._ejecutar_herramienta(
        "encargar_codigo", {"texto": "arregla X", "directorio": "C:\\proy"}
    ))
    agente, peticion = chat_listo[-1]
    assert (agente, peticion["texto"]) == ("dev", "arregla X")


def test_despacho_usar_mcp_exige_servidor(chat_listo) -> None:
    with pytest.raises(chat.ErrorHerramienta):
        asyncio.run(chat._ejecutar_herramienta("usar_mcp", {"servidor": "", "herramienta": "x"}))


# --------------------------------------------------------------------------- #
# consultar_trabajo: el estado REAL de un encargo
#
# El 2026-08-24 un subagente terminó a los veinte segundos y el modelo llevaba
# la razón del señor Persus contestando «sigue en curso» desde memoria: ninguna
# herramienta sabía mirar un trabajo cerrado. Esto es lo que lo impide.
# --------------------------------------------------------------------------- #


def test_consultar_trabajo_cuenta_lo_hecho_con_su_resultado(db: almacen.Configuracion) -> None:
    trabajo = almacen.encolar("dev", {"texto": "crea una carpeta"}, "texto")
    almacen.completar(trabajo["id"], {"texto": "Carpeta creada en el escritorio.", "vueltas": 3})

    respuesta = chat._consultar_trabajo(trabajo["id"])
    assert "TERMINÓ" in respuesta
    assert "Carpeta creada en el escritorio." in respuesta


def test_consultar_trabajo_dice_el_fallo(db: almacen.Configuracion) -> None:
    trabajo = almacen.encolar("dev", {"texto": "algo"}, "texto")
    almacen.fallar(trabajo["id"], "la raíz no existe")

    respuesta = chat._consultar_trabajo(trabajo["id"])
    assert "FALLÓ" in respuesta and "la raíz no existe" in respuesta


def test_consultar_trabajo_desconocido_no_es_un_error(db: almacen.Configuracion) -> None:
    respuesta = chat._consultar_trabajo(99999)
    assert "No veo ningún trabajo" in respuesta


def test_consultar_trabajo_sin_id_lista_los_encargos(db: almacen.Configuracion) -> None:
    dev = almacen.encolar("dev", {"texto": "encargo de código"}, "voz")
    almacen.completar(dev["id"], {"titular": "Encargo de código terminado (2 vueltas)"})
    turno = almacen.encolar("chat", {"sesion": 1, "mensaje": 1, "texto": "hola"}, "texto")
    almacen.completar(turno["id"], {"turno": "completado"})

    respuesta = chat._consultar_trabajo(None)
    assert "#%d" % dev["id"] in respuesta and "terminado" in respuesta.lower()
    # Los turnos del propio chat son ruido aquí: no salen.
    assert "#%d" % turno["id"] not in respuesta


def test_despacho_correo_lee_la_base_de_verdad(
    chat_listo, db: almacen.Configuracion
) -> None:
    # Un trabajo de correo hecho, como los deja `almacen.completar`: la lectura
    # del chat debe devolver ESTE asunto y no otro.
    conexion = __import__("sqlite3").connect(db.ruta_db)
    try:
        conexion.executescript(almacen._ESQUEMA)
        peticion = {"mensajes": [{
            "id": "m-1", "remitente": "a@b.es", "asunto": "Asunto real",
            "extracto": "cuerpo", "fecha": "2026-08-25",
        }]}
        resultado = {"clasificados": [{
            "id": "m-1", "remitente": "a@b.es", "asunto": "Asunto real",
            "clase": "requiere_accion", "motivo": "pide pago",
        }]}
        conexion.execute(
            "INSERT INTO trabajos (estado, agente, origen, peticion, resultado, creado_en, actualizado_en)"
            " VALUES ('hecho','correo','disparador',?,?,?,?)",
            (json.dumps(peticion), json.dumps(resultado), "2026-08-25T10:00:00", "2026-08-25T10:00:00"),
        )
        conexion.commit()
    finally:
        conexion.close()

    texto = asyncio.run(chat._ejecutar_herramienta("consultar_correo", {}))
    assert "Asunto real" in texto and "requiere acción" in texto

    detalle = asyncio.run(chat._ejecutar_herramienta("detalle_correo", {"id_mensaje": "m-1"}))
    assert "cuerpo" in detalle


# --------------------------------------------------------------------------- #
# La cascada de modelos: cada uno tiene su propio cubo de cuota (2026-08-24)
# --------------------------------------------------------------------------- #


class _RespuestaFalsa:
    """Lo justo de una respuesta de aiohttp para este camino."""

    def __init__(self, status: int, lineas: list[bytes] | None = None) -> None:
        self.status = status
        self.content = _Cuerpo(lineas or [])

    async def text(self) -> str:
        return "{}"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False


class _Cuerpo:
    def __init__(self, lineas: list[bytes]) -> None:
        self._lineas = lineas

    def __aiter__(self):
        async def gen():
            for linea in self._lineas:
                yield linea

        return gen()


class _SesionFalsa:
    """Contesta 429 a los modelos que se le digan y 200 al resto, apuntando
    a quién se llamó y en qué orden."""

    def __init__(self, sin_cuota: set[str], respuesta: str = "ok") -> None:
        self.sin_cuota = sin_cuota
        self.respuesta = respuesta
        self.llamados: list[str] = []

    def post(self, url, params=None, json=None):
        modelo = url.split("/models/")[1].split(":")[0]
        self.llamados.append(modelo)
        if modelo in self.sin_cuota:
            return _RespuestaFalsa(429)
        cuerpo = {"candidates": [{"content": {"parts": [{"text": self.respuesta}]}}]}
        import json as _json

        return _RespuestaFalsa(200, [b"data: " + _json.dumps(cuerpo).encode()])


def _turno(sesion, monkeypatch) -> list[dict]:
    monkeypatch.setattr(chat.almacen, "apuntar_uso", lambda *a, **k: None)

    async def correr():
        return [e async for e in chat._llamar_modelo(sesion, "clave", [])]

    return asyncio.run(correr())


# ── La firma del pensamiento (2026-08-25) ─────────────────────────────────── #


class _SesionConHerramienta:
    """Contesta con una llamada a herramienta firmada, como Gemini 2.5."""

    def __init__(self, firma: str = "FIRMA-123") -> None:
        self.firma = firma

    def post(self, url, params=None, json=None):
        import json as _json

        parte = {"functionCall": {"name": "consultar_trabajo", "args": {"id": 7}}}
        if self.firma:
            parte["thoughtSignature"] = self.firma
        cuerpo = {"candidates": [{"content": {"parts": [parte]}}]}
        return _RespuestaFalsa(200, [b"data: " + _json.dumps(cuerpo).encode()])


def test_la_firma_de_la_llamada_se_recoge(monkeypatch) -> None:
    """Sin recogerla no hay forma de devolverla, y Gemini contesta 400."""
    eventos = _turno(_SesionConHerramienta(), monkeypatch)
    assert eventos[-1]["llamadas"][0]["firma"] == "FIRMA-123"


def test_la_firma_vuelve_pegada_a_la_llamada() -> None:
    """«Function call is missing a thought_signature in functionCall parts»."""
    parte = chat._parte_de_llamada(
        {"nombre": "consultar_trabajo", "argumentos": {"id": 7}, "firma": "FIRMA-123"}
    )
    assert parte["thoughtSignature"] == "FIRMA-123"
    assert parte["functionCall"] == {"name": "consultar_trabajo", "args": {"id": 7}}


def test_sin_firma_la_parte_va_limpia() -> None:
    """Los modelos que no firman no pueden recibir una firma vacía."""
    parte = chat._parte_de_llamada({"nombre": "x", "argumentos": {}, "firma": ""})
    assert "thoughtSignature" not in parte


def test_aplanar_convierte_las_herramientas_en_prosa() -> None:
    """El plan B: un turno degradado es mejor que un turno perdido."""
    aplanado = chat._aplanar_herramientas([
        {"role": "user", "parts": [{"text": "hola"}]},
        {"role": "model", "parts": [{"functionCall": {"name": "ver", "args": {"a": 1}}}]},
        {"role": "user", "parts": [{"functionResponse": {"name": "ver", "response": {"result": "ok"}}}]},
    ])
    assert all("functionCall" not in p and "functionResponse" not in p
               for turno in aplanado for p in turno["parts"])
    assert "ver" in aplanado[1]["parts"][0]["text"]
    assert "ok" in aplanado[2]["parts"][0]["text"]
    assert aplanado[0]["parts"][0] == {"text": "hola"}


def test_sin_cuota_y_peticion_rechazada_no_se_cuentan_igual() -> None:
    """Esperar arregla lo primero y no arregla lo segundo (H-73)."""
    sin_cuota = chat._por_que_no_hubo_modelo(
        chat.ErrorGemini("Ningún modelo del chat tiene cuota ahora mismo: a, b")
    )
    rechazo = chat._por_que_no_hubo_modelo(
        chat.ErrorGemini("Gemini respondió 400: Function call is missing a thought_signature")
    )
    assert "cuota" in sin_cuota
    assert "esperar no lo arregla" in rechazo
    assert "thought_signature" in rechazo


def test_por_defecto_hay_dos_modelos_y_los_dos_dan_500_al_dia() -> None:
    """Un cubo por modelo: dos hermanos son 1.000 peticiones al día, no 500."""
    assert len(chat.MODELOS_POR_DEFECTO) >= 2
    assert all("flash-lite" in m for m in chat.MODELOS_POR_DEFECTO)


def test_si_el_primero_no_tiene_cuota_contesta_el_segundo(monkeypatch) -> None:
    sesion = _SesionFalsa(sin_cuota={chat.MODELOS_POR_DEFECTO[0]})
    eventos = _turno(sesion, monkeypatch)
    assert sesion.llamados == list(chat.MODELOS_POR_DEFECTO[:2])
    assert eventos[-1]["texto"] == "ok"


def test_sin_429_no_se_molesta_al_segundo(monkeypatch) -> None:
    sesion = _SesionFalsa(sin_cuota=set())
    _turno(sesion, monkeypatch)
    assert sesion.llamados == [chat.MODELOS_POR_DEFECTO[0]]


def test_cambiar_de_modelo_no_hace_esperar(monkeypatch) -> None:
    """La espera es para cuando NO queda ninguno; cambiar de cubo es gratis."""
    dormido: list[float] = []

    async def falso_sleep(segundos):
        dormido.append(segundos)

    monkeypatch.setattr(chat.asyncio, "sleep", falso_sleep)
    _turno(_SesionFalsa(sin_cuota={chat.MODELOS_POR_DEFECTO[0]}), monkeypatch)
    assert dormido == []


def test_si_ninguno_tiene_cuota_se_dice_claro(monkeypatch) -> None:
    async def falso_sleep(segundos):
        return None

    monkeypatch.setattr(chat.asyncio, "sleep", falso_sleep)
    sesion = _SesionFalsa(sin_cuota=set(chat.MODELOS_POR_DEFECTO))
    with pytest.raises(chat.ErrorGemini, match="cuota"):
        _turno(sesion, monkeypatch)


def test_la_variable_de_entorno_acepta_una_lista(monkeypatch) -> None:
    monkeypatch.setenv("PERSEO_CHAT_MODELO", "uno, dos ,tres")
    assert chat._modelos() == ("uno", "dos", "tres")


def test_un_modelo_suelto_sigue_valiendo(monkeypatch) -> None:
    monkeypatch.setenv("PERSEO_CHAT_MODELO", "solo-este")
    assert chat._modelos() == ("solo-este",)
    assert chat._modelo() == "solo-este"
