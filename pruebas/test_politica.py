"""La política de §7: qué nivel tiene cada cosa y cuándo se pregunta."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from perseo_core.infra import identidad, politica


def test_leer_es_libre() -> None:
    assert politica.nivel("memoria", {"accion": "buscar"}) == politica.LIBRE
    assert politica.nivel("memoria", {"accion": "leer"}) == politica.LIBRE
    assert politica.nivel("web", {"accion": "leer"}) == politica.LIBRE
    assert politica.nivel("correo", {"accion": "triar"}) == politica.LIBRE


def test_escribir_es_reversible() -> None:
    assert politica.nivel("memoria", {"accion": "anotar"}) == politica.REVERSIBLE
    assert politica.nivel("dev", {"texto": "arregla esto"}) == politica.REVERSIBLE


def test_teclear_a_ciegas_es_irreversible() -> None:
    """Va sobre la ventana que tenga el foco, y esa puede ser cualquiera."""
    assert politica.nivel("pc", {"accion": "escribir_teclado"}) == politica.IRREVERSIBLE
    assert politica.nivel("pc", {"accion": "atajo_teclado"}) == politica.IRREVERSIBLE


def test_abrir_una_app_no_lo_es() -> None:
    assert politica.nivel("pc", {"accion": "abrir_app"}) == politica.LIBRE


def test_lo_que_no_esta_en_la_tabla_pregunta() -> None:
    """La decisión que sostiene el módulo entero."""
    assert politica.nivel("agente_del_futuro", {}) == politica.IRREVERSIBLE
    assert politica.nivel("memoria", {"accion": "borrar"}) == politica.IRREVERSIBLE
    assert politica.nivel("pc", {"accion": "formatear"}) == politica.IRREVERSIBLE


def test_el_simulacro_se_queda_fuera() -> None:
    """Lleva su propia confirmación dentro; con la política encima, dos."""
    assert politica.nivel("simulacro", {"accion": "lo que sea"}) == politica.LIBRE


def test_la_accion_manda_sobre_el_agente() -> None:
    assert politica.nivel("pc", {"accion": "abrir_app"}) != politica.nivel(
        "pc", {"accion": "escribir_teclado"}
    )


def test_sin_peticion_se_mira_el_agente_a_secas() -> None:
    assert politica.nivel("eco") == politica.LIBRE
    assert politica.nivel("dev") == politica.REVERSIBLE


def test_de_entrada_no_hay_confianza() -> None:
    assert not politica.hay_confianza()
    assert politica.confianza_hasta() is None


def test_la_confianza_baja_lo_irreversible() -> None:
    assert politica.pide_confirmacion("pc", {"accion": "escribir_teclado"})
    politica.activar_confianza(30)
    assert not politica.pide_confirmacion("pc", {"accion": "escribir_teclado"})


def test_la_confianza_no_tapa_lo_critico() -> None:
    """La llamada de voz enciende la confianza sola, y con ella nada preguntaba.

    Tener a alguien hablando no es su sí a ESTA orden: lo que no se deshace
    pregunta igual.
    """
    politica.registrar_niveles(lambda agente, peticion: politica.CRITICO)
    try:
        politica.activar_confianza(30)
        assert politica.hay_confianza()
        assert politica.pide_confirmacion("mcp", {"servidor": "windows"})
        assert "no se puede deshacer" in politica.resumir("mcp", {"accion": "llamar"})
    finally:
        politica.registrar_niveles(None)


def test_la_confianza_no_cambia_lo_libre() -> None:
    politica.activar_confianza(30)
    assert not politica.pide_confirmacion("memoria", {"accion": "buscar"})


def test_la_confianza_tiene_tope() -> None:
    hasta = politica.activar_confianza(99999)
    margen = hasta - datetime.now(timezone.utc)
    assert margen <= timedelta(minutes=politica.MAX_MINUTOS_CONFIANZA)


def test_pedir_cero_minutos_da_al_menos_uno() -> None:
    hasta = politica.activar_confianza(0)
    assert hasta > datetime.now(timezone.utc)


def test_una_confianza_caducada_no_vale(tmp_path: Path) -> None:
    politica.iniciar(tmp_path)
    caducada = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    (tmp_path / "confianza.txt").write_text(caducada, encoding="utf-8")

    assert not politica.hay_confianza()
    # Y el fichero se quita de en medio, para que el disco no mienta.
    assert not (tmp_path / "confianza.txt").exists()


def test_un_fichero_ilegible_se_trata_como_sin_confianza(tmp_path: Path) -> None:
    politica.iniciar(tmp_path)
    (tmp_path / "confianza.txt").write_text("esto no es una fecha", encoding="utf-8")
    assert not politica.hay_confianza()


def test_apagar_la_confianza() -> None:
    politica.activar_confianza(30)
    politica.desactivar_confianza()
    assert not politica.hay_confianza()


def test_el_resumen_dice_que_es_sin_soltar_el_detalle() -> None:
    """Sale por Telegram, donde solo va el titular."""
    resumen = politica.resumir("pc", {"accion": "escribir_teclado", "parametro": "mi contraseña"})
    assert "irreversible" in resumen
    assert "pc" in resumen and "escribir_teclado" in resumen
    assert "contraseña" not in resumen


# --------------------------------------------------------------------------- #
# Quién lo pide
# --------------------------------------------------------------------------- #


def test_el_dueno_se_reconoce_por_sus_alias() -> None:
    """El perfil puede llamarse «Persus» o «Jesús»: los dos son él."""
    assert identidad.es_el_dueno("Persus")
    assert identidad.es_el_dueno("jesus")
    assert not identidad.es_el_dueno("Javi")
    assert not identidad.es_el_dueno("Desconocido 2")
    # Sin nombre no es él, pero tampoco es una visita: ver `pide_confirmacion`.
    assert not identidad.es_el_dueno(None)


def test_una_visita_no_mueve_las_manos_ni_con_confianza(tmp_path: Path) -> None:
    """La lección de N-3, por el otro lado: tener a alguien delante hablando no
    es el sí del señor Persus, y menos si quien habla no es él."""
    politica.iniciar(tmp_path)
    politica.activar_confianza(30)
    try:
        teclear = {"accion": "escribir_teclado", "parametro": "rm -rf"}
        # Él, con confianza: pasa.
        assert not politica.pide_confirmacion("pc", teclear, "Persus")
        # Una visita, con la misma confianza encendida: para.
        assert politica.pide_confirmacion("pc", teclear, "Javi")
        assert politica.pide_confirmacion("pc", teclear, "Desconocido 1")
    finally:
        politica.desactivar_confianza()


def test_una_visita_tampoco_escribe_lo_reversible(tmp_path: Path) -> None:
    """Anotar en el vault es reversible para él; para una visita, no es suyo."""
    politica.iniciar(tmp_path)
    anotar = {"accion": "anotar", "texto": "lo que sea"}
    assert not politica.pide_confirmacion("memoria", anotar, "Persus")
    assert politica.pide_confirmacion("memoria", anotar, "Javi")


def test_leer_sigue_siendo_libre_para_cualquiera(tmp_path: Path) -> None:
    """Lo que no cambia nada no se para; qué se cuente delante de quién lo
    deciden las instrucciones de trato, no esta tabla."""
    politica.iniciar(tmp_path)
    assert not politica.pide_confirmacion("memoria", {"accion": "buscar"}, "Javi")
    assert not politica.pide_confirmacion("correo", {"accion": "triar"}, "Desconocido 3")


def test_sin_nombre_se_comporta_como_siempre(tmp_path: Path) -> None:
    """El panel y el chat escrito no mandan hablante, y el reconocimiento puede
    estar apagado: eso no puede convertir el sistema en un pedigüeño."""
    politica.iniciar(tmp_path)
    teclear = {"accion": "escribir_teclado", "parametro": "hola"}
    assert politica.pide_confirmacion("pc", teclear, None)
    politica.activar_confianza(30)
    try:
        assert not politica.pide_confirmacion("pc", teclear, None)
    finally:
        politica.desactivar_confianza()


def test_el_resumen_dice_quien_lo_pide() -> None:
    """Quién lo pidió ES la decisión, así que va en el titular."""
    resumen = politica.resumir("pc", {"accion": "escribir_teclado"}, "Javi")
    assert "Javi" in resumen
    # Y con él, la pregunta de siempre.
    assert "Javi" not in politica.resumir("pc", {"accion": "escribir_teclado"}, "Persus")


def test_toda_accion_de_un_agente_tiene_nivel_decidido() -> None:
    """El guardián de la cobertura.

    Lo que no está en la tabla es irreversible —decisión 1 de la cabecera—, y eso
    es lo correcto para un agente nuevo. Lo que NO puede pasar es que una acción
    que ya existe caiga ahí por olvido: se descubre en una llamada, con el
    trabajo parado esperando un sí que nadie sabe que hay que dar. Esta prueba
    falla el día que alguien añada una acción y no decida su nivel.
    """
    acciones_por_agente = {
        "memoria": ("buscar", "leer", "anotar", "conversacion"),
        "web": ("leer", "buscar"),
        "correo": ("triar", "redactar"),
        "agenda": ("avisar", "proximos"),
        "pc": (
            "abrir_app",
            "buscar_youtube",
            "volumen",
            "mover_raton",
            "click_raton",
            "escribir_teclado",
            "atajo_teclado",
        ),
        "mcp": ("servidores",),
    }
    sin_decidir = [
        f"{agente}.{accion}"
        for agente, acciones in acciones_por_agente.items()
        for accion in acciones
        if f"{agente}.{accion}" not in politica.TABLA and agente not in politica.TABLA
    ]
    assert not sin_decidir, f"Acciones sin nivel en la tabla: {sin_decidir}"


# --------------------------------------------------------------------------- #
# «Igual que el anterior»
# --------------------------------------------------------------------------- #


def test_un_si_vale_para_la_repeticion_exacta(tmp_path: Path) -> None:
    politica.iniciar(tmp_path)
    politica.olvidar_repeticiones()
    teclear = {"accion": "escribir_teclado", "parametro": "calle Mayor 3"}
    assert politica.pide_confirmacion("pc", teclear)

    politica.recordar_aprobacion("pc", teclear)
    assert not politica.pide_confirmacion("pc", teclear)
    # Mismos datos en otro orden siguen siendo lo mismo.
    assert not politica.pide_confirmacion(
        "pc", {"parametro": "calle Mayor 3", "accion": "escribir_teclado"}
    )
    politica.olvidar_repeticiones()


def test_una_orden_parecida_vuelve_a_preguntar(tmp_path: Path) -> None:
    """Estrecha a propósito: cambia una letra y es otra orden."""
    politica.iniciar(tmp_path)
    politica.olvidar_repeticiones()
    politica.recordar_aprobacion("pc", {"accion": "escribir_teclado", "parametro": "hola"})
    assert politica.pide_confirmacion("pc", {"accion": "escribir_teclado", "parametro": "hola "})
    assert politica.pide_confirmacion("pc", {"accion": "atajo_teclado", "parametro": "hola"})
    politica.olvidar_repeticiones()


def test_lo_critico_no_se_acumula(tmp_path: Path) -> None:
    politica.iniciar(tmp_path)
    politica.olvidar_repeticiones()
    borrar = {"accion": "borrar", "parametro": "C:/"}
    politica.TABLA["pruebas_criticas.borrar"] = politica.CRITICO
    try:
        politica.recordar_aprobacion("pruebas_criticas", borrar)
        assert politica.pide_confirmacion("pruebas_criticas", borrar)
    finally:
        politica.TABLA.pop("pruebas_criticas.borrar", None)
        politica.olvidar_repeticiones()


def test_el_si_de_una_visita_no_se_recuerda(tmp_path: Path) -> None:
    politica.iniciar(tmp_path)
    politica.olvidar_repeticiones()
    teclear = {"accion": "escribir_teclado", "parametro": "lo que sea"}
    politica.recordar_aprobacion("pc", teclear, "Javi")
    assert politica.pide_confirmacion("pc", teclear)
    politica.olvidar_repeticiones()


def test_apagar_la_confianza_olvida_los_sies(tmp_path: Path) -> None:
    """Apagarla es decir «vuelve a preguntármelo todo»."""
    politica.iniciar(tmp_path)
    teclear = {"accion": "escribir_teclado", "parametro": "hola"}
    politica.recordar_aprobacion("pc", teclear)
    politica.desactivar_confianza()
    assert politica.pide_confirmacion("pc", teclear)


def test_una_repeticion_caduca(tmp_path: Path) -> None:
    politica.iniciar(tmp_path)
    politica.olvidar_repeticiones()
    teclear = {"accion": "escribir_teclado", "parametro": "hola"}
    politica.recordar_aprobacion("pc", teclear)
    huella = politica._huella("pc", teclear)
    politica._repeticiones[huella] = datetime.now(timezone.utc) - timedelta(seconds=1)
    assert politica.pide_confirmacion("pc", teclear)
    # Y la huella caducada se va de la tabla al mirarla.
    assert huella not in politica._repeticiones
