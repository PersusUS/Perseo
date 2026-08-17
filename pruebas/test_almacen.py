"""La cola: encolar, reclamar, cerrar y sobrevivir a un reinicio."""

from __future__ import annotations

import pytest

from perseo_core import almacen


def test_encolar_nace_pendiente(db) -> None:
    trabajo = almacen.encolar("eco", {"texto": "hola"})
    assert trabajo["estado"] == almacen.PENDIENTE
    assert trabajo["agente"] == "eco"
    assert trabajo["peticion"] == {"texto": "hola"}
    assert trabajo["intentos"] == 0


def test_encolar_rechaza_un_origen_inventado(db) -> None:
    with pytest.raises(ValueError):
        almacen.encolar("eco", {}, origen="telepatia")


def test_reclamar_toma_el_mas_viejo_primero(db) -> None:
    primero = almacen.encolar("eco", {"n": 1})
    almacen.encolar("eco", {"n": 2})
    assert almacen.reclamar()["id"] == primero["id"]


def test_reclamar_marca_en_curso_y_cuenta_el_intento(db) -> None:
    almacen.encolar("eco", {})
    reclamado = almacen.reclamar()
    assert reclamado["estado"] == almacen.EN_CURSO
    assert reclamado["intentos"] == 1
    assert reclamado["reclamado_en"] is not None


def test_reclamar_no_devuelve_dos_veces_el_mismo(db) -> None:
    almacen.encolar("eco", {})
    assert almacen.reclamar() is not None
    assert almacen.reclamar() is None


def test_reclamar_filtra_por_agente(db) -> None:
    """Los dos carriles de la Fase E: `dev` va por su lado."""
    almacen.encolar("dev", {"texto": "largo"})
    almacen.encolar("eco", {"texto": "corto"})

    general = almacen.reclamar(excluir=("dev",))
    assert general["agente"] == "eco"

    solo_dev = almacen.reclamar(agentes=("dev",))
    assert solo_dev["agente"] == "dev"


def test_reclamar_excluyendo_no_ve_nada_si_solo_hay_de_ese(db) -> None:
    almacen.encolar("dev", {})
    assert almacen.reclamar(excluir=("dev",)) is None


def test_completar_guarda_el_resultado(db) -> None:
    trabajo = almacen.encolar("eco", {})
    hecho = almacen.completar(trabajo["id"], {"texto": "ya"})
    assert hecho["estado"] == almacen.HECHO
    assert hecho["resultado"] == {"texto": "ya"}


def test_fallar_guarda_el_error_y_no_el_resultado(db) -> None:
    trabajo = almacen.encolar("eco", {})
    fallido = almacen.fallar(trabajo["id"], "se rompio")
    assert fallido["estado"] == almacen.FALLIDO
    assert fallido["error"] == "se rompio"
    assert fallido["resultado"] is None


def test_recuperar_huerfanos_devuelve_los_en_curso(db) -> None:
    """Sin esto, un cierre inesperado los deja clavados para siempre."""
    almacen.encolar("eco", {})
    almacen.reclamar()

    assert almacen.recuperar_huerfanos() == 1
    assert almacen.reclamar() is not None


def test_recuperar_huerfanos_no_toca_los_que_esperan(db) -> None:
    """Un trabajo esperando un sí no es un huérfano: sigue esperando."""
    trabajo = almacen.encolar("simulacro", {})
    almacen.reclamar()
    almacen.pedir_confirmacion(trabajo["id"], "¿seguro?", "detalle")

    assert almacen.recuperar_huerfanos() == 0
    assert almacen.obtener(trabajo["id"])["estado"] == almacen.ESPERANDO


def test_pedir_confirmacion_guarda_la_pregunta(db) -> None:
    trabajo = almacen.encolar("simulacro", {})
    almacen.reclamar()
    esperando = almacen.pedir_confirmacion(trabajo["id"], "¿borro?", "37 ficheros")

    assert esperando["estado"] == almacen.ESPERANDO
    assert esperando["confirmacion"]["resumen"] == "¿borro?"
    assert esperando["confirmacion"]["detalle"] == "37 ficheros"
    assert esperando["confirmacion"]["decision"] is None
    # Deja de estar reclamado: si no, la recuperación de huérfanos lo barrería.
    assert esperando["reclamado_en"] is None


def test_aprobar_devuelve_el_trabajo_a_la_cola(db) -> None:
    trabajo = almacen.encolar("simulacro", {})
    almacen.reclamar()
    almacen.pedir_confirmacion(trabajo["id"], "¿seguro?")

    aprobado = almacen.resolver_confirmacion(trabajo["id"], True)
    assert aprobado["estado"] == almacen.PENDIENTE
    assert aprobado["confirmacion"]["decision"] == "aprobado"


def test_rechazar_lo_cierra_sin_ejecutarlo(db) -> None:
    trabajo = almacen.encolar("simulacro", {})
    almacen.reclamar()
    almacen.pedir_confirmacion(trabajo["id"], "¿seguro?")

    rechazado = almacen.resolver_confirmacion(trabajo["id"], False)
    assert rechazado["estado"] == almacen.RECHAZADO
    assert rechazado["resultado"] is None


def test_contestar_dos_veces_no_cuenta_dos_veces(db) -> None:
    """La misma pregunta puede estar abierta en la web y en Telegram."""
    trabajo = almacen.encolar("simulacro", {})
    almacen.reclamar()
    almacen.pedir_confirmacion(trabajo["id"], "¿seguro?")

    assert almacen.resolver_confirmacion(trabajo["id"], True) is not None
    assert almacen.resolver_confirmacion(trabajo["id"], False) is None


def test_listar_filtra_por_estado(db) -> None:
    almacen.encolar("eco", {"n": 1})
    segundo = almacen.encolar("eco", {"n": 2})
    almacen.completar(segundo["id"], {})

    assert len(almacen.listar(estado=almacen.PENDIENTE)) == 1
    assert len(almacen.listar(estado=almacen.HECHO)) == 1
    assert len(almacen.listar()) == 2


def test_recuento_por_estado(db) -> None:
    almacen.encolar("eco", {})
    hecho = almacen.encolar("eco", {})
    almacen.completar(hecho["id"], {})
    assert almacen.recuento_por_estado() == {almacen.PENDIENTE: 1, almacen.HECHO: 1}


def test_obtener_un_trabajo_que_no_existe(db) -> None:
    assert almacen.obtener(9999) is None


def test_las_fechas_van_en_utc_con_z(db) -> None:
    """Mezclar horas locales entre máquinas es una fuente de errores gratuita."""
    trabajo = almacen.encolar("eco", {})
    assert trabajo["creado_en"].endswith("Z")


def test_resolver_hosts_nunca_cae_en_todas_las_interfaces() -> None:
    """`tailscale` sin tailnet se queda en local, no abre `0.0.0.0`."""
    assert almacen._resolver_hosts("127.0.0.1") == ("127.0.0.1",)
    assert almacen._resolver_hosts("") == ("127.0.0.1",)
    assert "0.0.0.0" not in almacen._resolver_hosts("tailscale")


def test_resolver_hosts_no_repite() -> None:
    """aiohttp falla si se le repite una interfaz."""
    assert almacen._resolver_hosts("127.0.0.1, 127.0.0.1") == ("127.0.0.1",)


def test_url_por_defecto_prefiere_lo_que_no_es_local() -> None:
    """El enlace lo abre el móvil: `127.0.0.1` allí es el propio teléfono."""
    assert almacen._url_por_defecto(("127.0.0.1", "100.64.0.1"), 8787) == (
        "http://100.64.0.1:8787"
    )


def test_url_por_defecto_sin_tailnet_se_queda_en_local() -> None:
    """No hay nada mejor que ofrecer, pero el enlace no vale desde fuera."""
    assert almacen._url_por_defecto(("127.0.0.1",), 8787) == "http://127.0.0.1:8787"


def test_un_enlace_local_se_marca_como_inalcanzable(cfg) -> None:
    """Es lo que hace que el aviso salga en el registro en vez de en el móvil."""
    from dataclasses import replace

    assert not replace(cfg, url_base="http://127.0.0.1:8787").url_base_alcanzable
    assert not replace(cfg, url_base="http://localhost:8787").url_base_alcanzable


def test_un_enlace_del_tailnet_si_es_alcanzable(cfg) -> None:
    from dataclasses import replace

    assert replace(cfg, url_base="http://100.64.0.1:8787").url_base_alcanzable


# --------------------------------------------------------------------------- #
# Cuota de los servicios de fuera
# --------------------------------------------------------------------------- #


def test_el_uso_empieza_a_cero(db) -> None:
    assert almacen.uso_de_hoy() == {}


def test_apuntar_uso_suma(db) -> None:
    almacen.apuntar_uso("gemma-4-31b-it")
    almacen.apuntar_uso("gemma-4-31b-it")
    almacen.apuntar_uso("otro-modelo", 3)
    assert almacen.uso_de_hoy() == {"gemma-4-31b-it": 2, "otro-modelo": 3}


def test_apuntar_uso_sin_base_de_datos_no_lanza(cfg) -> None:
    """Llevar la cuenta no puede tumbar una clasificación de correo.

    Es el único sitio del sistema donde tragarse el error es lo correcto: lo que
    se pierde es un número informativo, y lo que se protege es el trabajo.
    """
    almacen.cerrar()
    almacen.apuntar_uso("gemma-4-31b-it")  # no debe lanzar


# --------------------------------------------------------------------------- #
# Correos triados: qué se ha hecho con cada uno
# --------------------------------------------------------------------------- #


def test_sin_marcar_nada_no_hay_correos(db) -> None:
    """Lo que no está en la tabla está pendiente: es el estado de casi todos."""
    assert almacen.correos_marcados() == {}


def test_marcar_un_correo_lo_deja_escrito(db) -> None:
    marcado = almacen.marcar_correo("msg-1", almacen.ATENDIDO)
    assert marcado["estado"] == almacen.ATENDIDO
    assert almacen.correos_marcados() == {"msg-1": almacen.ATENDIDO}


def test_marcar_dos_veces_no_duplica(db) -> None:
    """La misma pregunta puede estar abierta en la web y en el móvil."""
    almacen.marcar_correo("msg-1", almacen.ATENDIDO)
    almacen.marcar_correo("msg-1", almacen.DESCARTADO)
    assert almacen.correos_marcados() == {"msg-1": almacen.DESCARTADO}


def test_volver_a_pendiente_borra_la_fila(db) -> None:
    """Pendiente es no estar. Con dos formas de decirlo, una acaba mintiendo."""
    almacen.marcar_correo("msg-1", almacen.ATENDIDO)
    almacen.marcar_correo("msg-1", almacen.PENDIENTE_CORREO)
    assert almacen.correos_marcados() == {}


def test_un_estado_inventado_no_pasa(db) -> None:
    with pytest.raises(ValueError):
        almacen.marcar_correo("msg-1", "archivado")


def test_un_correo_sin_id_no_se_marca(db) -> None:
    with pytest.raises(ValueError):
        almacen.marcar_correo("   ", almacen.ATENDIDO)


def test_los_correos_marcados_sobreviven_al_reinicio(db, cfg) -> None:
    """Es una decisión tuya, no un cacheo: si se pierde al reiniciar, no sirve."""
    almacen.marcar_correo("msg-1", almacen.ATENDIDO)
    almacen.cerrar()
    almacen.abrir(cfg)
    assert almacen.correos_marcados() == {"msg-1": almacen.ATENDIDO}
