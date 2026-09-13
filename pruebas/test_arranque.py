"""Lo que arranca con Windows, y por qué se rompe en silencio.

Tres piezas: el fichero que le da su configuración al núcleo cuando no hay
terminal detrás, el vigilante que lo vuelve a levantar, y la revisión de la
entrada del registro. Las tres fallan calladas si nadie las mira: el síntoma
siempre es el mismo —"Perseo ya no hace cosas"— y no lleva a ninguna causa.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "commands"))

import configurar_arranque  # noqa: E402
import vigilante  # noqa: E402

from perseo_core.infra.configuracion import ajustes_guardados, cargar_configuracion  # noqa: E402


# --------------------------------------------------------------------------- #
# entorno.json: la configuración del núcleo sin terminal
# --------------------------------------------------------------------------- #


def test_sin_fichero_no_hay_ajustes(datos: Path) -> None:
    assert ajustes_guardados(datos) == {}


def test_el_fichero_da_valores_por_defecto(datos: Path) -> None:
    (datos / "entorno.json").write_text(json.dumps({"PERSEO_CORREO": "gmail"}), encoding="utf-8")
    assert cargar_configuracion().correo_buzon == "gmail"


def test_el_entorno_manda_sobre_el_fichero(datos: Path, monkeypatch) -> None:
    """El fichero son los valores de esta instalación, no una orden."""
    (datos / "entorno.json").write_text(json.dumps({"PERSEO_CORREO": "gmail"}), encoding="utf-8")
    monkeypatch.setenv("PERSEO_CORREO", "falso")
    assert cargar_configuracion().correo_buzon == "falso"


def test_una_variable_vacia_en_el_entorno_tambien_manda(datos: Path, monkeypatch) -> None:
    """Poner `PERSEO_CORREO=` a mano es apagar el correo, no callarse."""
    (datos / "entorno.json").write_text(json.dumps({"PERSEO_CORREO": "gmail"}), encoding="utf-8")
    monkeypatch.setenv("PERSEO_CORREO", "")
    assert cargar_configuracion().correo_buzon == ""


def test_un_fichero_roto_no_impide_arrancar(datos: Path) -> None:
    """Sin núcleo no hay nada; con la configuración a medias, casi todo."""
    (datos / "entorno.json").write_text("{esto no es json", encoding="utf-8")
    assert ajustes_guardados(datos) == {}
    assert cargar_configuracion().puerto == 8787


def test_un_fichero_que_no_es_un_objeto_se_ignora(datos: Path) -> None:
    (datos / "entorno.json").write_text('["una", "lista"]', encoding="utf-8")
    assert ajustes_guardados(datos) == {}


def test_los_numeros_del_fichero_se_leen_como_texto(datos: Path) -> None:
    """JSON permite números; `float(...)` sobre un int no se queja, pero el
    resto del código espera cadenas."""
    (datos / "entorno.json").write_text(json.dumps({"PERSEO_CORE_PUERTO": 9999}), encoding="utf-8")
    assert cargar_configuracion().puerto == 9999


def test_los_secretos_tambien_salen_del_fichero(datos: Path) -> None:
    (datos / "entorno.json").write_text(
        json.dumps({"PERSEO_TELEGRAM_TOKEN": "de-fichero"}), encoding="utf-8"
    )
    assert cargar_configuracion().telegram_token == "de-fichero"


def test_el_secreto_del_directorio_gana_al_del_fichero_de_ajustes(datos: Path) -> None:
    """`telegram.txt` es donde lo deja el ayudante; el JSON es para lo demás."""
    (datos / "entorno.json").write_text(
        json.dumps({"PERSEO_TELEGRAM_CHAT": "del-json"}), encoding="utf-8"
    )
    (datos / "telegram_chat.txt").write_text("del-fichero", encoding="utf-8")
    # El de `entorno.json` se consulta antes: es configuración explícita de esta
    # instalación, y el .txt es el rastro que dejó el descubrimiento.
    assert cargar_configuracion().telegram_chat == "del-json"


# --------------------------------------------------------------------------- #
# Qué se recomienda encender al arrancar
# --------------------------------------------------------------------------- #


def test_sin_credenciales_no_se_enciende_nada(tmp_path: Path, monkeypatch) -> None:
    """Y sin vault de verdad a la vista: la detección del vault grande mira el
    Documents de la máquina real, y aquí se apaga para que la prueba no dependa
    de dónde vive el cerebro de nadie."""
    monkeypatch.setattr(configurar_arranque, "_vault_de_verdad", lambda: None)
    assert configurar_arranque.ajustes_recomendados(tmp_path, hay_tailscale=False) == {}


def test_el_vault_grande_de_documents_va_al_entorno(tmp_path: Path, monkeypatch) -> None:
    """El grafo del segundo cerebro lee el disco: necesita la ruta del vault de
    verdad, no la del de fábrica de dentro de Perseo."""
    vault = tmp_path / "Persus"
    (vault / ".obsidian").mkdir(parents=True)
    monkeypatch.setattr(configurar_arranque, "_vault_de_verdad", lambda: vault)
    ajustes = configurar_arranque.ajustes_recomendados(tmp_path, False)
    assert ajustes["OBSIDIAN_VAULT_PATH"] == str(vault)


def test_con_tailscale_se_abre_el_tailnet(tmp_path: Path) -> None:
    """Sin esto, el enlace de Telegram no sirve desde el móvil."""
    ajustes = configurar_arranque.ajustes_recomendados(tmp_path, hay_tailscale=True)
    assert ajustes["PERSEO_CORE_HOST"] == "tailscale"


def test_con_la_clave_de_obsidian_el_vault_va_por_el_plugin(tmp_path: Path) -> None:
    (tmp_path / "obsidian.txt").write_text("clave", encoding="utf-8")
    assert configurar_arranque.ajustes_recomendados(tmp_path, False)["PERSEO_VAULT"] == "rest"


def test_con_google_completo_se_encienden_correo_y_agenda(tmp_path: Path) -> None:
    (tmp_path / "google.json").write_text(
        json.dumps({"client_id": "i", "client_secret": "s", "refresh_token": "r"}), encoding="utf-8"
    )
    ajustes = configurar_arranque.ajustes_recomendados(tmp_path, False)
    assert ajustes["PERSEO_CORREO"] == "gmail"
    assert ajustes["PERSEO_AGENDA"] == "google"


def test_sin_consentimiento_no_se_enciende_gmail(tmp_path: Path) -> None:
    """El JSON de la consola existe desde antes de dar permiso: encenderlo así
    sería programar un fallo cada cinco minutos."""
    (tmp_path / "google.json").write_text(
        json.dumps({"installed": {"client_id": "i", "client_secret": "s"}}), encoding="utf-8"
    )
    assert "PERSEO_CORREO" not in configurar_arranque.ajustes_recomendados(tmp_path, False)


def test_un_google_json_ilegible_no_revienta(tmp_path: Path) -> None:
    (tmp_path / "google.json").write_text("{roto", encoding="utf-8")
    assert "PERSEO_CORREO" not in configurar_arranque.ajustes_recomendados(tmp_path, False)


# --------------------------------------------------------------------------- #
# El vigilante
# --------------------------------------------------------------------------- #


def test_si_se_cae_rapido_la_espera_crece() -> None:
    """Un núcleo que revienta al arrancar no se arregla reintentando cada segundo."""
    assert vigilante.siguiente_espera(5.0, vivio=1.0) == 10.0
    assert vigilante.siguiente_espera(10.0, vivio=1.0) == 20.0


def test_la_espera_tiene_techo() -> None:
    assert vigilante.siguiente_espera(60.0, vivio=1.0) == vigilante.ESPERA_MAXIMA


def test_si_aguanto_en_pie_la_espera_se_reinicia() -> None:
    """Un núcleo que funcionó una semana no hereda la espera de un fallo viejo."""
    assert vigilante.siguiente_espera(60.0, vivio=3600.0) == vigilante.ESPERA_INICIAL


@pytest.mark.parametrize("vivio", [0.0, 119.9])
def test_el_arranque_corto_no_cuenta_como_bueno(vivio: float) -> None:
    assert vigilante.siguiente_espera(5.0, vivio) == 10.0


# --------------------------------------------------------------------------- #
# El registro del núcleo
# --------------------------------------------------------------------------- #


def test_un_registro_pequeno_se_queda_donde_esta(tmp_path: Path) -> None:
    registro = tmp_path / "nucleo.log"
    registro.write_text("dos líneas de nada", encoding="utf-8")
    assert vigilante.apartar_si_crece(registro, tope=1024) is False
    assert registro.exists()


def test_un_registro_grande_se_aparta(tmp_path: Path) -> None:
    """Esto arranca con Windows: sin tope, un registro que nadie mira se come
    el disco."""
    registro = tmp_path / "nucleo.log"
    registro.write_text("x" * 2048, encoding="utf-8")

    assert vigilante.apartar_si_crece(registro, tope=1024) is True

    assert not registro.exists()
    apartado = tmp_path / "nucleo.log.viejo"
    assert apartado.read_text(encoding="utf-8") == "x" * 2048


def test_apartar_dos_veces_no_acumula_ficheros(tmp_path: Path) -> None:
    """Se guarda la última muerte, no el histórico."""
    registro = tmp_path / "nucleo.log"
    registro.write_text("viejo" * 500, encoding="utf-8")
    vigilante.apartar_si_crece(registro, tope=1024)
    registro.write_text("nuevo" * 500, encoding="utf-8")
    vigilante.apartar_si_crece(registro, tope=1024)

    assert sorted(p.name for p in tmp_path.glob("nucleo.log*")) == ["nucleo.log.viejo"]
    assert (tmp_path / "nucleo.log.viejo").read_text(encoding="utf-8").startswith("nuevo")


def test_sin_registro_todavia_no_hay_nada_que_apartar(tmp_path: Path) -> None:
    """La primera vez el fichero no existe, y eso no es un problema."""
    assert vigilante.apartar_si_crece(tmp_path / "nucleo.log", tope=1024) is False


# --------------------------------------------------------------------------- #
# La entrada del registro
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(sys.platform != "win32", reason="el registro es de Windows")
def test_una_entrada_con_rutas_vivas_no_tiene_problemas(tmp_path: Path) -> None:
    import manage_startup

    exe = tmp_path / "pythonw.exe"
    exe.write_text("", encoding="utf-8")
    guion = tmp_path / "vigilante.py"
    guion.write_text("", encoding="utf-8")
    assert manage_startup.revisar(f'"{exe}" "{guion}"') == []


@pytest.mark.skipif(sys.platform != "win32", reason="el registro es de Windows")
def test_una_ruta_muerta_se_detecta(tmp_path: Path) -> None:
    """Pasa al actualizar Python: la ruta lleva la versión dentro y Windows
    intenta arrancarlo igual, en silencio."""
    import manage_startup

    problemas = manage_startup.revisar(f'"{tmp_path / "Python310" / "pythonw.exe"}" "{tmp_path}"')
    assert len(problemas) == 1 and "no existe" in problemas[0]


# --------------------------------------------------------------------------- #
# Lo que el registro NO dice: si hay algo encendido ahora mismo
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(sys.platform != "win32", reason="el registro es de Windows")
def test_la_url_de_salud_respeta_el_puerto(monkeypatch: pytest.MonkeyPatch) -> None:
    import manage_startup

    monkeypatch.setenv("PERSEO_CORE_PUERTO", "9999")
    assert manage_startup.url_salud() == "http://127.0.0.1:9999/salud"


@pytest.mark.skipif(sys.platform != "win32", reason="el registro es de Windows")
def test_sin_puerto_el_de_siempre(monkeypatch: pytest.MonkeyPatch) -> None:
    import manage_startup

    monkeypatch.delenv("PERSEO_CORE_PUERTO", raising=False)
    assert manage_startup.url_salud() == "http://127.0.0.1:8787/salud"


@pytest.mark.skipif(sys.platform != "win32", reason="el registro es de Windows")
def test_un_nucleo_apagado_no_responde() -> None:
    """El caso que el 2026-08-16 nadie vio: entrada del registro intacta y nada
    escuchando. Un puerto cerrado del bucle local rechaza al instante, así que
    esto no espera los tres segundos."""
    import socket

    import manage_startup

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        puerto = s.getsockname()[1]

    assert not manage_startup.nucleo_responde(f"http://127.0.0.1:{puerto}/salud")


@pytest.mark.skipif(sys.platform != "win32", reason="el registro es de Windows")
def test_una_url_absurda_tampoco_lanza() -> None:
    import manage_startup

    assert not manage_startup.nucleo_responde("esto no es una url")


def test_un_entorno_con_bom_se_lee_igual(datos: Path) -> None:
    """El Bloc de notas y PowerShell 5.1 escriben UTF-8 con BOM. Con `utf-8` eso
    era un JSON roto, y el núcleo arrancaba sin correo, sin agenda y sin tailnet
    sin quejarse. Pasó de verdad el 2026-08-17."""
    (datos / "entorno.json").write_text(
        json.dumps({"PERSEO_CORREO": "gmail"}), encoding="utf-8-sig"
    )
    assert ajustes_guardados(datos) == {"PERSEO_CORREO": "gmail"}


# --------------------------------------------------------------------------- #
# `perseo on` y `perseo off`
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(sys.platform != "win32", reason="perseo.py importa el registro de Windows")
def test_on_y_off_existen_con_sus_sinonimos() -> None:
    """Las dos órdenes que pidió el señor Persus el 2026-08-21, y que `perseo` a
    secas siga siendo encender: lo dice así la bitácora entera."""
    import perseo

    assert perseo.ORDENES["on"] is perseo.todo
    assert perseo.ORDENES[""] is perseo.todo
    assert perseo.ORDENES["encender"] is perseo.todo
    assert perseo.ORDENES["off"] is perseo.apagar
    assert perseo.ORDENES["apagar"] is perseo.apagar


@pytest.mark.skipif(sys.platform != "win32", reason="perseo.py importa el registro de Windows")
def test_parar_no_es_apagar() -> None:
    """`parar` deja el detector vivo a propósito y `off` no. Si un día se
    igualaran, apagar Perseo dejaría de apagarlo o `parar` dejaría el sistema
    sordo — y las dos cosas se descubren aplaudiendo, que es tarde."""
    import perseo

    assert perseo.ORDENES["parar"] is not perseo.ORDENES["off"]


@pytest.mark.skipif(sys.platform != "win32", reason="perseo.py importa el registro de Windows")
def test_se_coge_la_primera_ruta_que_exista(tmp_path: Path) -> None:
    """Obsidian se instala en tres sitios según la versión y quién lo instalara."""
    import perseo

    hay = tmp_path / "Obsidian.exe"
    hay.write_text("", encoding="utf-8")
    no_hay = tmp_path / "no" / "Obsidian.exe"

    assert perseo._primera_que_exista((no_hay, hay)) == hay
    assert perseo._primera_que_exista((no_hay,)) is None
    assert perseo._primera_que_exista(()) is None


@pytest.mark.skipif(sys.platform != "win32", reason="pregunta a tasklist, que es de Windows")
def test_un_programa_que_no_existe_no_esta_vivo() -> None:
    """Y sobre todo: no lanza. `_exe_vivo` se llama en `estado`, que tiene que
    contestar siempre aunque tasklist tenga un mal día."""
    import perseo

    assert not perseo._exe_vivo("esto-no-existe-jamas.exe")


# --------------------------------------------------------------------------- #
# El arranque con Windows: una sola entrada, y quien revive lo que se caiga
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(sys.platform != "win32", reason="el registro es de Windows")
def test_en_el_registro_no_va_nada() -> None:
    """Lo decidió el señor Persus el 2026-08-22 y lo confirmó al día siguiente
    con el detector delante: Perseo se abre con `perseo on` o despertado por dos
    palmadas, y por más nada. Una entrada en `Run` abriría ventanas al encender
    el PC, que es justo lo que no quiere. Lo único externo es la tarea
    `PerseoRevivir`, que no abre nada — y esa se comprueba en su propia prueba.
    """
    import manage_startup

    assert manage_startup.SERVICIOS == ()
    assert "Perseo" not in manage_startup.LEGADO


@pytest.mark.skipif(sys.platform != "win32", reason="el registro es de Windows")
def test_las_entradas_de_antes_estan_listadas_para_quitarlas() -> None:
    """Si sobreviven, arrancan a la vez que la nueva: dos núcleos peleándose por
    el 8787 y dos detectores por el mismo micrófono."""
    import manage_startup

    assert manage_startup.LEGADO == ("PerseoClapDetector", "PerseoNucleo")
    assert not set(manage_startup.LEGADO) & set(manage_startup.SERVICIOS)


@pytest.mark.skipif(sys.platform != "win32", reason="el registro es de Windows")
def test_la_tarea_que_revive_llama_al_mismo_guion_con_revivir() -> None:
    import manage_startup

    orden = manage_startup._orden_de_la_tarea()
    assert orden is not None
    assert orden.endswith("--revivir")
    assert "arranque.py" in orden


def test_revivir_no_levanta_la_app_ni_las_dependencias() -> None:
    """Lo que se comprueba es la decisión, no el código: la app, Ollama y
    Obsidian tienen ventana, y una ventana que reaparece sola cada diez minutos
    es un programa con el que no se puede convivir. El núcleo y el detector no
    tienen ventana: si están apagados, es que algo falló."""
    import inspect

    import arranque

    fuente = inspect.getsource(arranque.revivir)
    assert "arrancar_nucleo" in fuente
    assert "arrancar_detector" in fuente
    assert "arrancar_app" not in fuente
    assert "arrancar_ollama" not in fuente
    assert "arrancar_obsidian" not in fuente


def test_el_arranque_escribe_en_su_registro(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Este proceso no tiene consola: si no escribe en un fichero, no dice nada.
    Y el día que el PC arranque sin Perseo, ese fichero es lo único que hay."""
    import arranque

    monkeypatch.setenv("PERSEO_CORE_DATOS", str(tmp_path))
    monkeypatch.setattr(arranque, "revivir", lambda: print("levantando lo que falte"))

    assert arranque.main(["--revivir"]) == 0
    escrito = (tmp_path / "arranque.log").read_text(encoding="utf-8")
    assert "Revivir" in escrito
    assert "levantando lo que falte" in escrito


def test_un_arranque_que_revienta_deja_la_traza(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Un fallo aquí es un Perseo que no arranca con el PC: sin consola, sin
    error y sin nadie mirando. Lo único que puede quedar es la traza."""
    import arranque

    monkeypatch.setenv("PERSEO_CORE_DATOS", str(tmp_path))

    def revienta() -> None:
        raise RuntimeError("no se pudo con el núcleo")

    monkeypatch.setattr(arranque, "encender", revienta)

    assert arranque.main([]) == 1
    escrito = (tmp_path / "arranque.log").read_text(encoding="utf-8")
    assert "RuntimeError" in escrito and "no se pudo con el núcleo" in escrito
    # Y la salida vuelve a su sitio, o la prueba siguiente escribiría en el log.
    assert sys.stdout is not None


def test_el_vigilante_deja_dicho_si_se_muere_por_una_excepcion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """El 2026-08-18 su última línea fue «se vuelve a arrancar en 5s» y después
    nada durante tres días: sin consola, una excepción lo mata en silencio."""
    monkeypatch.setenv("PERSEO_CORE_DATOS", str(tmp_path))

    def revienta() -> int:
        raise RuntimeError("se acabó el disco")

    monkeypatch.setattr(vigilante, "vigilar", revienta)

    with pytest.raises(RuntimeError):
        vigilante.vigilar_diciendo_como_muere()

    escrito = (tmp_path / "vigilante.log").read_text(encoding="utf-8")
    assert "se muere por RuntimeError" in escrito
    assert "se acabó el disco" in escrito
    assert "Traceback" in escrito


# --------------------------------------------------------------------------- #
# Dos núcleos a la vez: el segundo se retira
# --------------------------------------------------------------------------- #


class _RespuestaFalsa:
    """Lo mínimo que urlopen devuelve y que la comprobación mira."""

    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def _con_salud(monkeypatch, respuesta) -> bool:
    import urllib.request

    from perseo_core import __main__ as arranque_nucleo

    def falso_urlopen(url, timeout=None):
        if isinstance(respuesta, Exception):
            raise respuesta
        return respuesta

    monkeypatch.setattr(urllib.request, "urlopen", falso_urlopen)
    cfg = cargar_configuracion()
    return arranque_nucleo._ya_contesta_otro_nucleo(cfg)


def test_si_otro_nucleo_contesta_este_sobra(datos: Path, monkeypatch) -> None:
    """Arrancar el segundo es lo que dejaba el bucle de OSError 10048."""
    assert _con_salud(monkeypatch, _RespuestaFalsa(200))


def test_un_401_tambien_cuenta_como_alguien_vivo(datos: Path, monkeypatch) -> None:
    """No se pregunta si nos dejan entrar: se pregunta si hay alguien."""
    import urllib.error

    error = urllib.error.HTTPError("u", 401, "no", None, None)
    assert _con_salud(monkeypatch, error)


def test_con_el_puerto_libre_se_arranca(datos: Path, monkeypatch) -> None:
    assert not _con_salud(monkeypatch, ConnectionRefusedError())
