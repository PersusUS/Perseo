"""Los otros proyectos: qué se puede abrir y, sobre todo, qué no.

El panel se alcanza desde el tailnet, así que lo que decide qué se ejecuta es un
fichero del disco y nunca la petición. Estas pruebas comprueban el borde.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from perseo_core import proyectos


def escribir(directorio: Path, entradas: list[dict]) -> None:
    (directorio / proyectos.NOMBRE_FICHERO).write_text(
        json.dumps(entradas), encoding="utf-8"
    )


def test_sin_fichero_no_hay_proyectos(tmp_path: Path) -> None:
    """Es un estado válido: la pantalla enseña cómo crearlo."""
    assert proyectos.listar(tmp_path) == []


def test_un_fichero_roto_no_revienta(tmp_path: Path) -> None:
    (tmp_path / proyectos.NOMBRE_FICHERO).write_text("{esto no es json", encoding="utf-8")
    assert proyectos.listar(tmp_path) == []


def test_una_carpeta_se_lee(tmp_path: Path) -> None:
    escribir(tmp_path, [{"id": "armario", "nombre": "Armario", "modo": "carpeta",
                         "destino": str(tmp_path)}])
    lista = proyectos.listar(tmp_path)
    assert [p.id for p in lista] == ["armario"]
    assert lista[0].nombre == "Armario"


def test_una_entrada_rota_no_se_lleva_las_buenas(tmp_path: Path) -> None:
    """El fichero lo escribe una persona: perder cinco proyectos por una coma
    es peor que perder el que está mal."""
    escribir(tmp_path, [
        {"id": "roto", "modo": "telepatia", "destino": "x"},
        {"id": "bueno", "modo": "carpeta", "destino": str(tmp_path)},
    ])
    assert [p.id for p in proyectos.listar(tmp_path)] == ["bueno"]


def test_un_esquema_que_no_es_http_se_descarta(tmp_path: Path) -> None:
    escribir(tmp_path, [{"id": "malo", "modo": "url", "destino": "file:///C:/Windows"}])
    assert proyectos.listar(tmp_path) == []


def test_un_programa_fuera_de_la_lista_blanca_se_descarta(tmp_path: Path) -> None:
    """La lista blanca es la misma del agente `pc`: dos acabarían discrepando."""
    escribir(tmp_path, [{"id": "malo", "modo": "programa", "destino": "powershell",
                         "carpeta": str(tmp_path)}])
    assert proyectos.listar(tmp_path) == []


def test_un_programa_de_la_lista_blanca_si_vale(tmp_path: Path) -> None:
    escribir(tmp_path, [{"id": "notas", "modo": "programa", "destino": "notepad",
                         "carpeta": str(tmp_path)}])
    assert [p.destino for p in proyectos.listar(tmp_path)] == ["notepad"]


def test_abrir_algo_que_no_esta_en_la_lista_es_un_error(tmp_path: Path) -> None:
    """Por HTTP llega cuál de los proyectos, no qué ejecutar."""
    escribir(tmp_path, [{"id": "armario", "modo": "carpeta", "destino": str(tmp_path)}])
    assert proyectos.abrir(tmp_path, "cmd").startswith("Error:")
    assert proyectos.abrir(tmp_path, "../../otra-cosa").startswith("Error:")


def test_abrir_una_carpeta_que_ya_no_existe_lo_dice(tmp_path: Path) -> None:
    escribir(tmp_path, [{"id": "fantasma", "modo": "carpeta",
                         "destino": str(tmp_path / "no-existe")}])
    assert proyectos.abrir(tmp_path, "fantasma").startswith("Error:")


# --------------------------------------------------------------------------- #
# Arrancar el proyecto, no abrir su carpeta (T-6, 2026-08-21)
# --------------------------------------------------------------------------- #


def _con_lista(tmp_path: Path, entradas: list[dict]) -> Path:
    escribir(tmp_path, entradas)
    return tmp_path


def test_un_arranque_bien_escrito_entra_en_la_lista(tmp_path) -> None:
    datos = _con_lista(
        tmp_path,
        [{"id": "web", "nombre": "Web", "modo": "arranque",
          "arranque": [sys.executable, "-c", "pass"], "carpeta": str(tmp_path)}],
    )
    lista = proyectos.listar(datos)
    assert len(lista) == 1
    assert lista[0].arranque == (sys.executable, "-c", "pass")


def test_un_arranque_sin_destino_vale(tmp_path) -> None:
    """Es el único modo que no lo necesita: lo que se abre es la orden."""
    datos = _con_lista(
        tmp_path,
        [{"id": "web", "modo": "arranque", "arranque": [sys.executable, "-V"]}],
    )
    assert len(proyectos.listar(datos)) == 1


def test_una_orden_en_una_sola_cadena_se_rechaza(tmp_path) -> None:
    """`"npm run dev"` de una pieza solo se puede ejecutar dándoselo a un shell,
    y ahí es donde viven las comillas y los `&&`. Se exige lista."""
    datos = _con_lista(
        tmp_path, [{"id": "web", "modo": "arranque", "arranque": "npm run dev"}]
    )
    assert proyectos.listar(datos) == []


def test_un_arranque_vacio_o_con_huecos_se_rechaza(tmp_path) -> None:
    datos = _con_lista(
        tmp_path,
        [
            {"id": "a", "modo": "arranque", "arranque": []},
            {"id": "b", "modo": "arranque", "arranque": [sys.executable, "  "]},
            {"id": "c", "modo": "arranque"},
        ],
    )
    assert proyectos.listar(datos) == []


def test_un_programa_que_no_existe_se_descarta_al_leer(tmp_path) -> None:
    """Como una carpeta que ya no está: se dice al leer la lista, en vez de
    dejar el botón puesto para que falle al pulsarlo."""
    datos = _con_lista(
        tmp_path,
        [{"id": "web", "modo": "arranque", "arranque": ["no-existe-este-programa-jamas"]}],
    )
    assert proyectos.listar(datos) == []


def test_una_entrada_mala_no_se_lleva_por_delante_a_las_buenas(tmp_path) -> None:
    datos = _con_lista(
        tmp_path,
        [
            {"id": "malo", "modo": "arranque", "arranque": "npm run dev"},
            {"id": "bueno", "modo": "arranque", "arranque": [sys.executable, "-V"]},
        ],
    )
    assert [p.id for p in proyectos.listar(datos)] == ["bueno"]


def test_arrancar_lanza_el_programa_y_lo_dice(tmp_path) -> None:
    """De punta a punta y con un proceso de verdad: se lanza y se contesta que
    se lanzó, que es lo único que se puede saber en ese momento."""
    testigo = tmp_path / "arranco.txt"
    guion = f"open(r'{testigo}', 'w').write('si')"
    datos = _con_lista(
        tmp_path,
        [{"id": "web", "nombre": "Web", "modo": "arranque",
          "arranque": [sys.executable, "-c", guion], "carpeta": str(tmp_path)}],
    )

    respuesta = proyectos.abrir(datos, "web")
    assert respuesta.startswith("Éxito")

    for _ in range(50):
        if testigo.exists():
            break
        time.sleep(0.1)
    assert testigo.read_text(encoding="utf-8") == "si"


def test_arrancar_desde_una_carpeta_que_ya_no_existe_lo_dice(tmp_path) -> None:
    datos = _con_lista(
        tmp_path,
        [{"id": "web", "nombre": "Web", "modo": "arranque",
          "arranque": [sys.executable, "-V"], "carpeta": str(tmp_path / "fantasma")}],
    )
    respuesta = proyectos.abrir(datos, "web")
    assert respuesta.startswith("Error") and "ya no existe" in respuesta


def test_puerto_responde_dice_si_el_servicio_vive(tmp_path) -> None:
    """El estado que pinta el carril: un puerto escuchando es «en marcha» y
    uno cerrado es «parado». Con un socket de verdad, no con imitaciones."""
    import socket

    with socket.socket() as oyente:
        oyente.bind(("127.0.0.1", 0))
        oyente.listen(1)
        puerto = oyente.getsockname()[1]
        assert proyectos.puerto_responde(f"http://127.0.0.1:{puerto}")
    assert not proyectos.puerto_responde(f"http://127.0.0.1:{puerto}")


def test_por_http_sigue_viajando_solo_el_id() -> None:
    """La regla que sostiene todo: `abrir` recibe un identificador, y lo que se
    ejecuta sale del fichero del disco. Si algún día acepta la orden por
    parámetro, esto tiene que ponerse rojo."""
    import inspect

    firma = inspect.signature(proyectos.abrir)
    assert list(firma.parameters) == ["directorio_datos", "id_proyecto"]


# --------------------------------------------------------------------------- #
# El servicio: correr el proyecto y abrir su pestaña (2026-08-24)
# --------------------------------------------------------------------------- #


def test_un_servicio_bien_escrito_entra_en_la_lista(tmp_path) -> None:
    """Dos procesos con carpetas distintas, como cvscraper: backend y frontend."""
    datos = _con_lista(
        tmp_path,
        [{
            "id": "app", "nombre": "App", "modo": "servicio",
            "destino": "http://localhost:5173",
            "servidores": [
                {"arranque": [sys.executable, "-V"], "carpeta": str(tmp_path / "b")},
                {"arranque": [sys.executable, "-V"], "carpeta": str(tmp_path / "f")},
            ],
        }],
    )
    lista = proyectos.listar(datos)
    assert len(lista) == 1
    assert lista[0].destino == "http://localhost:5173"
    assert [s["carpeta"] for s in lista[0].servidores] == [
        str(tmp_path / "b"),
        str(tmp_path / "f"),
    ]


def test_un_servicio_sin_carpeta_vale(tmp_path) -> None:
    """La carpeta es opcional por servidor: hay órdenes que no la necesitan."""
    datos = _con_lista(
        tmp_path,
        [{"id": "app", "modo": "servicio", "destino": "http://127.0.0.1:8123",
          "servidores": [{"arranque": [sys.executable, "-V"]}]}],
    )
    lista = proyectos.listar(datos)
    assert len(lista) == 1
    assert lista[0].servidores[0]["carpeta"] == ""


def test_un_servicio_con_url_mala_o_sin_url_se_descarta(tmp_path) -> None:
    """La pestaña es la mitad del encargo: sin URL http/https no hay servicio."""
    datos = _con_lista(
        tmp_path,
        [
            {"id": "a", "modo": "servicio", "destino": "ftp://x",
             "servidores": [{"arranque": [sys.executable, "-V"]}]},
            {"id": "b", "modo": "servicio",
             "servidores": [{"arranque": [sys.executable, "-V"]}]},
        ],
    )
    assert proyectos.listar(datos) == []


def test_un_servicio_sin_servidores_se_descarta(tmp_path) -> None:
    """Falta la clave, viene vacía, no es una lista o la orden va en cadena:
    todo eso es una entrada mal escrita y se ignora con las buenas a salvo."""
    datos = _con_lista(
        tmp_path,
        [
            {"id": "a", "modo": "servicio", "destino": "http://x.es"},
            {"id": "b", "modo": "servicio", "destino": "http://x.es", "servidores": []},
            {"id": "c", "modo": "servicio", "destino": "http://x.es",
             "servidores": ["npm run dev"]},
            {"id": "d", "modo": "servicio", "destino": "http://x.es",
             "servidores": [{"arranque": "npm run dev"}]},
        ],
    )
    assert proyectos.listar(datos) == []


def test_un_servicio_con_programa_desconocido_se_descarta_al_leer(tmp_path) -> None:
    """Igual que en `arranque`: se dice al leer la lista, no al pulsar."""
    datos = _con_lista(
        tmp_path,
        [{"id": "app", "modo": "servicio", "destino": "http://x.es",
          "servidores": [{"arranque": ["no-existe-este-programa-jamas"]}]}],
    )
    assert proyectos.listar(datos) == []


def test_servicio_ya_en_marcha_lo_dice_y_no_arranca_nada(
    tmp_path, monkeypatch
) -> None:
    """Si el puerto ya respira, arrancar otra vez costaría dos servidores."""
    monkeypatch.setattr(proyectos, "_puerto_abierto", lambda url, plazo=0.6: True)
    datos = _con_lista(
        tmp_path,
        [{"id": "app", "nombre": "App", "modo": "servicio",
          "destino": "http://127.0.0.1:8123",
          "servidores": [{"arranque": [sys.executable, "-V"]}]}],
    )
    try:
        respuesta = proyectos.abrir(datos, "app")
        assert respuesta.startswith("Éxito") and "ya está en marcha" in respuesta
        # Y el cerrojo quedó libre: no hay nadie «arrancando».
        assert "app" not in proyectos._EN_MARCHA
    finally:
        proyectos._EN_MARCHA.discard("app")


def test_la_ventana_declarada_entra_en_la_lista(tmp_path) -> None:
    """Cada app pide el sitio que necesita; el fichero lo escribe una persona."""
    datos = _con_lista(
        tmp_path,
        [{"id": "app", "modo": "servicio", "destino": "http://127.0.0.1:8123",
          "servidores": [{"arranque": [sys.executable, "-V"]}],
          "ventana": {"ancho": 1500, "alto": 950}}],
    )
    lista = proyectos.listar(datos)
    assert lista[0].ventana == {"ancho": 1500, "alto": 950}


def test_una_ventana_sin_declara_vale_y_mal_escrita_descarta(tmp_path) -> None:
    """Sin 'ventana' todo va bien; con una que no se entiende, fuera la
    entrada — igual que cualquier otra mal escrita."""
    bien = {"id": "bien", "modo": "servicio", "destino": "http://x.es",
            "servidores": [{"arranque": [sys.executable, "-V"]}]}
    datos = _con_lista(
        tmp_path,
        [
            dict(bien, id="sin-ventana"),
            {**bien, "id": "texto", "ventana": "grande"},
            {**bien, "id": "negativa", "ventana": {"ancho": -5, "alto": 900}},
            bien,
        ],
    )
    assert [p.id for p in proyectos.listar(datos)] == ["sin-ventana", "bien"]
    assert proyectos.listar(datos)[0].ventana is None


def test_servicio_arranca_los_procesos_y_deja_un_vigilante(tmp_path, monkeypatch) -> None:
    """De punta a punta y con procesos de verdad: los dos despegan y el
    vigilante queda esperando el puerto para abrir la pestaña."""
    testigos = [tmp_path / "uno.txt", tmp_path / "dos.txt"]
    guiones = [f"open(r'{t}', 'w').write('si')" for t in testigos]
    vigilantes: list[tuple[str, str]] = []

    def falso_vigilante(id_proyecto: str, nombre: str, url: str) -> None:
        vigilantes.append((id_proyecto, url))
        # Lo mismo que hace el de verdad al terminar: libera el cerrojo.
        with proyectos._CERROJO_SERVICIOS:
            proyectos._EN_MARCHA.discard(id_proyecto)

    monkeypatch.setattr(proyectos, "_puerto_abierto", lambda url, plazo=0.6: False)
    monkeypatch.setattr(proyectos, "_cuando_este_listo", falso_vigilante)
    datos = _con_lista(
        tmp_path,
        [{
            "id": "app", "nombre": "App", "modo": "servicio",
            "destino": "http://127.0.0.1:5173",
            "servidores": [
                {"arranque": [sys.executable, "-c", guiones[0]], "carpeta": str(tmp_path)},
                {"arranque": [sys.executable, "-c", guiones[1]], "carpeta": str(tmp_path)},
            ],
        }],
    )
    respuesta = proyectos.abrir(datos, "app")
    assert respuesta.startswith("Éxito") and "arrancando" in respuesta
    assert vigilantes == [("app", "http://127.0.0.1:5173")]

    for t in testigos:
        for _ in range(50):
            if t.exists():
                break
            time.sleep(0.1)
        assert t.read_text(encoding="utf-8") == "si"


def test_doble_pulsacion_mientras_arranca_no_duplica(tmp_path, monkeypatch) -> None:
    """El cerrojo por proyecto: mientras el puerto no contesta, una segunda
    pulsación no vuelve a lanzar nada."""
    monkeypatch.setattr(proyectos, "_puerto_abierto", lambda url, plazo=0.6: False)

    def vigilante_mudo(id_proyecto: str, nombre: str, url: str) -> None:
        pass  # Simula un arranque lento: no libera el cerrojo todavía.

    monkeypatch.setattr(proyectos, "_cuando_este_listo", vigilante_mudo)
    datos = _con_lista(
        tmp_path,
        [{"id": "lento", "nombre": "Lento", "modo": "servicio",
          "destino": "http://127.0.0.1:5173",
          "servidores": [{"arranque": [sys.executable, "-c", "pass"]}]}],
    )
    try:
        assert proyectos.abrir(datos, "lento").startswith("Éxito")
        segunda = proyectos.abrir(datos, "lento")
        assert segunda.startswith("Éxito") and "ya se está arrancando" in segunda
    finally:
        proyectos._EN_MARCHA.discard("lento")


def test_si_el_arranque_falla_el_cerrojo_se_libera(tmp_path, monkeypatch) -> None:
    """Un fallo a medias no puede dejar el proyecto marcado como «en marcha»
    para siempre: la siguiente pulsación tiene que poder reintentarlo."""
    monkeypatch.setattr(proyectos, "_puerto_abierto", lambda url, plazo=0.6: False)
    monkeypatch.setattr(proyectos, "_cuando_este_listo", lambda *a: None)
    datos = _con_lista(
        tmp_path,
        [{"id": "roto", "nombre": "Roto", "modo": "servicio",
          "destino": "http://127.0.0.1:5173",
          "servidores": [{"arranque": [sys.executable, "-V"],
                          "carpeta": str(tmp_path / "fantasma")}]}],
    )
    try:
        assert proyectos.abrir(datos, "roto").startswith("Error")
        assert "roto" not in proyectos._EN_MARCHA
    finally:
        proyectos._EN_MARCHA.discard("roto")


def test_nunca_se_invoca_un_shell() -> None:
    """La otra regla de la casa. Un `shell=True` aquí convertiría la lista del
    disco en una línea que el sistema vuelve a parsear.

    Se mira el árbol y no el texto: la cabecera del módulo habla de `shell=True`
    justo para explicar por qué no lo hay, y buscar la cadena encontraría eso.
    """
    import ast
    import inspect

    arbol = ast.parse(inspect.getsource(proyectos))
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.Call):
            continue
        for argumento in nodo.keywords:
            assert argumento.arg != "shell" or not getattr(
                argumento.value, "value", False
            ), "hay un shell=True en proyectos.py"
