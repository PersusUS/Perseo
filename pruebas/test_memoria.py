"""La memoria: añade, no sobrescribe, y no sale del vault."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from perseo_core import memoria

#: Una ruta absoluta que existe fuera del vault, sea cual sea el sistema. Las
#: pruebas corren en Windows y en el CI de Linux, y `C:\Windows` en Linux no es
#: una ruta absoluta: es un nombre de fichero con barras invertidas, que sí cae
#: dentro del vault y haría pasar la prueba por el motivo equivocado.
FUERA_DEL_DISCO = r"C:\Windows\win.ini" if os.name == "nt" else "/etc/passwd"
SUBIR_DOS = r"..\..\secreto.md" if os.name == "nt" else "../../secreto.md"


@pytest.fixture()
def vault_ficheros(vault: Path) -> memoria.VaultFicheros:
    return memoria.VaultFicheros(vault)


def test_anotar_crea_la_nota_con_cabecera(vault_ficheros, vault: Path) -> None:
    ruta = asyncio.run(vault_ficheros.anotar("Reforma", "Se cambia el plato de ducha"))
    contenido = (vault / ruta).read_text(encoding="utf-8")

    assert contenido.startswith("---\n")
    assert "Se cambia el plato de ducha" in contenido
    assert ruta.startswith(memoria.CARPETA_MEMORIAS)


def test_anotar_dos_veces_no_borra_lo_anterior(vault_ficheros, vault: Path) -> None:
    """Sobrescribir una memoria es la pérdida que no se nota hasta meses después."""
    ruta = asyncio.run(vault_ficheros.anotar("Reforma", "lo primero"))
    asyncio.run(vault_ficheros.anotar("Reforma", "lo segundo"))

    contenido = (vault / ruta).read_text(encoding="utf-8")
    assert "lo primero" in contenido
    assert "lo segundo" in contenido
    assert contenido.count("\n## ") >= 2


def test_anotar_rechaza_un_titulo_sin_nombre_util(vault_ficheros) -> None:
    with pytest.raises(ValueError):
        asyncio.run(vault_ficheros.anotar("///", "texto"))


def test_buscar_encuentra_por_contenido(vault_ficheros) -> None:
    asyncio.run(vault_ficheros.anotar("Reforma", "el plato de ducha"))
    encontradas = asyncio.run(vault_ficheros.buscar("plato de ducha"))
    assert len(encontradas) == 1
    assert encontradas[0].titulo == "Reforma"


def test_buscar_encuentra_por_nombre_de_nota(vault_ficheros) -> None:
    asyncio.run(vault_ficheros.anotar("Aurelio", "un amigo"))
    assert len(asyncio.run(vault_ficheros.buscar("aureli"))) == 1


def test_las_tildes_no_estorban(vault_ficheros, vault: Path) -> None:
    """Buscar con tildes exactas en un vault escrito a mano no encuentra nada."""
    (vault / "Cumple.md").write_text("El cumpleaños es en marzo", encoding="utf-8")
    assert len(asyncio.run(vault_ficheros.buscar("cumpleanos"))) == 1
    assert len(asyncio.run(vault_ficheros.buscar("CUMPLEAÑOS"))) == 1


def test_buscar_lo_que_no_esta(vault_ficheros) -> None:
    asyncio.run(vault_ficheros.anotar("Reforma", "algo"))
    assert asyncio.run(vault_ficheros.buscar("hipopotamo")) == []


def test_buscar_con_consulta_vacia_no_devuelve_el_vault_entero(vault_ficheros) -> None:
    asyncio.run(vault_ficheros.anotar("Reforma", "algo"))
    assert asyncio.run(vault_ficheros.buscar("   ")) == []


def test_buscar_respeta_el_limite(vault_ficheros) -> None:
    for n in range(5):
        asyncio.run(vault_ficheros.anotar(f"Nota {n}", "comun"))
    assert len(asyncio.run(vault_ficheros.buscar("comun", limite=2))) == 2


def test_la_ruta_que_devuelve_es_relativa(vault_ficheros) -> None:
    asyncio.run(vault_ficheros.anotar("Reforma", "algo"))
    nota = asyncio.run(vault_ficheros.buscar("algo"))[0]
    assert not Path(nota.ruta).is_absolute()


def test_leer_devuelve_el_contenido(vault_ficheros) -> None:
    ruta = asyncio.run(vault_ficheros.anotar("Reforma", "el contenido"))
    assert "el contenido" in asyncio.run(vault_ficheros.leer(ruta))


def test_leer_lo_que_no_existe(vault_ficheros) -> None:
    with pytest.raises(FileNotFoundError):
        asyncio.run(vault_ficheros.leer("Memorias_Sistema/no_existe.md"))


@pytest.mark.parametrize(
    "intento",
    ["../secreto.md", SUBIR_DOS, "Notas/../../fuera.md", FUERA_DEL_DISCO],
)
def test_ninguna_ruta_sale_del_vault(vault_ficheros, intento: str) -> None:
    """Un `../` en un asunto de correo no puede acabar leyendo el disco."""
    with pytest.raises(memoria.FueraDelVault):
        asyncio.run(vault_ficheros.leer(intento))


def test_tampoco_al_anotar(vault_ficheros) -> None:
    with pytest.raises(memoria.FueraDelVault):
        asyncio.run(vault_ficheros.anotar("colada", "texto", carpeta="../fuera"))


def test_el_agente_busca(vault: Path, monkeypatch) -> None:
    monkeypatch.setattr(memoria, "_vault", memoria.VaultFicheros(vault))
    asyncio.run(memoria._vault.anotar("Reforma", "el plato de ducha"))

    resultado = asyncio.run(
        memoria._memoria({"peticion": {"accion": "buscar", "texto": "ducha"}})
    )
    assert len(resultado["notas"]) == 1
    assert resultado["titular"]


def test_el_agente_anota_y_devuelve_titular(vault: Path, monkeypatch) -> None:
    monkeypatch.setattr(memoria, "_vault", memoria.VaultFicheros(vault))
    resultado = asyncio.run(
        memoria._memoria({"peticion": {"accion": "anotar", "titulo": "X", "texto": "y"}})
    )
    assert resultado["ruta"].endswith("X.md")
    assert "Anotado" in resultado["titular"]


def test_el_agente_guarda_una_conversacion(vault: Path, monkeypatch) -> None:
    monkeypatch.setattr(memoria, "_vault", memoria.VaultFicheros(vault))
    resultado = asyncio.run(
        memoria._memoria(
            {
                "peticion": {
                    "accion": "conversacion",
                    "mensajes": [{"tipo": "user", "texto": "hola"}, {"tipo": "ai", "texto": "qué tal"}],
                }
            }
        )
    )
    assert resultado["mensajes"] == 2
    assert memoria.CARPETA_CONVERSACIONES in resultado["ruta"]
    # Una conversación no manda titular: no hay nada que avisar.
    assert resultado["titular"] is None


def test_una_conversacion_vacia_no_escribe_nada(vault: Path, monkeypatch) -> None:
    monkeypatch.setattr(memoria, "_vault", memoria.VaultFicheros(vault))
    resultado = asyncio.run(
        memoria._memoria({"peticion": {"accion": "conversacion", "mensajes": []}})
    )
    assert resultado["ruta"] is None
    assert list(vault.rglob("*.md")) == []


def test_el_agente_rechaza_una_accion_que_no_existe(vault: Path, monkeypatch) -> None:
    monkeypatch.setattr(memoria, "_vault", memoria.VaultFicheros(vault))
    with pytest.raises(ValueError):
        asyncio.run(memoria._memoria({"peticion": {"accion": "borrar"}}))


def test_anotar_sin_texto_se_rechaza(vault: Path, monkeypatch) -> None:
    monkeypatch.setattr(memoria, "_vault", memoria.VaultFicheros(vault))
    with pytest.raises(ValueError):
        asyncio.run(memoria._memoria({"peticion": {"accion": "anotar", "titulo": "X"}}))


# --------------------------------------------------------------------------- #
# El respaldo del plugin de Obsidian
#
# Aquí no hay red: lo que se comprueba es lo que se decide **antes** de mandar
# una petición —qué ruta se acepta, qué URL sale, qué respaldo se elige— que es
# justo lo que no puede depender de que Obsidian esté abierto. El camino HTTP
# entero, contra un plugin de mentira, está en `verificar_memoria.py`.
# --------------------------------------------------------------------------- #


@pytest.fixture()
def vault_rest() -> memoria.VaultRest:
    return memoria.VaultRest("https://127.0.0.1:27124", "clave-de-prueba")


@pytest.mark.parametrize(
    "intento",
    ["../secreto.md", SUBIR_DOS, "Notas/../../fuera.md", FUERA_DEL_DISCO, "/etc/passwd", ""],
)
def test_rest_ninguna_ruta_sale_del_vault(vault_rest, intento: str) -> None:
    """Sin disco que resolver, esta comprobación es la única que hay."""
    with pytest.raises(memoria.FueraDelVault):
        memoria._ruta_relativa(intento)


def test_rest_las_barras_invertidas_cuentan_como_separador() -> None:
    r"""`..\..\x` no puede colar en Linux por ser allí un nombre de fichero."""
    with pytest.raises(memoria.FueraDelVault):
        memoria._ruta_relativa(r"..\..\secreto.md")
    assert memoria._ruta_relativa(r"Memorias_Sistema\Reforma.md") == "Memorias_Sistema/Reforma.md"


def test_rest_una_ruta_normal_pasa_tal_cual() -> None:
    assert memoria._ruta_relativa("./Notas/Cumpleanos.md") == "Notas/Cumpleanos.md"


def test_rest_la_url_escapa_el_nombre_pero_no_la_jerarquia(vault_rest) -> None:
    url = vault_rest._url("/vault/", "Memorias Sistema/Reforma baño.md")
    assert url.startswith("https://127.0.0.1:27124/vault/")
    assert " " not in url and "ñ" not in url
    assert url.count("/vault/") == 1 and "Sistema/Reforma" in url.replace("%20", " ")


def test_rest_un_titulo_imposible_se_rechaza_antes_de_salir_a_la_red(vault_rest) -> None:
    with pytest.raises(ValueError):
        asyncio.run(vault_rest.anotar("///", "texto"))


def test_rest_leer_fuera_del_vault_no_llega_a_pedir_nada(vault_rest) -> None:
    with pytest.raises(memoria.FueraDelVault):
        asyncio.run(vault_rest.leer("../secreto.md"))


def test_rest_una_busqueda_vacia_no_pregunta(vault_rest) -> None:
    assert asyncio.run(vault_rest.buscar("   ")) == []


def test_rest_las_notas_salen_de_la_respuesta_del_plugin() -> None:
    crudas = [
        {"filename": "Notas/Cumpleanos.md", "matches": [{"context": "es en marzo"}]},
        {"filename": "Memorias_Sistema/Reforma.md", "matches": []},
    ]
    notas = memoria._notas_de_busqueda(crudas, 10)

    assert [n.ruta for n in notas] == ["Notas/Cumpleanos.md", "Memorias_Sistema/Reforma.md"]
    assert notas[0].titulo == "Cumpleanos"
    assert notas[0].extracto == "es en marzo"
    # El plugin no dice cuándo se modificó, y no se inventa una fecha.
    assert notas[0].modificada == ""


def test_rest_la_busqueda_respeta_el_limite_y_aguanta_basura() -> None:
    crudas = ["no soy un dict", {"sin": "filename"}, {"filename": "a.md"}, {"filename": "b.md"}]
    assert [n.ruta for n in memoria._notas_de_busqueda(crudas, 1)] == ["a.md"]
    assert memoria._notas_de_busqueda("esto tampoco es una lista", 10) == []


@pytest.mark.parametrize(
    ("base", "verifica"),
    [
        ("https://127.0.0.1:27124", False),
        ("https://localhost:27124", False),
        ("http://127.0.0.1:27123", True),
        ("https://obsidian.example.com", True),
    ],
)
def test_rest_el_certificado_solo_se_deja_pasar_en_el_bucle_local(base: str, verifica: bool) -> None:
    """El plugin firma su propio certificado; fuera de casa eso no vale."""
    assert memoria._verificar_certificado(base) is verifica


def test_los_dos_respaldos_escriben_el_mismo_markdown(vault: Path) -> None:
    """Si uno de los dos cambia de formato, esto se entera antes que el vault."""
    ruta = asyncio.run(memoria.VaultFicheros(vault).anotar("Reforma", "lo primero"))
    escrito = (vault / ruta).read_text(encoding="utf-8")
    momento = escrito.splitlines()[3].removeprefix("fecha_creacion: ")

    assert escrito == memoria._nota_nueva("Reforma", "lo primero", momento)


def test_el_respaldo_por_defecto_son_ficheros(cfg, tmp_path: Path, monkeypatch) -> None:
    from dataclasses import replace

    monkeypatch.setattr(memoria, "_vault", None)
    elegido = memoria.iniciar(replace(cfg, vault=str(tmp_path / "v")))
    assert isinstance(elegido, memoria.VaultFicheros)


def test_con_rest_y_clave_se_usa_el_plugin(cfg, tmp_path: Path, monkeypatch) -> None:
    from dataclasses import replace

    monkeypatch.setattr(memoria, "_vault", None)
    elegido = memoria.iniciar(
        replace(cfg, vault=str(tmp_path / "v"), vault_respaldo="rest", vault_rest_clave="k")
    )
    assert isinstance(elegido, memoria.VaultRest)
    asyncio.run(memoria.detener())


def test_rest_sin_clave_no_deja_al_nucleo_sin_memoria(cfg, tmp_path: Path, monkeypatch) -> None:
    """No configurado no es lo mismo que roto: se avisa y se sigue con ficheros."""
    from dataclasses import replace

    monkeypatch.setattr(memoria, "_vault", None)
    elegido = memoria.iniciar(
        replace(cfg, vault=str(tmp_path / "v"), vault_respaldo="rest", vault_rest_clave="")
    )
    assert isinstance(elegido, memoria.VaultFicheros)
