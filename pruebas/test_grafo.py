"""El grafo del segundo cerebro: qué es un nodo, qué es una arista, qué no.

Las reglas están en la cabecera de `perseo_core/grafo.py` y son las de
Obsidian con una excepción (los enlaces a notas sin crear no dibujan nodo).
Estas pruebas las clavan para que nadie las «simplifique» sin darse cuenta.
"""

from __future__ import annotations

from pathlib import Path

from perseo_core import grafo


def _vault(tmp_path: Path, ficheros: dict[str, str]) -> Path:
    for ruta, texto in ficheros.items():
        destino = tmp_path / ruta
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(texto, encoding="utf-8")
    return tmp_path


def _ids(datos: dict) -> set[str]:
    return {n["id"] for n in datos["nodos"]}


def test_un_enlace_sencillo_es_una_arista(tmp_path: Path) -> None:
    vault = _vault(tmp_path, {
        "a.md": "Esto enlaza con [[b]].",
        "b.md": "Y b no enlaza a nadie.",
    })
    datos = grafo.construir(vault)
    assert _ids(datos) == {"a", "b"}
    assert {"a": "a", "b": "b"} | {} and datos["enlaces"] == [{"a": "a", "b": "b"}]


def test_el_alias_y_el_ancla_no_cambian_el_destino(tmp_path: Path) -> None:
    """`[[b|con otro nombre]]` y `[[b#punto]]` enlazan a `b`, y solo a `b`."""
    vault = _vault(tmp_path, {
        "a.md": "Vale [[b|lo que sea]] y también [[b#un punto]].",
        "b.md": "",
    })
    datos = grafo.construir(vault)
    assert datos["enlaces"] == [{"a": "a", "b": "b"}]
    # Dos enlaces al mismo destino son UNA conexión, no dos.
    assert datos["nodos"][0]["grado"] == 1 or datos["nodos"][1]["grado"] == 1


def test_las_mayusculas_no_separan_las_notas(tmp_path: Path) -> None:
    """Como en Obsidian: `[[B]]` resuelve contra `b.md`."""
    vault = _vault(tmp_path, {"a.md": "Miro a [[B]].", "b.md": ""})
    datos = grafo.construir(vault)
    assert datos["enlaces"] == [{"a": "a", "b": "b"}]


def test_un_enlace_a_nota_sin_crear_es_un_fantasma(tmp_path: Path) -> None:
    """La excepción con el grafo de Obsidian: no se dibuja, pero se cuenta."""
    vault = _vault(tmp_path, {"a.md": "Hablo de [[fantasma]] y de [[b]].", "b.md": ""})
    datos = grafo.construir(vault)
    assert _ids(datos) == {"a", "b"}
    assert datos["fantasmas"] == 1
    assert {"a": "a", "b": "b"} in [dict(e) for e in datos["enlaces"]]
    # `a` tiene un enlace firme con `b`: no es huérfana aunque otro de sus
    # enlaces acabe en el vacío.
    assert datos["apartadas"] == 0


def test_una_nota_que_se_menciona_a_si_misma_no_se_enlaza(tmp_path: Path) -> None:
    vault = _vault(tmp_path, {"a.md": "Yo mismo: [[a]].", "b.md": ""})
    datos = grafo.construir(vault)
    assert datos["enlaces"] == []


def test_las_notas_huerfanas_se_apartan(tmp_path: Path) -> None:
    """La enmienda del 2026-08-24: solo se dibuja lo que está conectado.

    Las sueltas no desaparecen callando: se cuentan en `apartadas`, y el
    total del vault sigue diciendo cuántas notas hay de verdad.
    """
    vault = _vault(tmp_path, {
        "a.md": "[[b]]",
        "b.md": "[[a]]",
        "suelta.md": "No enlazo con nadie.",
        "otra-suelta.md": "",
    })
    datos = grafo.construir(vault)
    assert _ids(datos) == {"a", "b"}
    assert datos["total_notas"] == 4
    assert datos["apartadas"] == 2


def test_una_nota_que_solo_enlaza_a_fantasma_tambien_se_aparta(tmp_path: Path) -> None:
    """Su único enlace acaba en una nota que no existe: ni arista dibujada
    ni nodo al que agarrarse — flotaría sola con un hilo roto."""
    vault = _vault(tmp_path, {
        "a.md": "[[b]]",
        "b.md": "",
        "flotante.md": "Solo hablo de [[fantasma]].",
    })
    datos = grafo.construir(vault)
    assert _ids(datos) == {"a", "b"}
    assert "fantasma" not in _ids(datos)
    assert datos["apartadas"] == 1
    assert datos["fantasmas"] == 1


def test_las_carpetas_internas_de_obsidian_no_son_cerebro(tmp_path: Path) -> None:
    vault = _vault(tmp_path, {
        "a.md": "[[b]]",
        "b.md": "",
        ".obsidian/plugins/x.md": "[[a]]",
        ".trash/viejo.md": "[[a]]",
    })
    datos = grafo.construir(vault)
    assert _ids(datos) == {"a", "b"}
    assert datos["total_notas"] == 2


def test_las_notas_de_subcarpetas_cuentan(tmp_path: Path) -> None:
    vault = _vault(tmp_path, {
        "a.md": "[[proyectos/idea]]",
        "proyectos/idea.md": "Aquí estoy.",
    })
    datos = grafo.construir(vault)
    assert "idea" in _ids(datos)
    assert {"a": "a", "b": "idea"} in [dict(e) for e in datos["enlaces"]] or \
           datos["enlaces"] == [{"a": "a", "b": "idea"}]


def test_un_vault_que_no_existe_no_rompe(tmp_path: Path) -> None:
    datos = grafo.construir(tmp_path / "no-esta")
    assert datos["nodos"] == [] and datos["enlaces"] == []


def test_por_encima_del_tope_se_quedan_las_encrucijadas(tmp_path: Path, monkeypatch) -> None:
    """Con más notas de las que se dibujan bien, se quedan las más conectadas."""
    monkeypatch.setattr(grafo, "TOPE_NODOS", 3)
    ficheros = {"centro.md": "".join(f"[[n{i}]] " for i in range(10))}
    ficheros.update({f"n{i}.md": "[[centro]]" for i in range(10)})
    datos = grafo.construir(_vault(tmp_path, ficheros))
    assert len(datos["nodos"]) == 3
    assert "centro" in _ids(datos)  # la encrucijada nunca se recorta


def test_la_cache_dura_un_rato_y_luego_suelta(tmp_path: Path, monkeypatch) -> None:
    """Mientras la caché vale, cambiar el disco no se ve; al caducar, sí."""
    monkeypatch.setattr(grafo, "TTL_SEGUNDOS", 0.0)  # caduca al instante
    vault = _vault(tmp_path, {"a.md": "[[b]]", "b.md": ""})
    assert len(grafo.construir(vault)["enlaces"]) == 1
    (vault / "c.md").write_text("[[a]]", encoding="utf-8")
    datos = grafo.construir(vault)
    assert "c" in _ids(datos)
    grafo.invalidar()  # y por si alguien la quiere tira a mano


# --------------------------------------------------------------------------- #
# Abrir una nota en Obsidian
# --------------------------------------------------------------------------- #


def test_cada_nodo_lleva_su_ruta_y_el_vault_su_nombre(tmp_path: Path) -> None:
    """Lo que Obsidian necesita para abrir una nota: su ruta relativa sin
    extensión, y el nombre del vault para el URI."""
    vault = _vault(tmp_path, {"carpeta/nota.md": "[[otra]]", "otra.md": ""})
    datos = grafo.construir(vault)
    rutas = {n["id"]: n["ruta"] for n in datos["nodos"]}
    assert rutas == {"nota": "carpeta/nota", "otra": "otra"}
    assert datos["vault"] == vault.name


def test_abrir_nota_lo_monta_y_lo_abre(tmp_path: Path, monkeypatch) -> None:
    """Pulsar un nodo acaba en un `obsidian://open` con vault y fichero."""
    vault = _vault(tmp_path, {"carpeta/Mi Nota.md": "", "otra.md": ""})
    abiertas: list[str] = []
    monkeypatch.setattr(grafo.webbrowser, "open", lambda url: abiertas.append(url) or True)

    texto = grafo.abrir_nota(vault, "mi nota")

    assert texto.startswith("Éxito:")
    assert len(abiertas) == 1
    assert abiertas[0].startswith("obsidian://open?vault=")
    assert vault.name in abiertas[0]
    assert "carpeta/Mi%20Nota" in abiertas[0]


def test_abrir_nota_no_distingue_mayusculas(tmp_path: Path, monkeypatch) -> None:
    """Como en el grafo y en Obsidian: el título se compara relajado."""
    vault = _vault(tmp_path, {"Agenda.md": ""})
    abiertas: list[str] = []
    monkeypatch.setattr(grafo.webbrowser, "open", lambda url: abiertas.append(url) or True)
    assert grafo.abrir_nota(vault, "AGENDA").startswith("Éxito:")
    assert abiertas and "Agenda" in abiertas[0]


def test_dos_notas_con_el_mismo_titulo_gana_la_menos_profunda(
    tmp_path: Path, monkeypatch
) -> None:
    """El título repetido resuelve como Obsidian: la ruta más corta primero."""
    vault = _vault(tmp_path, {"archivo/x.md": "", "x.md": "", "y/y/x.md": ""})
    abiertas: list[str] = []
    monkeypatch.setattr(grafo.webbrowser, "open", lambda url: abiertas.append(url) or True)
    assert grafo.abrir_nota(vault, "x").startswith("Éxito:")
    assert abiertas and abiertas[0].endswith("file=x")


def test_abrir_una_nota_que_no_esta_es_un_error(tmp_path: Path) -> None:
    vault = _vault(tmp_path, {"a.md": ""})
    assert grafo.abrir_nota(vault, "fantasma").startswith("Error:")


def test_abrir_sin_nota_o_sin_vault_es_un_error(tmp_path: Path) -> None:
    assert grafo.abrir_nota(tmp_path / "no-esta", "a").startswith("Error:")
    assert grafo.abrir_nota(_vault(tmp_path / "v", {"a.md": ""}), "   ").startswith("Error:")


def test_abrir_un_fallo_del_navegador_no_tumba_nada(tmp_path: Path, monkeypatch) -> None:
    """Que Obsidian no responda es un aviso en pantalla, no una excepción."""

    def revienta(url: str) -> bool:
        raise OSError("no hay asociación")

    vault = _vault(tmp_path, {"a.md": ""})
    monkeypatch.setattr(grafo.webbrowser, "open", revienta)
    assert grafo.abrir_nota(vault, "a").startswith("Error:")
