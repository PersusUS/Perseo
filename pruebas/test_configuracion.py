

def test_lo_de_entorno_json_llega_al_entorno_sin_pisar_la_terminal(datos, monkeypatch) -> None:
    """Hay piezas que miran el entorno a pelo; lo de `entorno.json` tiene que llegarles."""
    import json
    import os

    from perseo_core.infra.configuracion import exportar_ajustes

    (datos / "entorno.json").write_text(
        json.dumps({"PERSEO_PRUEBA_FICHERO": "1", "PERSEO_PRUEBA_TERMINAL": "del fichero"}), encoding="utf-8"
    )
    monkeypatch.delenv("PERSEO_PRUEBA_FICHERO", raising=False)
    monkeypatch.setenv("PERSEO_PRUEBA_TERMINAL", "de la terminal")
    puestas = exportar_ajustes(datos)
    try:
        assert os.environ["PERSEO_PRUEBA_FICHERO"] == "1"
        assert os.environ["PERSEO_PRUEBA_TERMINAL"] == "de la terminal"
        assert puestas == ["PERSEO_PRUEBA_FICHERO"]
    finally:
        os.environ.pop("PERSEO_PRUEBA_FICHERO", None)
