"""La biometría: que reconozca a quien conoce y aprenda a quien no.

Lo que se comprueba aquí es la **decisión**, no el modelo: con motores falsos
que devuelven vectores fijos se prueba todo el camino —perfiles en disco,
umbrales, racimos de desconocidos, aprendizaje, renombrar y borrar— sin torch
ni cv2 ni red. Si esto pasa, lo único que puede fallar en producción es el
vector que meten los motores reales, y eso lo decide ECAPA y SFace, no este
código.
"""

from __future__ import annotations

import base64
import struct
from pathlib import Path

import pytest

from perseo_core import biometria


# --------------------------------------------------------------------------- #
# Dobles de los motores
# --------------------------------------------------------------------------- #


class MotorVozFalso:
    """Devuelve el vector i-ésimo de su lista; si se acaba, repite el último."""

    def __init__(self, *vectores: list[float]) -> None:
        self.vectores = list(vectores)
        self.llamadas = 0

    def incrustar(self, muestras: list[float]) -> list[float]:
        self.llamadas += 1
        indice = min(self.llamadas - 1, len(self.vectores) - 1)
        return self.vectores[indice]


class MotorCaraFalso:
    """Lo mismo, con cajas: cada llamada devuelve la siguiente detección."""

    def __init__(self, *detecciones: tuple[list[int], list[float]]) -> None:
        self.detecciones = list(detecciones)
        self.llamadas = 0

    def detectar(self, jpeg: bytes) -> list[dict]:
        self.llamadas += 1
        if not self.detecciones:
            return []
        indice = min(self.llamadas - 1, len(self.detecciones) - 1)
        caja, vector = self.detecciones[indice]
        return [{"caja": caja, "vector": vector}]


def _audio_falso(muestras: int = 16000) -> str:
    """PCM int16 base64 del tamaño pedido. Al motor falso le da igual el valor."""
    return base64.b64encode(struct.pack(f"<{muestras}h", *([100] * muestras))).decode()


def _imagen_falsa() -> str:
    return base64.b64encode(b"jpeg-de-mentira").decode()


@pytest.fixture(autouse=True)
def _limpio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Motores fuera y sesión vacía antes y después de cada prueba."""
    biometria.pon_motores(None, None)
    biometria.reiniciar_sesion()
    yield
    biometria.pon_motores(None, None)
    biometria.reiniciar_sesion()


# --------------------------------------------------------------------------- #
# Álgebra y almacen
# --------------------------------------------------------------------------- #


def test_el_coseno_sabe_de_angulos() -> None:
    mismo = [1.0, 0.0]
    assert biometria._coseno(mismo, [2.0, 0.0]) == pytest.approx(1.0)
    assert biometria._coseno(mismo, [0.0, 5.0]) == pytest.approx(0.0)
    assert biometria._coseno(mismo, [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cargar_sin_fichero_es_vacio_y_no_rompe(tmp_path: Path) -> None:
    assert biometria.cargar(tmp_path) == {}


def test_un_fichero_roto_se_ignora_y_se_repone(tmp_path: Path) -> None:
    biometria.ruta_perfiles(tmp_path).write_text("{esto no es json", encoding="utf-8")
    assert biometria.cargar(tmp_path) == {}
    # Y una escritura nueva lo deja sano otra vez (atómica: tmp + replace).
    biometria.guardar(tmp_path, {"Alguien": {"voz": [1.0], "caras": []}})
    assert list(biometria.cargar(tmp_path)) == ["Alguien"]


def test_el_resumen_no_ensena_vectores(tmp_path: Path) -> None:
    biometria.guardar(
        tmp_path,
        {"Javi": {"voz": [0.1] * 192, "caras": [[0.2] * 128], "muestras": 3, "creado": "hoy"}},
    )
    ficha = biometria.resumen(biometria.cargar(tmp_path))[0]
    assert ficha["nombre"] == "Javi"
    assert ficha["caras"] == 1 and ficha["voz"] is True
    assert all(not isinstance(valor, list) or valor == [] for valor in ficha.values())


# --------------------------------------------------------------------------- #
# Identificar voz
# --------------------------------------------------------------------------- #


def test_identifica_a_quien_tiene_perfil(tmp_path: Path) -> None:
    vector = [1.0, 0.0, 0.0]
    biometria.guardar(
        tmp_path,
        {"Javi": {"voz": vector, "caras": [], "muestras": 1, "creado": "hoy"}},
    )
    biometria.pon_motores(voz=MotorVozFalso(vector))

    resultado = biometria.identificar_voz(tmp_path, _audio_falso())

    assert resultado["nombre"] == "Javi"
    assert resultado["confianza"] >= biometria.UMBRAL_VOZ
    # Cada acierto refuerza el perfil: la voz guardada tira hacia la de hoy.
    assert biometria.cargar(tmp_path)["Javi"]["muestras"] == 2


def test_por_debajo_del_umbral_aprende_al_desconocido(tmp_path: Path) -> None:
    conocido = [1.0, 0.0]
    otro = [0.0, 1.0]  # ortogonal: coseno 0, muy por debajo del umbral
    biometria.guardar(
        tmp_path,
        {"Persus": {"voz": conocido, "caras": [], "muestras": 1, "creado": "hoy"}},
    )

    # Primera frase del desconocido: ya lleva etiqueta provisional, y aprende.
    biometria.pon_motores(voz=MotorVozFalso(otro))
    trozo = _audio_falso(32000)  # dos segundos por llamada
    primero = biometria.identificar_voz(tmp_path, trozo)
    assert primero["nombre"] == "Desconocido 1"
    assert primero["aprendiendo"]["etiqueta"] == "Desconocido 1"
    assert primero["aprendiendo"]["peso"] == pytest.approx(2.0)

    # Frases siguientes del mismo vector engordan el MISMO racimo…
    for _ in range(3):
        resultado = biometria.identificar_voz(tmp_path, trozo)
    assert resultado["aprendiendo"]["peso"] == pytest.approx(8.0)

    # …hasta cruzar el objetivo y fijarse como perfil permanente.
    while "aprendiendo" in (ultimo := biometria.identificar_voz(tmp_path, trozo)):
        pass
    assert ultimo["nombre"] == "Desconocido 1"
    assert ultimo.get("aprendido") is True

    perfiles = biometria.cargar(tmp_path)
    assert "Desconocido 1" in perfiles
    assert perfiles["Desconocido 1"]["voz"]
    # Y a partir de ahora lo reconoce por su etiqueta, ya sin aprender nada.
    reconocido = biometria.identificar_voz(tmp_path, trozo)
    assert reconocido["nombre"] == "Desconocido 1"
    assert "aprendiendo" not in reconocido


def test_dos_desconocidos_distintos_abren_racimos_distintos(tmp_path: Path) -> None:
    uno, dos = [1.0, 0.0], [0.0, 1.0]
    biometria.pon_motores(voz=MotorVozFalso(uno, dos, uno, dos))

    primera = biometria.identificar_voz(tmp_path, _audio_falso())
    segunda = biometria.identificar_voz(tmp_path, _audio_falso())
    assert primera["nombre"] == "Desconocido 1"
    assert segunda["nombre"] == "Desconocido 2"


def test_audio_demasiado_corto_no_molesta(tmp_path: Path) -> None:
    biometria.pon_motores(voz=MotorVozFalso([1.0]))
    resultado = biometria.identificar_voz(tmp_path, _audio_falso(800))
    assert resultado == {"nombre": None}
    assert biometria.ruta_perfiles(tmp_path).exists() is False


def test_sin_motor_dice_que_falta_y_por_que(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Esta máquina puede tener speechbrain instalado, así que «no hay motor» se
    # simula: lo que importa es la RESPUESTA del módulo cuando el motor falta,
    # no que torch esté o no en el entorno.
    monkeypatch.setattr(biometria, "obtener_motor_voz", lambda _: None)
    resultado = biometria.identificar_voz(tmp_path, _audio_falso())
    assert "error" in resultado
    estado = biometria.estado_completo(tmp_path)
    assert estado["disponibilidad"]["voz"] is False
    assert estado["disponibilidad"]["motivo_voz"]


# --------------------------------------------------------------------------- #
# Identificar caras
# --------------------------------------------------------------------------- #


def test_reconoce_la_cara_conocida_y_devuelve_su_caja(tmp_path: Path) -> None:
    caja, vector = [10, 20, 100, 100], [1.0, 0.0]
    biometria.guardar(
        tmp_path,
        {"Fátima": {"voz": None, "caras": [vector], "muestras": 0, "creado": "hoy"}},
    )
    biometria.pon_motores(cara=MotorCaraFalso((caja, vector)))

    resultado = biometria.identificar_cara(tmp_path, _imagen_falsa())

    assert len(resultado["caras"]) == 1
    cara = resultado["caras"][0]
    assert cara["nombre"] == "Fátima"
    assert cara["caja"] == caja


def test_la_cara_desconocida_se_aprende_contando_detecciones(tmp_path: Path) -> None:
    vector = [0.0, 1.0]
    biometria.pon_motores(cara=MotorCaraFalso(([0, 0, 50, 50], vector)))

    for _ in range(biometria.FRAMES_CARA_APRENDER - 1):
        resultado = biometria.identificar_cara(tmp_path, _imagen_falsa())
        assert resultado["caras"][0]["nombre"] == "Desconocido 1"
        assert "aprendiendo" in resultado["caras"][0]

    final = biometria.identificar_cara(tmp_path, _imagen_falsa())
    assert final["caras"][0]["nombre"] == "Desconocido 1"
    perfiles = biometria.cargar(tmp_path)
    assert perfiles["Desconocido 1"]["caras"]


def test_varias_caras_en_un_cuadro_salen_todas(tmp_path: Path) -> None:
    class DosCaras(MotorCaraFalso):
        def detectar(self, jpeg: bytes) -> list[dict]:
            return [
                {"caja": [0, 0, 40, 40], "vector": [1.0, 0.0]},
                {"caja": [60, 0, 40, 40], "vector": [0.0, 1.0]},
            ]

    biometria.guardar(
        tmp_path,
        {"Persus": {"voz": None, "caras": [[1.0, 0.0]], "muestras": 0, "creado": "hoy"}},
    )
    biometria.pon_motores(cara=DosCaras())

    resultado = biometria.identificar_cara(tmp_path, _imagen_falsa())
    nombres = {cara["nombre"] for cara in resultado["caras"]}
    assert nombres == {"Persus", "Desconocido 1"}


# --------------------------------------------------------------------------- #
# Gestión manual
# --------------------------------------------------------------------------- #


def test_enrolar_manual_crea_el_perfil_y_lo_usa(tmp_path: Path) -> None:
    biometria.pon_motores(voz=MotorVozFalso([1.0, 0.0]))

    creado = biometria.enrolar(tmp_path, "Javi", audio=_audio_falso())
    assert creado["ok"] is True and "voz" in creado["añadido"]

    reconocido = biometria.identificar_voz(tmp_path, _audio_falso())
    assert reconocido["nombre"] == "Javi"


def test_enrolar_sin_caras_en_la_imagen_lo_dice(tmp_path: Path) -> None:
    biometria.pon_motores(cara=MotorCaraFalso())  # nunca devuelve detecciones
    resultado = biometria.enrolar(tmp_path, "Alguien", imagen=_imagen_falsa())
    assert resultado["error"] == "No se ve ninguna cara en la imagen"


def test_renombrar_mueve_el_perfil_entero(tmp_path: Path) -> None:
    biometria.guardar(
        tmp_path,
        {"Desconocido 1": {"voz": [1.0], "caras": [[0.5]], "muestras": 4, "creado": "hoy"}},
    )
    resultado = biometria.renombrar(tmp_path, "Desconocido 1", "Fátima")
    assert resultado["ok"] is True
    perfiles = biometria.cargar(tmp_path)
    assert "Desconocido 1" not in perfiles
    assert perfiles["Fátima"]["muestras"] == 4


def test_renombrar_choca_con_nombres_ocupados(tmp_path: Path) -> None:
    biometria.guardar(
        tmp_path,
        {
            "A": {"voz": [1.0], "caras": [], "creado": "hoy"},
            "B": {"voz": [0.0], "caras": [], "creado": "hoy"},
        },
    )
    assert biometria.renombrar(tmp_path, "A", "B")["error"].startswith("Ya existe")
    assert biometria.renombrar(tmp_path, "Z", "C")["error"].startswith("No hay")


def test_renombrar_fija_al_que_todavia_se_esta_aprendiendo(tmp_path: Path) -> None:
    """El nombre cierra el aprendizaje: no hay que esperar a los doce segundos.

    Es el caso de la llamada del 2026-08-25. Alguien entra, Perseo le pregunta
    cómo se llama y lo dice enseguida; si el nombre solo valiera para perfiles
    ya fijados, esa persona seguiría siendo «Desconocido» el resto de la
    conversación y la siguiente empezaría igual.
    """
    biometria.pon_motores(voz=MotorVozFalso([1.0, 0.0]))
    # Un solo trozo: el racimo existe pero está lejos de fijarse solo.
    primera = biometria.identificar_voz(tmp_path, _audio_falso())
    assert primera["nombre"] == "Desconocido 1"
    assert "aprendiendo" in primera

    resultado = biometria.renombrar(tmp_path, "Desconocido 1", "Antonio")
    assert resultado["ok"] is True
    assert resultado["nombre"] == "Antonio"

    perfiles = biometria.cargar(tmp_path)
    assert "Antonio" in perfiles and "Desconocido 1" not in perfiles
    # Y a partir de aquí se le reconoce por su nombre, no por la etiqueta.
    assert biometria.identificar_voz(tmp_path, _audio_falso())["nombre"] == "Antonio"


def test_nombrar_una_cara_a_medias_suma_al_perfil_que_ya_existe(tmp_path: Path) -> None:
    """La misma persona vista por otro canal no abre una ficha aparte."""
    biometria.guardar(
        tmp_path,
        {"Antonio": {"voz": [1.0, 0.0], "caras": [], "muestras": 1, "creado": "hoy"}},
    )
    biometria.pon_motores(cara=MotorCaraFalso(([0, 0, 40, 40], [0.0, 1.0])))
    biometria.identificar_cara(tmp_path, _imagen_falsa())

    assert biometria.renombrar(tmp_path, "Desconocido 1", "Antonio")["ok"] is True
    perfil = biometria.cargar(tmp_path)["Antonio"]
    assert perfil["voz"] == [1.0, 0.0]
    assert len(perfil["caras"]) == 1


def test_borrar_quita_los_vectores_de_verdad(tmp_path: Path) -> None:
    biometria.guardar(
        tmp_path,
        {"Javi": {"voz": [1.0] * 192, "caras": [], "creado": "hoy"}},
    )
    assert biometria.borrar(tmp_path, "Javi")["ok"] is True
    assert biometria.cargar(tmp_path) == {}
    assert biometria.borrar(tmp_path, "Javi")["error"].startswith("No hay")


def test_el_estado_completo_trae_todo_para_pintarlo(tmp_path: Path) -> None:
    biometria.pon_motores(voz=MotorVozFalso([1.0]), cara=MotorCaraFalso(([0, 0, 9, 9], [1.0])))
    biometria.enrolar(tmp_path, "Javi", audio=_audio_falso())

    estado = biometria.estado_completo(tmp_path)
    assert [p["nombre"] for p in estado["perfiles"]] == ["Javi"]
    assert estado["disponibilidad"]["voz"] is True
    # El motor falso de cara está puesto, así que también se declara disponible.
    assert estado["disponibilidad"]["cara"] is True


# --------------------------------------------------------------------------- #
# La mitad legible del reconocimiento: la nota de la persona
# --------------------------------------------------------------------------- #


def test_ponerle_nombre_deja_nota_en_el_vault(tmp_path) -> None:
    """Los vectores no le dicen nada a nadie; la nota sí, y se corrige a mano."""
    import asyncio

    from perseo_core import memoria

    vault = memoria.VaultFicheros(tmp_path)
    memoria._vault = vault
    try:
        ruta = asyncio.run(memoria.anotar_persona("Desconocido 3", "Javi"))
    finally:
        memoria._vault = None

    assert ruta.startswith(memoria.CARPETA_PERSONAS)
    escrito = (tmp_path / ruta).read_text(encoding="utf-8")
    assert "Desconocido 3" in escrito and "Javi" in escrito


def test_sin_memoria_iniciada_lo_dice(monkeypatch) -> None:
    import asyncio

    from perseo_core import memoria

    memoria._vault = None
    try:
        asyncio.run(memoria.anotar_persona("Desconocido 1", "Nadie"))
    except RuntimeError as e:
        assert "iniciada" in str(e)
    else:
        raise AssertionError("tenía que quejarse")
