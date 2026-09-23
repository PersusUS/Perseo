"""La biometría de después del 2026-09-23: turnos, galería, cámara que ayuda.

Cada prueba de aquí es un fallo que había, o la regla que lo impide:

  · el contador de «Desconocido N» volvía a cero y **pisaba** un perfil guardado;
  · un acierto por los pelos arrastraba medio perfil hacia el último trozo;
  · decidir trozo a trozo hacía bailar la etiqueta;
  · la misma persona acababa con una ficha de voz y otra de cara;
  · una cara movida o de lado se quedaba para siempre en la galería;
  · un mal fotograma de alguien conocido abría un «Desconocido» con su cara.

Mismos dobles que `test_biometria.py`: vectores fijos, sin torch ni cv2.
"""

from __future__ import annotations

import base64
import math
import struct
from pathlib import Path

import pytest

from perseo_core.servicios import biometria, biometria_galeria, biometria_sesion


class MotorVozFalso:
    def __init__(self, *vectores: list[float]) -> None:
        self.vectores = list(vectores)
        self.llamadas = 0

    def incrustar(self, muestras: list[float]) -> list[float]:
        self.llamadas += 1
        return self.vectores[min(self.llamadas - 1, len(self.vectores) - 1)]


class MotorCaraFalso:
    """Cada llamada devuelve la siguiente lista de detecciones (dicts enteros)."""

    def __init__(self, *cuadros: list[dict]) -> None:
        self.cuadros = list(cuadros)
        self.llamadas = 0

    def detectar(self, jpeg: bytes) -> list[dict]:
        self.llamadas += 1
        return self.cuadros[min(self.llamadas - 1, len(self.cuadros) - 1)]


def _voz(segundos: float = 2.0) -> str:
    muestras = int(16000 * segundos)
    valores = [
        int(8000 * (0.2 + 0.8 * abs(math.sin(2 * math.pi * 3 * i / 16000))) * math.sin(i * 0.06))
        for i in range(muestras)
    ]
    return base64.b64encode(struct.pack(f"<{muestras}h", *valores)).decode()


IMAGEN = base64.b64encode(b"jpeg-de-mentira").decode()


def _con_coseno(c: float) -> list[float]:
    """Un vector que forma ese coseno con [1, 0, 0]."""
    return [c, math.sqrt(1 - c * c), 0.0]


def _cara(vector: list[float], caja: list[int] | None = None, **calidad) -> dict:
    return {"caja": caja or [100, 100, 80, 80], "vector": vector, **calidad}


class Reloj:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture(autouse=True)
def _limpio(monkeypatch: pytest.MonkeyPatch):
    biometria.pon_motores(None, None)
    biometria.reiniciar_sesion()
    reloj = Reloj()
    monkeypatch.setattr(biometria_sesion, "reloj", reloj)
    yield reloj
    biometria.pon_motores(None, None)
    biometria.reiniciar_sesion()


def _perfil(voz=None, caras=None, muestras=1) -> dict:
    return {"voz": voz, "caras": caras or [], "muestras": muestras, "creado": "hoy"}


# --------------------------------------------------------------------------- #
# La galería
# --------------------------------------------------------------------------- #


def test_el_centro_es_una_media_de_verdad() -> None:
    """Con `_media(a, b)` el último pesaba lo mismo que todos los anteriores juntos."""
    centro = biometria_galeria.centroide([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert centro[0] == pytest.approx(2 / math.sqrt(5))
    assert centro[1] == pytest.approx(1 / math.sqrt(5))


def test_lo_que_no_aporta_no_entra_en_la_galeria() -> None:
    galeria = [[1.0, 0.0]]
    assert biometria_galeria.incorporar(galeria, [0.999, 0.01]) is False
    assert biometria_galeria.incorporar(galeria, [0.7, 0.7]) is True
    assert len(galeria) == 2


def test_la_galeria_llena_cambia_la_mas_redundante() -> None:
    # Tres casi iguales y una distinta: se va una de las tres, no la distinta.
    galeria = [[1.0, 0.0, 0.0], [0.99, 0.14, 0.0], [0.99, -0.14, 0.0], [0.0, 1.0, 0.0]]
    nueva = [0.0, 0.0, 1.0]
    assert biometria_galeria.incorporar(galeria, nueva, maximo=4) is True
    assert nueva in galeria
    assert [0.0, 1.0, 0.0] in galeria
    assert len(galeria) == 4


def test_puntuar_promedia_los_tres_mejores() -> None:
    galeria = [[1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [0.8, 0.6]]
    # Contra [1, 0]: 1.0, 0.8, 0.6 y 0.0 → los tres mejores dan 0.8.
    assert biometria_galeria.puntuar(galeria, [1.0, 0.0]) == pytest.approx(0.8)


# --------------------------------------------------------------------------- #
# El contador que pisaba perfiles
# --------------------------------------------------------------------------- #


def test_un_desconocido_nuevo_no_pisa_al_que_ya_esta_en_disco(tmp_path: Path) -> None:
    guardado = _perfil(caras=[[1.0, 0.0]])
    biometria.guardar(tmp_path, {"Desconocido 1": guardado})
    biometria.reiniciar_sesion()  # como un núcleo recién arrancado
    biometria.pon_motores(voz=MotorVozFalso([0.0, 1.0]))

    assert biometria.identificar_voz(tmp_path, _voz())["nombre"] == "Desconocido 2"
    assert biometria.renombrar(tmp_path, "Desconocido 2", "Ana")["ok"] is True

    perfiles = biometria.cargar(tmp_path)
    assert perfiles["Desconocido 1"]["caras"] == [[1.0, 0.0]]
    assert perfiles["Ana"]["voces"]


def test_dar_de_alta_no_reinicia_el_contador(tmp_path: Path) -> None:
    biometria.pon_motores(voz=MotorVozFalso([0.0, 1.0], [1.0, 0.0], [0.0, 0.0, 1.0]))
    assert biometria.identificar_voz(tmp_path, _voz())["nombre"] == "Desconocido 1"
    biometria.enrolar(tmp_path, "Javi", audio=_voz(3.0))
    # El racimo a medias se fue con el alta, pero el número no se repite.
    assert biometria.identificar_voz(tmp_path, _voz())["nombre"] == "Desconocido 2"


# --------------------------------------------------------------------------- #
# El turno: decidir con la media, no con el último trozo
# --------------------------------------------------------------------------- #


def test_un_trozo_dudoso_no_nombra_ni_aprende(tmp_path: Path) -> None:
    biometria.guardar(tmp_path, {"Javi": _perfil(voz=[1.0, 0.0, 0.0])})
    biometria.pon_motores(voz=MotorVozFalso(_con_coseno(0.47)))

    r = biometria.identificar_voz(tmp_path, _voz())
    assert r["nombre"] is None and r["dudoso"] == "Javi"
    # Y sobre todo no abre un racimo con la voz de Javi.
    assert biometria.estado_completo(tmp_path)["aprendiendo"]["voz"] is None


def test_la_media_del_turno_resuelve_la_duda(tmp_path: Path) -> None:
    biometria.guardar(tmp_path, {"Javi": _perfil(voz=[1.0, 0.0, 0.0])})
    biometria.pon_motores(voz=MotorVozFalso(_con_coseno(0.50), _con_coseno(0.66)))

    assert biometria.identificar_voz(tmp_path, _voz())["nombre"] is None
    segundo = biometria.identificar_voz(tmp_path, _voz())
    assert segundo["nombre"] == "Javi"
    assert segundo["confianza"] == pytest.approx(0.58, abs=0.01)


def test_una_pausa_larga_empieza_turno_nuevo(tmp_path: Path, _limpio) -> None:
    biometria.guardar(tmp_path, {"Javi": _perfil(voz=[1.0, 0.0, 0.0])})
    # 0,64 nombra sin reforzar la galería (el refuerzo pide 0,65), así que lo
    # único que cambia entre los dos trozos es si comparten turno.
    biometria.pon_motores(voz=MotorVozFalso(_con_coseno(0.64), _con_coseno(0.47)))

    assert biometria.identificar_voz(tmp_path, _voz())["nombre"] == "Javi"
    _limpio.t += biometria_sesion.PAUSA_TURNO + 1
    # Sin la pausa la media sería 0,555 y diría Javi; con ella, 0,47 solo duda.
    assert biometria.identificar_voz(tmp_path, _voz())["nombre"] is None


def test_otra_voz_conocida_corta_el_turno_aunque_la_media_diga_otra_cosa(tmp_path: Path) -> None:
    biometria.guardar(
        tmp_path,
        {"Javi": _perfil(voz=[1.0, 0.0, 0.0]), "Ana": _perfil(voz=[0.0, 0.0, 1.0])},
    )
    biometria.pon_motores(voz=MotorVozFalso([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]))
    assert biometria.identificar_voz(tmp_path, _voz())["nombre"] == "Javi"
    assert biometria.identificar_voz(tmp_path, _voz())["nombre"] == "Ana"


def test_un_acierto_por_los_pelos_no_toca_la_galeria(tmp_path: Path) -> None:
    """Lo que antes arrastraba medio perfil: pasar el umbral no basta para enseñar."""
    biometria.guardar(tmp_path, {"Javi": _perfil(voz=[1.0, 0.0, 0.0])})
    biometria.pon_motores(voz=MotorVozFalso(_con_coseno(0.60), _con_coseno(0.80)))

    assert biometria.identificar_voz(tmp_path, _voz())["nombre"] == "Javi"
    assert "voces" not in biometria.cargar(tmp_path)["Javi"]

    assert biometria.identificar_voz(tmp_path, _voz())["nombre"] == "Javi"
    perfil = biometria.cargar(tmp_path)["Javi"]
    assert perfil["voces"] == [_con_coseno(0.80)]
    assert perfil["muestras"] == 2


def test_un_perfil_de_antes_se_lee_y_se_migra(tmp_path: Path) -> None:
    """Los perfiles guardados solo tenían `voz`: esa es la primera de la galería."""
    biometria.guardar(tmp_path, {"Persus": {"voz": [1.0, 0.0, 0.0], "caras": [], "muestras": 8}})
    biometria.pon_motores(voz=MotorVozFalso(_con_coseno(0.8)))

    assert biometria.resumen(biometria.cargar(tmp_path))[0]["voz_antigua"] is True
    assert biometria.identificar_voz(tmp_path, _voz())["nombre"] == "Persus"
    perfil = biometria.cargar(tmp_path)["Persus"]
    # El vector antiguo se hizo con el refuerzo roto y pudo tragarse otras
    # voces: la primera muestra buena lo sustituye en vez de sumarse a él.
    assert perfil["voces"] == [_con_coseno(0.8)]
    ficha = biometria.resumen({"Persus": perfil})[0]
    assert ficha["voces"] == 1 and ficha["voz_antigua"] is False


# --------------------------------------------------------------------------- #
# Voz y cara juntas
# --------------------------------------------------------------------------- #


def test_la_camara_no_decide_quien_habla(tmp_path: Path) -> None:
    """El dueño delante de su portátil y una visita hablando fuera de plano.

    Hasta el 2026-09-23 una voz dudosa con el dueño solo en cámara era él, y
    así se llamaba «señor Persus» a la visita. La cámara dice quién está.
    """
    biometria.guardar(tmp_path, {"Javi": _perfil(voz=[1.0, 0.0, 0.0], caras=[[0.0, 1.0]])})
    biometria.pon_motores(
        voz=MotorVozFalso(_con_coseno(0.47)),
        cara=MotorCaraFalso([_cara([0.0, 1.0])]),
    )

    assert biometria.identificar_cara(tmp_path, IMAGEN)["caras"][0]["nombre"] == "Javi"
    r = biometria.identificar_voz(tmp_path, _voz())
    assert r["nombre"] is None and r["dudoso"] == "Javi"


def test_dos_voces_parecidas_no_se_nombran(tmp_path: Path) -> None:
    """Un padre y un hijo: acertar a medias es equivocarse del todo."""
    padre = [math.cos(0.45), math.sin(0.45), 0.0]
    biometria.guardar(
        tmp_path,
        {"Persus": _perfil(voz=[1.0, 0.0, 0.0]), "Padre": _perfil(voz=padre)},
    )
    # Justo entre los dos: se parece a ambos lo mismo, y mucho.
    biometria.pon_motores(voz=MotorVozFalso([math.cos(0.225), math.sin(0.225), 0.0]))
    r = biometria.identificar_voz(tmp_path, _voz())
    assert r["nombre"] is None and r["dudoso"] in ("Persus", "Padre")


def test_dos_caras_parecidas_no_se_nombran(tmp_path: Path) -> None:
    biometria.guardar(
        tmp_path,
        {"Persus": _perfil(caras=[[1.0, 0.0]]), "Padre": _perfil(caras=[[0.98, 0.2]])},
    )
    biometria.pon_motores(cara=MotorCaraFalso([_cara([0.99, 0.1])]))
    cara = biometria.identificar_cara(tmp_path, IMAGEN)["caras"][0]
    assert cara["nombre"] is None and cara["dudoso"] in ("Persus", "Padre")


def test_una_muestra_que_tambien_se_parece_a_otro_no_entra(tmp_path: Path) -> None:
    """Lo que contamina un perfil es la muestra que vale para dos personas."""
    biometria.guardar(
        tmp_path,
        {
            "Persus": _perfil(voz=[1.0, 0.0, 0.0]) | {"voces": [[1.0, 0.0, 0.0]]},
            "Visita": _perfil(voz=[0.0, 1.0, 0.0]) | {"voces": [[0.0, 1.0, 0.0]]},
        },
    )
    # 0,80 a Persus y 0,60 a la visita: nombra a Persus, pero no aprende de ella.
    biometria.pon_motores(voz=MotorVozFalso([0.8, 0.6, 0.0]))
    assert biometria.identificar_voz(tmp_path, _voz())["nombre"] == "Persus"
    assert len(biometria.cargar(tmp_path)["Persus"]["voces"]) == 1


def test_la_voz_mejora_con_el_uso_y_queda_anotado(tmp_path: Path) -> None:
    """Cada turno claro deja una muestra nueva, hasta `MAX_VOCES`, y una línea en el registro."""
    biometria.guardar(tmp_path, {"Persus": _perfil(voz=[1.0, 0.0, 0.0]) | {"voces": [[1.0, 0.0, 0.0]]}})
    muestras = [[0.9, 0.3 * (k % 2), 0.3 * (1 - k % 2) + 0.05 * k] for k in range(6)]
    biometria.pon_motores(voz=MotorVozFalso(*muestras))
    for _ in muestras:
        assert biometria.identificar_voz(tmp_path, _voz())["nombre"] == "Persus"
    assert len(biometria.cargar(tmp_path)["Persus"]["voces"]) > 1
    registro = biometria.estado_completo(tmp_path)["registro"]
    assert any("muestra" in linea and "Persus" in linea for linea in registro)


def test_nombrar_deja_constancia_en_el_registro(tmp_path: Path) -> None:
    biometria.pon_motores(voz=MotorVozFalso([0.0, 1.0, 0.0]))
    biometria.identificar_voz(tmp_path, _voz())
    biometria.renombrar(tmp_path, "Desconocido 1", "Lucía")
    biometria.borrar(tmp_path, "Lucía")
    registro = (tmp_path / "personas.log").read_text(encoding="utf-8")
    assert "nombrado" in registro and "Lucía" in registro and "era «Desconocido 1»" in registro
    assert "borrado" in registro


def test_desconocido_no_vale_como_nombre(tmp_path: Path) -> None:
    biometria.pon_motores(voz=MotorVozFalso([0.0, 1.0, 0.0]))
    biometria.identificar_voz(tmp_path, _voz())
    assert "no es un nombre" in biometria.renombrar(tmp_path, "Desconocido 1", "Desconocido 7")["error"]
    assert "no es un nombre" in biometria.enrolar(tmp_path, "Desconocido 2", audio=_voz(3.0))["error"]


def test_con_dos_caras_delante_la_camara_no_decide(tmp_path: Path) -> None:
    biometria.guardar(
        tmp_path,
        {
            "Javi": _perfil(voz=[1.0, 0.0, 0.0], caras=[[0.0, 1.0]]),
            "Ana": _perfil(caras=[[1.0, 0.0]]),
        },
    )
    biometria.pon_motores(
        voz=MotorVozFalso(_con_coseno(0.47)),
        cara=MotorCaraFalso(
            [_cara([0.0, 1.0], [0, 0, 80, 80]), _cara([1.0, 0.0], [300, 0, 80, 80])]
        ),
    )
    biometria.identificar_cara(tmp_path, IMAGEN)
    assert biometria.identificar_voz(tmp_path, _voz())["nombre"] is None


def test_voz_y_cara_de_un_desconocido_comparten_etiqueta_y_ficha(tmp_path: Path) -> None:
    biometria.pon_motores(
        voz=MotorVozFalso([0.0, 1.0, 0.0]),
        cara=MotorCaraFalso([_cara([0.0, 1.0])]),
    )
    cara = biometria.identificar_cara(tmp_path, IMAGEN)["caras"][0]
    voz = biometria.identificar_voz(tmp_path, _voz())
    assert cara["nombre"] == voz["nombre"] == "Desconocido 1"

    assert biometria.renombrar(tmp_path, "Desconocido 1", "Lucía")["ok"] is True
    perfiles = biometria.cargar(tmp_path)
    assert list(perfiles) == ["Lucía"]
    assert perfiles["Lucía"]["voces"] and perfiles["Lucía"]["caras"]


def test_la_voz_de_un_desconocido_no_se_cuelga_de_alguien_con_nombre(tmp_path: Path) -> None:
    """Julio sale solo en cámara y habla otro fuera de plano: esa voz no es de Julio."""
    biometria.guardar(tmp_path, {"Julio": _perfil(caras=[[0.0, 1.0]])})
    biometria.pon_motores(
        voz=MotorVozFalso([0.0, 0.0, 1.0]),
        cara=MotorCaraFalso([_cara([0.0, 1.0])]),
    )
    biometria.identificar_cara(tmp_path, IMAGEN)
    assert biometria.identificar_voz(tmp_path, _voz())["nombre"] == "Desconocido 1"


def test_una_cara_desconocida_que_habla_hereda_la_etiqueta_de_su_voz(tmp_path: Path) -> None:
    biometria.pon_motores(
        voz=MotorVozFalso([0.0, 1.0, 0.0]),
        cara=MotorCaraFalso([_cara([0.0, 1.0])]),
    )
    assert biometria.identificar_voz(tmp_path, _voz())["nombre"] == "Desconocido 1"
    assert biometria.identificar_cara(tmp_path, IMAGEN)["caras"][0]["nombre"] == "Desconocido 1"


# --------------------------------------------------------------------------- #
# Caras: calidad y seguimiento
# --------------------------------------------------------------------------- #


def test_una_cara_movida_no_se_aprende(tmp_path: Path) -> None:
    biometria.pon_motores(cara=MotorCaraFalso([_cara([0.0, 1.0], nitidez=5.0)]))
    for _ in range(biometria.FRAMES_CARA_APRENDER + 2):
        cara = biometria.identificar_cara(tmp_path, IMAGEN)["caras"][0]
    assert cara["nombre"] is None
    assert "movida" in cara["calidad"]
    assert biometria.cargar(tmp_path) == {}


def test_una_cara_conocida_se_reconoce_aunque_salga_mal(tmp_path: Path) -> None:
    biometria.guardar(tmp_path, {"Ana": _perfil(caras=[[1.0, 0.0]])})
    biometria.pon_motores(cara=MotorCaraFalso([_cara([1.0, 0.0], nitidez=5.0, giro=0.9)]))
    cara = biometria.identificar_cara(tmp_path, IMAGEN)["caras"][0]
    assert cara["nombre"] == "Ana"
    # Pero no entra en su galería: reconocer sí, enseñar no.
    assert biometria.cargar(tmp_path)["Ana"]["caras"] == [[1.0, 0.0]]


def test_un_mal_fotograma_de_alguien_conocido_no_cria_un_desconocido(tmp_path: Path) -> None:
    biometria.guardar(tmp_path, {"Ana": _perfil(caras=[[1.0, 0.0]])})
    buena, rara = _cara([1.0, 0.0]), _cara([0.0, 1.0], [104, 98, 80, 80])
    biometria.pon_motores(cara=MotorCaraFalso([buena], [buena], [rara]))

    biometria.identificar_cara(tmp_path, IMAGEN)
    biometria.identificar_cara(tmp_path, IMAGEN)
    tercera = biometria.identificar_cara(tmp_path, IMAGEN)["caras"][0]
    assert tercera["nombre"] == "Ana"
    assert biometria.estado_completo(tmp_path)["aprendiendo"]["cara"] is None


def test_la_calidad_se_explica_para_repetir_la_toma() -> None:
    assert "acércate" in biometria.calidad_cara({"caja": [0, 0, 20, 20]})
    assert "de lado" in biometria.calidad_cara({"caja": [0, 0, 90, 90], "giro": 0.9})
    assert biometria.calidad_cara({"caja": [0, 0, 90, 90], "nitidez": 80.0, "giro": 0.1}) is None


# --------------------------------------------------------------------------- #
# El alta desde Ajustes
# --------------------------------------------------------------------------- #


def test_una_toma_con_poca_voz_se_rechaza_diciendo_cuanta_habia(tmp_path: Path) -> None:
    biometria.pon_motores(voz=MotorVozFalso([1.0, 0.0]))
    r = biometria.enrolar(tmp_path, "Javi", audio=_voz(1.2))
    assert "poca voz" in r["error"]
    assert r["segundos_voz"] < biometria.SEGUNDOS_ENROLAR
    assert biometria.cargar(tmp_path) == {}


def test_varias_tomas_suman_a_la_galeria_y_dicen_cuanto_se_parecen(tmp_path: Path) -> None:
    biometria.pon_motores(voz=MotorVozFalso([1.0, 0.0, 0.0], _con_coseno(0.8)))
    primera = biometria.enrolar(tmp_path, "Javi", audio=_voz(3.0))
    segunda = biometria.enrolar(tmp_path, "Javi", audio=_voz(3.0))
    assert "parecido_voz" not in primera
    assert segunda["parecido_voz"] == pytest.approx(0.8)
    assert segunda["voces"] == 2


def test_una_foto_de_lado_no_sirve_de_alta(tmp_path: Path) -> None:
    biometria.pon_motores(cara=MotorCaraFalso([_cara([1.0, 0.0], giro=0.9)]))
    r = biometria.enrolar(tmp_path, "Javi", imagen=IMAGEN)
    assert "de lado" in r["error"]


def test_en_la_foto_de_alta_manda_la_cara_mas_grande(tmp_path: Path) -> None:
    biometria.pon_motores(
        cara=MotorCaraFalso(
            [_cara([0.0, 1.0], [0, 0, 50, 50]), _cara([1.0, 0.0], [200, 0, 160, 160])]
        )
    )
    assert biometria.enrolar(tmp_path, "Javi", imagen=IMAGEN)["ok"] is True
    assert biometria.cargar(tmp_path)["Javi"]["caras"] == [[1.0, 0.0]]
