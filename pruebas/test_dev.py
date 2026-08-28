"""El agente `dev`: de la raíz no se sale, y las listas son las que son."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from dataclasses import replace

from perseo_core import almacen, dev

#: Ruta absoluta fuera de la raíz permitida, en cualquiera de los dos sistemas
#: donde corren las pruebas. Ver la nota de `pruebas/test_memoria.py`.
FUERA_DEL_DISCO = r"C:\Windows" if os.name == "nt" else "/etc"


@pytest.fixture()
def dev_falso(cfg: almacen.Configuracion, tmp_path: Path, monkeypatch):
    """Motor de mentira y una raíz de usar y tirar."""
    raiz = tmp_path / "proyecto"
    (raiz / "dentro").mkdir(parents=True)
    monkeypatch.setenv("PERSEO_DEV_MOTOR", "falso")
    monkeypatch.setenv("PERSEO_DEV_RAIZ", str(raiz))
    nueva = almacen.cargar_configuracion()

    monkeypatch.setattr(dev, "_motor", None)
    dev.iniciar(nueva)
    yield raiz
    dev.detener()


def test_con_motor_falso_hay_motor(dev_falso) -> None:
    assert isinstance(dev._motor, dev.MotorFalso)


def test_sin_directorio_el_encargo_va_a_la_raiz(dev_falso: Path) -> None:
    assert dev.resolver_raiz("") == dev_falso.resolve()


def test_un_subdirectorio_vale(dev_falso: Path) -> None:
    assert dev.resolver_raiz("dentro") == (dev_falso / "dentro").resolve()


@pytest.mark.parametrize(
    "intento",
    [
        FUERA_DEL_DISCO,
        # Fuera del perfil del usuario, que desde el 2026-08-24 es el cerco.
        str(Path.home().resolve().parent),
    ],
)
def test_del_perfil_no_se_sale(dev_falso, intento: str) -> None:
    """Un encargo puede venir de un correo: lo de fuera, fuera."""
    with pytest.raises(ValueError):
        dev.resolver_raiz(intento)


def test_subir_hacia_el_perfil_ahora_vale(dev_falso: Path) -> None:
    """El cerco es el perfil entero, no la raíz: «..» cae dentro y se permite."""
    destino = dev.resolver_raiz("..")
    assert Path.home() in destino.parents or destino == Path.home()


def test_un_directorio_que_no_existe_se_rechaza(dev_falso) -> None:
    with pytest.raises(ValueError):
        dev.resolver_raiz("no_existe")


def test_git_push_esta_denegado() -> None:
    """Publicar es del usuario, no del agente."""
    assert any("git push" in h for h in dev.HERRAMIENTAS_DENEGADAS)


def test_borrar_esta_denegado() -> None:
    assert any(h.startswith("Bash(rm ") for h in dev.HERRAMIENTAS_DENEGADAS)
    assert any("git reset --hard" in h for h in dev.HERRAMIENTAS_DENEGADAS)


def test_no_hay_un_bash_abierto_entre_las_permitidas() -> None:
    """Un `Bash` a secas haría inútiles las denegadas."""
    assert "Bash" not in dev.HERRAMIENTAS_PERMITIDAS
    assert all(h.startswith("Bash(") or "(" not in h for h in dev.HERRAMIENTAS_PERMITIDAS)


def test_hay_tope_de_vueltas() -> None:
    assert 0 < dev.MAX_VUELTAS <= 100


def test_el_encargo_pasa_por_el_motor(dev_falso) -> None:
    resultado = asyncio.run(dev._dev({"peticion": {"texto": "arregla el bug"}}))
    assert "arregla el bug" in resultado["texto"]
    assert resultado["titular"]
    assert resultado["sesion"] == "falsa"
    assert dev._motor.encargos == ["arregla el bug"]


def test_el_titular_concuerda_en_singular(dev_falso) -> None:
    resultado = asyncio.run(dev._dev({"peticion": {"texto": "algo"}}))
    assert "1 vuelta)" in resultado["titular"]


def test_un_encargo_sin_texto_se_rechaza(dev_falso) -> None:
    with pytest.raises(ValueError):
        asyncio.run(dev._dev({"peticion": {"texto": "   "}}))


def test_un_encargo_fuera_del_perfil_se_rechaza(dev_falso) -> None:
    with pytest.raises(ValueError):
        asyncio.run(dev._dev({"peticion": {"texto": "algo", "directorio": FUERA_DEL_DISCO}}))


def test_sin_motor_el_encargo_falla_con_un_error_util(dev_falso, monkeypatch) -> None:
    monkeypatch.setattr(dev, "_motor", None)
    with pytest.raises(RuntimeError, match="PERSEO_DEV_MOTOR"):
        asyncio.run(dev._dev({"peticion": {"texto": "algo"}}))


def test_un_resultado_fallido_del_motor_falla_el_trabajo(dev_falso, monkeypatch) -> None:
    class MotorQueFalla:
        async def ejecutar(self, encargo, avisar=None):
            return dev.Resultado(texto="no pude", ok=False)

    monkeypatch.setattr(dev, "_motor", MotorQueFalla())
    with pytest.raises(RuntimeError, match="no pude"):
        asyncio.run(dev._dev({"peticion": {"texto": "algo"}}))


# ── La elección de motor por encargo (2026-08-24) ─────────────────────────── #

def test_la_eleccion_por_encargo_manda(dev_falso, monkeypatch) -> None:
    """Aunque el motor configurado sea otro, `peticion.motor` decide."""
    lanzados = []

    class MotorQueAnota:
        async def ejecutar(self, encargo, avisar=None):
            lanzados.append(encargo)
            return dev.Resultado(texto="hecho", vueltas=1)

    monkeypatch.setattr(dev, "_motor_de", lambda nombre: MotorQueAnota())
    asyncio.run(dev._dev({"peticion": {"texto": "algo", "motor": "opencode"}}))
    assert len(lanzados) == 1


def test_pedir_un_motor_desconocido_es_error_claro(dev_falso) -> None:
    with pytest.raises(ValueError, match="motor"):
        asyncio.run(dev._dev({"peticion": {"texto": "algo", "motor": "gemini"}}))


def test_opencode_no_instalado_da_error_util(dev_falso, monkeypatch) -> None:
    monkeypatch.setattr(dev.shutil, "which", lambda _: None)
    with pytest.raises(ValueError, match="opencode"):
        asyncio.run(dev._dev({"peticion": {"texto": "algo", "motor": "opencode"}}))


def test_abrir_motor_opencode(monkeypatch) -> None:
    monkeypatch.setenv("PERSEO_DEV_MOTOR", "opencode")
    monkeypatch.setattr(dev.shutil, "which", lambda n: "C:/falso/opencode.exe" if n == "opencode" else None)
    motor = dev.abrir_motor(almacen.cargar_configuracion())
    assert isinstance(motor, dev.MotorOpencode)


# ── El encargo en lenguaje natural (2026-08-24) ───────────────────────────── #

def test_el_motor_sale_del_texto() -> None:
    assert dev._motor_del_texto("En Armario, arregla el bug con opencode") == "opencode"
    assert dev._motor_del_texto("hazlo usando Claude") == "claude"
    assert dev._motor_del_texto("hazlo") == ""


def test_el_proyecto_sale_del_texto(tmp_path: Path) -> None:
    (tmp_path / "proyectos.json").write_text(
        json.dumps([
            {
                "id": "armario-app",
                "nombre": "Armario · App",
                "modo": "servicio",
                "destino": "http://127.0.0.1:8000",
                "servidores": [{"arranque": ["python", "x.py"], "carpeta": r"C:\Users\uno\armario"}],
            }
        ], ensure_ascii=False),
        encoding="utf-8",
    )
    # El modo servicio trabaja en la carpeta del servidor, no en su URL.
    assert dev._proyecto_del_texto("En Armario, añade un README", tmp_path) == r"C:\Users\uno\armario"
    # Sin proyecto nombrado, vacío: manda la raíz.
    assert dev._proyecto_del_texto("arregla el bug", tmp_path) == ""


def test_el_proyecto_no_salta_en_medio_de_una_palabra(tmp_path: Path) -> None:
    (tmp_path / "proyectos.json").write_text(
        json.dumps([{"id": "perseo", "nombre": "Perseo", "modo": "carpeta", "destino": r"C:\Users\uno\Perseo"}]),
        encoding="utf-8",
    )
    assert dev._proyecto_del_texto("hay que perseverar con esto", tmp_path) == ""


def test_el_perfil_del_usuario_esta_dentro_del_cerco(dev_falso) -> None:
    """«Que tenga permiso para trabajar en todo usuario» (2026-08-24)."""
    assert Path.home().resolve() in dev._raices


def test_un_encargo_nominando_proyecto_no_encierra_al_agente(
    dev_falso: Path, tmp_path: Path, monkeypatch
) -> None:
    """«En X…» se le CUENTA al agente, no se le usa de jaula (2026-08-26).

    Los proyectos del señor Persus se llaman unos a otros; un agente encerrado
    en la carpeta de Armario no puede leer Perseo. Se trabaja desde la raíz y
    la ruta del proyecto va en el contexto.
    """
    carpeta = tmp_path / "proyecto_real"
    carpeta.mkdir()
    (tmp_path / "proyectos.json").write_text(
        json.dumps([
            {"id": "armario-app", "nombre": "Armario · App", "modo": "carpeta", "destino": str(carpeta)}
        ], ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(dev, "_datos", tmp_path)
    resultado = asyncio.run(dev._dev({"peticion": {"texto": "En Armario, arregla el bug"}}))
    assert resultado["directorio"] == str(dev_falso.resolve())
    assert str(carpeta) in dev._contexto_del_encargo("En Armario, arregla el bug", dev_falso)


# ── Lo que se rompió el 2026-08-25, fuera de casa ─────────────────────────── #


def test_una_url_como_directorio_no_tumba_el_encargo(dev_falso: Path) -> None:
    """La lista del móvil mandaba `destino`, que en modo servicio es una URL.

    El encargo moría con «'http://127.0.0.1:8000' no es un directorio» antes de
    arrancar nada. Ahora se ignora y se trabaja desde la raíz.
    """
    resultado = asyncio.run(
        dev._dev({"peticion": {"texto": "algo", "directorio": "http://127.0.0.1:8000"}})
    )
    assert resultado["directorio"] == str(dev_falso.resolve())


def test_un_proyecto_que_solo_tiene_url_no_da_carpeta(tmp_path: Path) -> None:
    (tmp_path / "proyectos.json").write_text(
        json.dumps([
            {"id": "web", "nombre": "Web", "modo": "carpeta", "destino": "https://persus.netlify.app"}
        ], ensure_ascii=False),
        encoding="utf-8",
    )
    assert dev._proyecto_del_texto("mira la Web", tmp_path) == ""


def test_lo_que_pide_el_señor_persus_lleva_las_manos_anchas(dev_falso, monkeypatch) -> None:
    """«Abre la app de armario» necesita poder arrancar un proceso."""
    vistos = []

    class MotorQueAnota:
        async def ejecutar(self, encargo, avisar=None):
            vistos.append(encargo)
            return dev.Resultado(texto="hecho", vueltas=1)

    monkeypatch.setattr(dev, "_motor", MotorQueAnota())
    asyncio.run(dev._dev({"origen": "texto", "peticion": {"texto": "abre la app"}}))
    assert "Bash" in vistos[0].permitidas
    assert "Task" in vistos[0].permitidas


def test_lo_que_nace_de_un_correo_sigue_con_las_manos_cortas(dev_falso, monkeypatch) -> None:
    """Un disparador no es el señor Persus: la lista corta es la de siempre."""
    vistos = []

    class MotorQueAnota:
        async def ejecutar(self, encargo, avisar=None):
            vistos.append(encargo)
            return dev.Resultado(texto="hecho", vueltas=1)

    monkeypatch.setattr(dev, "_motor", MotorQueAnota())
    asyncio.run(dev._dev({"origen": "disparador", "peticion": {"texto": "haz algo"}}))
    assert vistos[0].permitidas == dev.HERRAMIENTAS_PERMITIDAS
    assert "Bash" not in vistos[0].permitidas


def test_el_modelo_del_encargo_llega_al_motor(dev_falso, monkeypatch) -> None:
    vistos = []

    class MotorQueAnota:
        async def ejecutar(self, encargo, avisar=None):
            vistos.append(encargo)
            return dev.Resultado(texto="hecho", vueltas=1)

    monkeypatch.setattr(dev, "_motor", MotorQueAnota())
    asyncio.run(dev._dev({"peticion": {"texto": "algo", "modelo": "opus"}}))
    assert vistos[0].modelo == "opus"


def test_el_agente_ve_las_demas_carpetas_del_perfil(dev_falso, monkeypatch) -> None:
    """Trabaja desde una, pero puede leer las otras: están interconectadas."""
    vistos = []

    class MotorQueAnota:
        async def ejecutar(self, encargo, avisar=None):
            vistos.append(encargo)
            return dev.Resultado(texto="hecho", vueltas=1)

    monkeypatch.setattr(dev, "_motor", MotorQueAnota())
    asyncio.run(dev._dev({"peticion": {"texto": "algo"}}))
    assert Path.home().resolve() in vistos[0].carpetas_extra


def test_el_contexto_le_prohibe_dar_por_hecho_lo_que_no_hizo(dev_falso: Path) -> None:
    """El fallo de fondo del 2026-08-25: HECHO sin haber hecho nada."""
    contexto = dev._contexto_del_encargo("abre la app", dev_falso)
    assert "No lo des por hecho" in contexto
    assert str(dev_falso) in contexto


# ── La bitácora: qué hizo, paso a paso ────────────────────────────────────── #


def test_la_bitacora_guarda_el_paso_a_paso(dev_falso, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(dev, "_datos", tmp_path)

    class MotorQueCuenta:
        async def ejecutar(self, encargo, avisar=None):
            avisar(dev.Paso(tipo="herramienta", titulo="Leyendo api.py"))
            avisar(dev.Paso(tipo="subagente", titulo="explorador: mira esto", agente="tu_1"))
            avisar(dev.Paso(tipo="herramienta", titulo="Buscando def", agente="tu_1"))
            avisar(dev.Paso(tipo="resultado", titulo="Error", agente="tu_1", ok=False))
            return dev.Resultado(texto="hecho", vueltas=3)

    monkeypatch.setattr(dev, "_motor", MotorQueCuenta())
    asyncio.run(dev._dev({"id": 41, "peticion": {"texto": "algo"}}))

    actividad = dev.actividad_de(41)
    assert [p["titulo"] for p in actividad["pasos"]][:2] == ["Leyendo api.py", "explorador: mira esto"]
    porNombre = {a["id"]: a for a in actividad["agentes"]}
    assert porNombre["principal"]["pasos"] == 1
    assert porNombre["tu_1"]["titulo"] == "explorador: mira esto"
    assert porNombre["tu_1"]["fallos"] == 1
    # Y sobrevive al encargo: la pregunta «¿qué hizo?» se hace después.
    assert not actividad["vivo"]


def test_la_bitacora_se_relee_del_disco(dev_falso, monkeypatch, tmp_path: Path) -> None:
    """El núcleo se reinicia; la pregunta de la mañana siguiente sigue en pie."""
    monkeypatch.setattr(dev, "_datos", tmp_path)
    dev._anotar(77, dev.Paso(tipo="herramienta", titulo="Editando api.py"))
    dev._bitacoras.pop(77, None)
    assert [p["titulo"] for p in dev.actividad_de(77)["pasos"]] == ["Editando api.py"]


def test_un_fallo_del_motor_queda_apuntado(dev_falso, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(dev, "_datos", tmp_path)

    class MotorQueRevienta:
        async def ejecutar(self, encargo, avisar=None):
            raise RuntimeError("se cayó el proveedor")

    monkeypatch.setattr(dev, "_motor", MotorQueRevienta())
    with pytest.raises(RuntimeError):
        asyncio.run(dev._dev({"id": 42, "peticion": {"texto": "algo"}}))
    pasos = dev.actividad_de(42)["pasos"]
    assert pasos[-1]["tipo"] == "error" and not pasos[-1]["ok"]


# --------------------------------------------------------------------------- #
# El motor opencode: los permisos y el fracaso que sale con código 0
# --------------------------------------------------------------------------- #


def _proceso_falso(lineas: list[bytes], codigo: int = 0, error: bytes = b""):
    """Un `opencode run --format json` de mentira, con la forma que lee el motor.

    Desde que MotorOpencode lee los eventos SEGÚN SALEN (para poder contar por
    dónde va el encargo), lo que hace falta simular no es un `communicate()`
    sino un `stdout` que se recorre línea a línea y un `stderr` que se vacía en
    paralelo. Los dobles viejos se quedaron sin `stdout` y reventaban todos a la
    vez, que es lo que pasa cuando un doble copia una firma en vez de un
    comportamiento.
    """

    class SalidaFalsa:
        def __aiter__(self):
            async def generar():
                for linea in lineas:
                    yield linea

            return generar()

    class ErrorFalso:
        async def read(self):
            return error

    class ProcesoFalso:
        returncode = codigo
        stdout = SalidaFalsa()
        stderr = ErrorFalso()

        async def wait(self):
            return codigo

    return ProcesoFalso()


def _evento_de_texto(texto: str) -> bytes:
    """Un evento `text` de opencode, que es como el agente dice las cosas."""
    return (json.dumps({"type": "text", "part": {"text": texto}}) + "\n").encode("utf-8")


def _argumentos_de_opencode(monkeypatch, **entorno) -> list[str]:
    """Lo que MotorOpencode le pide de verdad al sistema operativo."""
    vistos: dict[str, list[str]] = {}

    async def falso_exec(*argumentos, **kwargs):
        vistos["argumentos"] = list(argumentos)
        return _proceso_falso([_evento_de_texto("listo")])

    monkeypatch.delenv("PERSEO_DEV_MODELO", raising=False)
    for clave, valor in entorno.items():
        monkeypatch.setenv(clave, valor)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", falso_exec)
    motor = dev.MotorOpencode("opencode")
    asyncio.run(motor.ejecutar(dev.Encargo("haz algo", Path.cwd(), 30.0)))
    return vistos["argumentos"]


def test_opencode_va_con_auto(monkeypatch) -> None:
    """Sin `--auto` se deniega a sí mismo escribir y sale con éxito igual."""
    assert "--auto" in _argumentos_de_opencode(monkeypatch)


def test_opencode_lleva_la_carpeta_escrita(monkeypatch) -> None:
    """`opencode run` hereda el `cwd` y luego lo IGNORA: hay que decirle la raíz."""
    argumentos = _argumentos_de_opencode(monkeypatch)
    assert argumentos[argumentos.index("--dir") + 1] == str(Path.cwd())


def test_opencode_lleva_el_modelo_pedido(monkeypatch) -> None:
    argumentos = _argumentos_de_opencode(monkeypatch, PERSEO_DEV_MODELO="opencode/glm-5")
    assert argumentos[argumentos.index("-m") + 1] == "opencode/glm-5"


def test_opencode_nunca_sale_sin_modelo(monkeypatch) -> None:
    """Sin `-m`, opencode usa el que tenga configurado — y ese puede ser DE PAGO."""
    argumentos = _argumentos_de_opencode(monkeypatch)
    elegido = argumentos[argumentos.index("-m") + 1]
    assert elegido == dev.MODELO_OPENCODE_POR_DEFECTO
    assert elegido in dev.MODELOS_GRATIS_OPENCODE


def test_el_modelo_dictado_a_medias_se_completa(monkeypatch) -> None:
    """Perseo oye «el hy3» y lo manda sin proveedor: la barra se le pone aquí."""
    monkeypatch.delenv("PERSEO_DEV_MODELO", raising=False)
    assert dev.modelo_opencode("hy3-free") == "opencode/hy3-free"


@pytest.mark.parametrize(
    "salida",
    [
        "! permission requested: external_directory; auto-rejecting",
        "Error from provider (Console): Upstream request failed: Endpoint is unavailable.",
    ],
)
def test_un_exito_que_no_hizo_nada_cuenta_como_fallo(monkeypatch, salida: str) -> None:
    """Código 0 no basta: lo que dijo el CLI también cuenta (2026-08-24)."""

    async def falso_exec(*argumentos, **kwargs):
        return _proceso_falso([_evento_de_texto(salida)])

    monkeypatch.setattr(asyncio, "create_subprocess_exec", falso_exec)
    resultado = asyncio.run(dev.MotorOpencode("opencode").ejecutar(dev.Encargo("x", Path.cwd(), 30.0)))
    assert not resultado.ok


def test_una_salida_normal_sigue_siendo_exito(monkeypatch) -> None:
    """El detector no puede volverse un muro: lo bueno pasa."""

    async def falso_exec(*argumentos, **kwargs):
        return _proceso_falso([_evento_de_texto("Fichero creado con el texto ok.")])

    monkeypatch.setattr(asyncio, "create_subprocess_exec", falso_exec)
    resultado = asyncio.run(dev.MotorOpencode("opencode").ejecutar(dev.Encargo("x", Path.cwd(), 30.0)))
    assert resultado.ok


def test_opencode_apunta_en_la_bitacora_lo_que_va_haciendo(monkeypatch) -> None:
    """El motivo de `--format json`: con la salida bonita no se sabía nada
    hasta el final, y la pestaña de actividad se quedaba en blanco justo con el
    motor que el señor Persus quiere usar a diario."""
    evento = json.dumps(
        {
            "type": "tool_use",
            "part": {"tool": "write", "state": {"input": {"filePath": "hola.txt"}}},
        }
    )
    lineas = [(evento + "\n").encode("utf-8"), _evento_de_texto("HECHO")]

    async def falso_exec(*argumentos, **kwargs):
        return _proceso_falso(lineas)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", falso_exec)
    pasos: list[dev.Paso] = []
    asyncio.run(
        dev.MotorOpencode("opencode").ejecutar(
            dev.Encargo("x", Path.cwd(), 30.0), avisar=pasos.append
        )
    )
    assert any(p.tipo == "herramienta" for p in pasos)
    assert any(p.tipo == "dice" for p in pasos)


# --------------------------------------------------------------------------- #
# El motor sobre el Agent SDK oficial
# --------------------------------------------------------------------------- #


def test_opencode_manda_sin_motor_pedido(cfg, monkeypatch) -> None:
    """Decidido el 2026-08-26: sin `PERSEO_DEV_MOTOR` gana opencode, aunque el
    SDK esté instalado. No gasta suscripción y sabe contar por dónde va igual."""
    monkeypatch.setattr(dev, "hay_sdk", lambda: True)
    monkeypatch.setattr(
        dev.shutil, "which", lambda nombre: "/bin/opencode" if nombre == "opencode" else None
    )
    assert isinstance(dev.abrir_motor(replace(cfg, dev_motor="")), dev.MotorOpencode)


def test_sin_opencode_manda_el_sdk(cfg, monkeypatch) -> None:
    """El segundo de la fila: cuenta el progreso, que es lo que la consola no
    sabe hacer."""
    monkeypatch.setattr(dev, "hay_sdk", lambda: True)
    monkeypatch.setattr(dev, "MotorSdk", lambda: "el-sdk")
    monkeypatch.setattr(dev.shutil, "which", lambda nombre: None)
    assert dev.abrir_motor(replace(cfg, dev_motor="")) == "el-sdk"


def test_sin_opencode_ni_sdk_se_sigue_por_consola(cfg, monkeypatch) -> None:
    """Los dos paquetes son opcionales de verdad: sin ellos, `dev` no se queda
    sin motor."""
    monkeypatch.setattr(dev, "hay_sdk", lambda: False)
    monkeypatch.setattr(
        dev.shutil, "which", lambda nombre: None if nombre == "opencode" else "/bin/claude"
    )
    assert isinstance(dev.abrir_motor(replace(cfg, dev_motor="")), dev.MotorClaude)


def test_pedir_el_sdk_sin_tenerlo_no_deja_sin_motor(cfg, monkeypatch) -> None:
    monkeypatch.setattr(dev, "hay_sdk", lambda: False)
    monkeypatch.setattr(
        dev.shutil, "which", lambda nombre: None if nombre == "opencode" else "/bin/claude"
    )
    assert isinstance(dev.abrir_motor(replace(cfg, dev_motor="sdk")), dev.MotorClaude)


@pytest.mark.parametrize(
    "herramienta, entrada, espera",
    [
        ("Edit", {"file_path": "C:\\Users\\x\\api.py"}, "Editando api.py"),
        ("Bash", {"command": "python -m pytest"}, "Ejecutando python -m pytest"),
        ("Grep", {"pattern": "def iniciar"}, "Buscando def iniciar"),
        ("LoQueSea", {}, "LoQueSea"),
    ],
)
def test_el_progreso_se_cuenta_en_cristiano(herramienta, entrada, espera) -> None:
    """Quien mira quiere saber si avanza, no ver el JSON de la llamada."""
    assert dev._contar_herramienta(herramienta, entrada) == espera


def test_el_progreso_de_un_encargo_vivo_se_puede_consultar(dev_falso) -> None:
    dev._progreso[7] = "Editando api.py"
    try:
        assert dev.progreso_de(7) == "Editando api.py"
        assert dev.progreso_de(8) == ""
    finally:
        dev._progreso.clear()


def test_al_acabar_el_encargo_no_queda_progreso(dev_falso) -> None:
    """Si no se limpiara, el panel enseñaría para siempre el último paso."""
    asyncio.run(dev._dev({"id": 12, "peticion": {"texto": "algo"}}))
    assert dev.progreso_de(12) == ""
