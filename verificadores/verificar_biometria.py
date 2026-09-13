"""Verificación del reconocimiento de personas: las rutas `/biometria`.

Aquí no se mide la calidad de ECAPA ni de SFace —esa la dan sus papers—, se
mide que **la tubería completa** haga lo prometido contra el núcleo real:

- sin token, 401; con token, estado claro de perfiles y motores,
- una voz desconocida se aprende sola al acumular trozos y queda como perfil,
- reconocida después por su etiqueta, renombrable y borrable,
- el borrado quita los vectores de verdad,
- y todo lo que cree en el disco vive en el directorio temporal del arnés:
  los perfiles reales de `<datos>/perfiles.json` no se tocan.

La «voz» que viaja es un tono sintético: para esta verificación basta con que
sea PCM estable, que es justo lo que ECAPA convierte en vector. Ejecutar desde
la raíz:

    python verificadores/verificar_biometria.py
"""

from __future__ import annotations

import base64
import json
import math
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verificadores.arnes_pruebas import Nucleo, comprobar, resumir  # noqa: E402

NOMBRE = "Verificador"

#: Abre las peticiones SIN el proxy del sistema y con plazo holgado. Dos motivos
#: medidos el 2026-08-24: el primer trozo de voz carga ECAPA dentro del núcleo
#: (~8 s) y el `timeout=10` del arnés se queda corto; y un proxy del sistema
#: intercepta peticiones lentas a 127.0.0.1 y contesta su propio «500», que no
#: es culpa del núcleo ni huele igual.
ABRIDOR = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def pedir_biometria(
    nucleo: Nucleo,
    ruta: str,
    metodo: str = "GET",
    cuerpo: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    """Igual que `Nucleo.pedir`, pero sin proxy y con plazo de carga."""
    datos = json.dumps(cuerpo).encode() if cuerpo is not None else None
    peticion = urllib.request.Request(
        nucleo.base + ruta, data=datos, method=metodo
    )
    if datos:
        peticion.add_header("Content-Type", "application/json")
    peticion.add_header("Authorization", f"Bearer {nucleo.token}")
    try:
        with ABRIDOR.open(peticion, timeout=90) as respuesta:
            return respuesta.status, json.loads(respuesta.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        cuerpo_error = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(cuerpo_error or "{}")
        except json.JSONDecodeError:
            # Un 500 que no habla JSON merece verse entero, no tragarse.
            return e.code, {"error": cuerpo_error.strip()[:200]}


def tono(frecuencia: float, segundos: float = 2.5) -> str:
    """PCM int16 mono 16 kHz en base64: lo mismo que manda la llamada."""
    muestras = int(16000 * segundos)
    valores = [int(8000 * math.sin(i * frecuencia)) for i in range(muestras)]
    return base64.b64encode(struct.pack(f"<{muestras}h", *valores)).decode()


def comprobar_autenticacion() -> None:
    nucleo = Nucleo()
    nucleo.arrancar()
    try:
        # Sin cabecera: esta va por el arnés, que no añade Authorization.
        codigo, _ = nucleo.pedir("/biometria")
        comprobar("Sin token, /biometria devuelve 401", codigo == 401, f"HTTP {codigo}")

        codigo, estado = pedir_biometria(nucleo, "/biometria")
        comprobar("Con token, /biometria responde", codigo == 200, f"HTTP {codigo}")
        comprobar(
            "El estado trae perfiles y disponibilidad",
            "perfiles" in estado and "disponibilidad" in estado,
            str(sorted(estado)),
        )
    finally:
        nucleo.parar()


def comprobar_aprendizaje() -> None:
    nucleo = Nucleo()
    nucleo.arrancar()
    try:
        try:
            _aprendizaje(nucleo)
        except Exception:
            print("===== VOLCADO DEL HIJO (por el fallo de arriba) =====")
            nucleo.volcar()
            raise
    finally:
        nucleo.parar()


def _aprendizaje(nucleo: Nucleo) -> None:

    # Un desconocido habla: primero etiqueta provisional, luego perfil.
    ultimo: dict = {}
    aprendido_en = 99
    for vuelta in range(10):
        codigo, r = pedir_biometria(nucleo, "/biometria/voz", "POST", {"audio": tono(0.05)})
        if codigo != 200 or r.get("error"):
            comprobar("POST /biometria/voz funciona", False, f"{codigo} {r}")
            print("===== VOLCADO DEL HIJO =====")
            nucleo.volcar()
            return
        ultimo = r
        if r.get("aprendido"):
            aprendido_en = vuelta + 1
            break

    objetivo_segundos = 12.0 / 2.5
    comprobar(
        f"La voz se aprende sola ({aprendido_en} trozos)",
        ultimo.get("nombre") == "Desconocido 1" and ultimo.get("aprendido") is True,
        f"objetivo ~{objetivo_segundos:.0f} trozos",
    )

    # Y ya está reconocida, sin aprender nada otra vez.
    _, r = pedir_biometria(nucleo, "/biometria/voz", "POST", {"audio": tono(0.05)})
    comprobar(
        "Reconoce al aprendido por su etiqueta",
        r.get("nombre") == "Desconocido 1" and "aprendiendo" not in r,
        str(r),
    )

    # Le pone nombre real.
    codigo, r = pedir_biometria(
        nucleo, "/biometria/perfiles/Desconocido%201", "POST", {"nuevo_nombre": NOMBRE}
    )
    comprobar("Renombrar funciona", codigo == 200 and r.get("ok"), f"{codigo} {r}")

    _, r = pedir_biometria(nucleo, "/biometria/voz", "POST", {"audio": tono(0.05)})
    comprobar(
        "Lo reconoce por su nombre nuevo",
        r.get("nombre") == NOMBRE and r.get("confianza", 0) >= 0.55,
        str(r),
    )

    # La lista lo enseña, con tamaños pero sin vectores.
    _, estado = pedir_biometria(nucleo, "/biometria")
    nombres = [p["nombre"] for p in estado["perfiles"]]
    ficha = next((p for p in estado["perfiles"] if p["nombre"] == NOMBRE), {})
    comprobar("Aparece en la lista de perfiles", NOMBRE in nombres, str(nombres))
    comprobar(
        "La ficha no filtra vectores",
        all(not isinstance(v, list) for v in ficha.values()),
        str(ficha),
    )

    # Borrar es borrar: después ya no está y no lo reconoce a él.
    codigo, r = pedir_biometria(nucleo, f"/biometria/perfiles/{NOMBRE}", "DELETE")
    comprobar("Borrar funciona", codigo == 200 and r.get("ok"), f"{codigo} {r}")

    _, r = pedir_biometria(nucleo, "/biometria/voz", "POST", {"audio": tono(0.05)})
    comprobar(
        "Tras borrar vuelve a ser un desconocido nuevo",
        r.get("nombre") == "Desconocido 2"
        or (r.get("aprendiendo") or {}).get("etiqueta") == "Desconocido 2",
        str(r),
    )


def comprobar_gestion() -> None:
    """Errores claros: nombres ocupados, perfiles inexistentes, cuerpos vacíos."""
    nucleo = Nucleo()
    nucleo.arrancar()
    try:
        codigo, r = pedir_biometria(nucleo, "/biometria/voz", "POST", {})
        comprobar("Audio ausente es 400 con motivo", codigo == 400, f"{codigo} {r}")

        # Alta manual desde Ajustes: el camino corto del señor Persus. Regresión
        # del 2026-08-24: publicar el evento reventaba («got multiple values for
        # argument 'tipo'») y la ruta contestaba 500 con el perfil YA guardado.
        codigo, r = pedir_biometria(
            nucleo, "/biometria/perfiles", "POST", {"nombre": NOMBRE, "audio": tono(0.05)}
        )
        comprobar(
            "Enrolar con voz devuelve ok",
            codigo == 200 and r.get("ok") and "voz" in r.get("añadido", []),
            f"{codigo} {r}",
        )
        _, estado = pedir_biometria(nucleo, "/biometria")
        nombres = [p["nombre"] for p in estado["perfiles"]]
        comprobar("El enrolado aparece en la lista", NOMBRE in nombres, str(nombres))

        _, r = pedir_biometria(nucleo, "/biometria/perfiles/Nadie", "DELETE")
        comprobar(
            "Borrar un perfil inexistente lo dice",
            r.get("error", "").startswith("No hay"),
            str(r),
        )
        _, r = pedir_biometria(nucleo, "/biometria/perfiles/Nadie", "POST", {"nuevo_nombre": "X"})
        comprobar(
            "Renombrar un perfil inexistente lo dice",
            r.get("error", "").startswith("No hay"),
            str(r),
        )
    finally:
        nucleo.parar()


if __name__ == "__main__":
    print("== Autenticación ==")
    t0 = time.time()
    comprobar_autenticacion()
    resumir()

    print("\n== Aprendizaje de un desconocido ==")
    t0 = time.time()
    comprobar_aprendizaje()
    resumir()

    print("\n== Gestión y errores ==")
    comprobar_gestion()
    resumir()

    print(f"\nTotal: {time.time() - t0:.1f} s")
