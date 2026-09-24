"""Muestras que no son de quien dicen: la limpieza de los perfiles de antes.

Hasta el 2026-09-23 los perfiles se «reforzaban» solos: cualquier cara que
pasara el umbral por poco entraba en la galería de quien se pareciera, y la voz
se arrastraba medio camino hacia cada trozo oído. Así un perfil puede llevar
dentro muestras de otra persona —la cara del padre en el perfil del hijo—, y
esas muestras son las que hacen que se llame «señor Persus» a quien no es.

Esto las busca: una muestra es **sospechosa** si se parece más a la galería de
otra persona que al resto de la suya. No borra nada si no se le pide:

    python -m perseo_core.servicios.biometria_limpieza            # solo dice qué ve
    python -m perseo_core.servicios.biometria_limpieza --aplicar  # y lo quita

Lo que se quita queda anotado en `personas.log`. La última muestra de un perfil
no se quita nunca: un perfil vacío deja de reconocer a nadie, y eso se arregla
volviendo a enseñar la voz o la cara desde Ajustes, no dejándolo mudo.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from . import biometria_galeria as galeria
from . import biometria_perfiles as disco


def _galeria(perfil: dict[str, Any], canal: str) -> list[list[float]]:
    return galeria.voces(perfil) if canal == "voz" else list(perfil.get("caras") or [])


def sospechosas(perfiles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Cada muestra que se parece más a otra persona que a las suyas."""
    salida: list[dict[str, Any]] = []
    for canal in ("voz", "cara"):
        for nombre, perfil in perfiles.items():
            propias = _galeria(perfil, canal)
            for indice, vector in enumerate(propias):
                resto = [g for j, g in enumerate(propias) if j != indice]
                parecido_propio = max((galeria.coseno(vector, g) for g in resto), default=None)
                ajena, de = -1.0, ""
                for otro, perfil_otro in perfiles.items():
                    if otro == nombre:
                        continue
                    for g in _galeria(perfil_otro, canal):
                        similitud = galeria.coseno(vector, g)
                        if similitud > ajena:
                            ajena, de = similitud, otro
                if not de or parecido_propio is None or ajena <= parecido_propio:
                    continue
                salida.append(
                    {
                        "persona": nombre,
                        "canal": canal,
                        "indice": indice,
                        "propio": round(parecido_propio, 3),
                        "ajeno": round(ajena, 3),
                        "de": de,
                        "ultima": len(propias) == 1,
                    }
                )
    return salida


def limpiar(perfiles: dict[str, dict[str, Any]], lista: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Quita las sospechosas que se puedan quitar. Devuelve las que quitó."""
    quitadas: list[dict[str, Any]] = []
    # De atrás adelante, para que los índices sigan valiendo al quitar.
    for s in sorted(lista, key=lambda s: (s["persona"], s["canal"], -s["indice"])):
        perfil = perfiles[s["persona"]]
        propias = _galeria(perfil, s["canal"])
        if len(propias) <= 1:
            continue
        del propias[s["indice"]]
        if s["canal"] == "voz":
            galeria.poner_voces(perfil, propias)
        else:
            perfil["caras"] = propias
        quitadas.append(s)
    return quitadas


def _linea(s: dict[str, Any]) -> str:
    aviso = "  (es la única: vuelve a enseñarla desde Ajustes)" if s["ultima"] else ""
    return (
        f"  {s['persona']}: una muestra de {s['canal']} se parece más a {s['de']} "
        f"({s['ajeno']:.2f}) que a las suyas ({s['propio']:.2f}){aviso}"
    )


def main(argumentos: list[str]) -> int:
    from ..infra.configuracion import cargar_configuracion

    directorio = Path(cargar_configuracion().directorio_datos)
    perfiles = disco.cargar(directorio)
    lista = sospechosas(perfiles)
    if not lista:
        print("Ninguna muestra sospechosa: cada una se parece más a su perfil que a otro.")
        return 0
    print(f"{len(lista)} muestra(s) sospechosa(s):")
    for s in lista:
        print(_linea(s))
    if "--aplicar" not in argumentos:
        print("\nNo se ha tocado nada. Para quitarlas: --aplicar")
        return 0
    quitadas = limpiar(perfiles, lista)
    disco.guardar(directorio, perfiles)
    for s in quitadas:
        disco.anotar(
            directorio, "limpieza", s["persona"], f"fuera una muestra de {s['canal']} que era de {s['de']}"
        )
    print(f"\nQuitadas {len(quitadas)}. Queda anotado en {disco.NOMBRE_REGISTRO}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
