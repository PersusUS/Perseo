"""Verificación del agente `memoria`, sobre un vault de usar y tirar.

Nunca toca el vault de verdad: se monta uno en un directorio temporal y se le
apunta el núcleo con `OBSIDIAN_VAULT_PATH`, la única variable que resuelve el
vault en todo el sistema (ver H-22).

Lo que más importa aquí no es que escriba, sino que **no destruya**: que anotar
dos veces añada en vez de reemplazar, y que ninguna ruta pedida se salga del
vault. Lo segundo no es paranoia abstracta — lo que Perseo lee viene de correos
y de pantallas, y un `../` metido en un asunto no puede acabar leyendo el disco.

    python perseo_core/verificar_memoria.py
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perseo_core import memoria  # noqa: E402
from perseo_core.arnes_pruebas import Nucleo, comprobar, resumir  # noqa: E402

CONTENIDO_VIEJO = "Se cambia el plato de ducha del bano pequeno."
CONTENIDO_NUEVO = "Presupuesto aceptado, empiezan el lunes."


def comprobar_en_proceso(raiz: Path) -> None:
    print("--- sobre el vault de prueba ---\n")
    vault = memoria.VaultFicheros(raiz)

    async def guion() -> None:
        # 1. Crear una nota.
        ruta = await vault.anotar("Reforma bano", CONTENIDO_VIEJO)
        fichero = raiz / ruta
        comprobar("Anotar crea la nota", fichero.is_file(), ruta)
        texto = fichero.read_text(encoding="utf-8")
        comprobar("Con su cabecera de Obsidian", texto.startswith("---\n"), texto.splitlines()[0])
        comprobar("Y con lo que se le dijo dentro", CONTENIDO_VIEJO in texto)

        # 2. Lo que de verdad importa: anotar otra vez **añade**. Sobrescribir una
        #    memoria es la perdida que no se nota hasta meses despues.
        await vault.anotar("Reforma bano", CONTENIDO_NUEVO)
        texto = fichero.read_text(encoding="utf-8")
        comprobar("Anotar de nuevo NO borra lo anterior", CONTENIDO_VIEJO in texto)
        comprobar("Y añade lo nuevo debajo", CONTENIDO_NUEVO in texto)
        comprobar(
            "Cada añadido lleva su fecha",
            texto.count("\n## ") >= 2,
            f"{texto.count(chr(10) + '## ')} seccion(es)",
        )

        # 3. Buscar: por contenido, por nombre, y sin que las tildes estorben.
        (raiz / "Notas").mkdir(exist_ok=True)
        (raiz / "Notas" / "Cumpleanos.md").write_text(
            "El cumpleaños de Aurelio es en marzo.", encoding="utf-8"
        )
        comprobar("Buscar encuentra por contenido", len(await vault.buscar("plato de ducha")) == 1)
        comprobar("Buscar encuentra por nombre de nota", len(await vault.buscar("cumplean")) == 1)
        encontradas = await vault.buscar("cumpleanos")
        comprobar(
            "Y las tildes no estorban (busca 'cumpleanos', encuentra 'cumpleaños')",
            len(encontradas) == 1,
            str([n.ruta for n in encontradas]),
        )
        comprobar("Lo que no está no aparece", await vault.buscar("hipopotamo") == [])
        if encontradas:
            comprobar("La ruta que devuelve es relativa al vault", not Path(encontradas[0].ruta).is_absolute(), encontradas[0].ruta)

        # 4. Leer.
        comprobar("Leer devuelve el contenido", CONTENIDO_NUEVO in await vault.leer(ruta))

        # 5. Nada sale del vault. Ni subiendo con `..`, ni con una ruta absoluta,
        #    ni por la carpeta de destino al anotar.
        # La segunda cambia con el sistema: en Linux `..\..\x` no sube ningun
        # directorio, es un nombre de fichero con barras invertidas, y la prueba
        # pasaria por el motivo equivocado.
        subir_dos = "..\..\secreto.md" if os.name == "nt" else "../../secreto.md"
        for intento in ("../secreto.md", subir_dos, "Notas/../../fuera.md"):
            try:
                await vault.leer(intento)
                comprobar(f"Se niega a leer {intento!r}", False, "no se nego")
            except memoria.FueraDelVault:
                comprobar(f"Se niega a leer {intento!r}", True)
            except FileNotFoundError:
                comprobar(f"Se niega a leer {intento!r}", False, "llego a buscarlo en disco")

        try:
            await vault.anotar("colada", "texto", carpeta="../fuera")
            comprobar("Se niega a anotar fuera del vault", False, "no se nego")
        except memoria.FueraDelVault:
            comprobar("Se niega a anotar fuera del vault", True)

        # 6. Un titulo que no deja nombre de fichero se rechaza en vez de acabar
        #    en un `.md` sin nombre.
        try:
            await vault.anotar("///", "texto")
            comprobar("Un titulo imposible se rechaza", False, "no se nego")
        except ValueError:
            comprobar("Un titulo imposible se rechaza", True)

    asyncio.run(guion())
    print()


def comprobar_de_punta_a_punta(raiz: Path) -> None:
    nucleo = Nucleo({"OBSIDIAN_VAULT_PATH": str(raiz)})
    nucleo.arrancar()
    token = nucleo.token

    # 7. El agente esta registrado y se le puede encargar trabajo por HTTP.
    codigo, trabajo = nucleo.pedir(
        "/trabajos",
        token,
        "POST",
        {
            "agente": "memoria",
            "peticion": {"accion": "anotar", "titulo": "Encargo de prueba", "texto": "Perseo estuvo aqui."},
        },
    )
    comprobar("El agente memoria existe para la API", codigo == 201, f"HTTP {codigo}")
    if codigo != 201:
        nucleo.limpiar()
        return

    hecho = nucleo.esperar_estado(int(trabajo["id"]), ("hecho", "fallido"), intentos=60)
    comprobar(
        "Anotar por HTTP termina bien",
        hecho.get("estado") == "hecho",
        f"estado={hecho.get('estado')} error={hecho.get('error')}",
    )
    escrita = raiz / (hecho.get("resultado") or {}).get("ruta", "")
    comprobar("Y la nota aparece en el vault", escrita.is_file(), str(escrita))

    # 8. Y se vuelve a encontrar buscando.
    _, busqueda = nucleo.pedir(
        "/trabajos",
        token,
        "POST",
        {"agente": "memoria", "peticion": {"accion": "buscar", "texto": "Perseo estuvo"}},
    )
    resultado = nucleo.esperar_estado(int(busqueda["id"]), ("hecho", "fallido"), intentos=60)
    notas = (resultado.get("resultado") or {}).get("notas") or []
    comprobar("Buscar por HTTP la encuentra", len(notas) == 1, f"{len(notas)} nota(s)")

    # 9. Una accion que no existe ni siquiera llega al agente: la politica de §7
    #    la para antes, porque lo que no esta clasificado es irreversible. Y si
    #    se aprueba, entonces si falla — sin tumbar al trabajador.
    _, malo = nucleo.pedir(
        "/trabajos", token, "POST", {"agente": "memoria", "peticion": {"accion": "borrar"}}
    )
    parado = nucleo.esperar_estado(int(malo["id"]), ("esperando", "fallido", "hecho"), intentos=60)
    comprobar(
        "Una accion sin clasificar se para y pregunta",
        parado.get("estado") == "esperando",
        str(parado.get("estado")),
    )
    nucleo.pedir(f"/trabajos/{malo['id']}/aprobar", token, "POST", {})
    fallido = nucleo.esperar_estado(int(malo["id"]), ("fallido", "hecho"), intentos=60)
    comprobar(
        "Y aprobada, falla por desconocida",
        fallido.get("estado") == "fallido",
        str(fallido.get("error")),
    )
    _, salud = nucleo.pedir("/salud")
    comprobar("Y el nucleo sigue en pie", bool(salud), str(salud)[:60])

    nucleo.limpiar()


def main() -> None:
    raiz = Path(tempfile.mkdtemp(prefix="perseo_vault_"))
    try:
        comprobar_en_proceso(raiz)
        comprobar_de_punta_a_punta(raiz)
    finally:
        shutil.rmtree(raiz, ignore_errors=True)
    resumir()


if __name__ == "__main__":
    main()
