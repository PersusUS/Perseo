"""Verificación del agente `memoria`, sobre un vault de usar y tirar.

Nunca toca el vault de verdad: se monta uno en un directorio temporal y se le
apunta el núcleo con `OBSIDIAN_VAULT_PATH`, la única variable que resuelve el
vault en todo el sistema.

Lo que más importa aquí no es que escriba, sino que **no destruya**: que anotar
dos veces añada en vez de reemplazar, y que ninguna ruta pedida se salga del
vault. Lo segundo no es paranoia abstracta — lo que Perseo lee viene de correos
y de pantallas, y un `../` metido en un asunto no puede acabar leyendo el disco.

Y todo eso **dos veces**, una por respaldo: sobre ficheros y contra un plugin
Local REST API de mentira, que se levanta aquí mismo. Un puerto con dos
respaldos solo sirve si los dos cumplen las mismas reglas, y contra HTTP la de
no salir del vault se comprueba en un sitio distinto: no hay disco que resolver,
así que la ruta se para antes de mandarla o no se para nunca.

    python perseo_core/verificar_memoria.py
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perseo_core import memoria  # noqa: E402
from perseo_core.arnes_pruebas import Nucleo, comprobar, puerto_libre, resumir  # noqa: E402

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
    # Se comprueba que **está y es la primera**, no que sea la única. Desde el
    # 2026-08-16 la búsqueda también prueba palabra por palabra —"Perseo estuvo"
    # no aparece literal en ninguna parte y así se encontraba nada—, así que
    # ahora devuelve de más a propósito. Lo que importa es el orden: exigir un
    # solo resultado sería exigir que no busque bien.
    rutas = [str(n.get("ruta", "")) for n in notas]
    comprobar(
        "Buscar por HTTP la encuentra, y la primera",
        bool(rutas) and rutas[0].endswith("Encargo de prueba.md"),
        f"{len(notas)} nota(s): {rutas[:3]}",
    )

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


class PluginFalso:
    """Un Local REST API de Obsidian de mentira, para recorrer el camino real.

    Habla lo justo de lo que habla el plugin: `GET/PUT/POST /vault/<ruta>`,
    `POST /search/simple/` y el `GET /` que dice si la clave vale. Guarda las
    notas en memoria y **apunta cada petición**, que es como se comprueba lo más
    importante de todo: que una ruta con `../` no llega ni a salir de casa.
    """

    CLAVE = "clave-de-mentira"

    def __init__(self) -> None:
        self.puerto = puerto_libre()
        self.notas: dict[str, str] = {}
        self.peticiones: list[tuple[str, str]] = []
        self._servidor = ThreadingHTTPServer(("127.0.0.1", self.puerto), self._manejador())
        self._servidor.daemon_threads = True

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.puerto}"

    def arrancar(self) -> None:
        threading.Thread(target=self._servidor.serve_forever, daemon=True).start()

    def parar(self) -> None:
        self._servidor.shutdown()

    def _manejador(self):
        plugin = self

        class Manejador(BaseHTTPRequestHandler):
            def log_message(self, *_: Any) -> None:
                pass

            # -- utilidades ------------------------------------------------- #

            def _autorizado(self) -> bool:
                return self.headers.get("Authorization") == f"Bearer {plugin.CLAVE}"

            def _responder(self, codigo: int, cuerpo: str = "", tipo: str = "text/plain") -> None:
                crudo = cuerpo.encode("utf-8")
                self.send_response(codigo)
                self.send_header("Content-Type", f"{tipo}; charset=utf-8")
                self.send_header("Content-Length", str(len(crudo)))
                self.end_headers()
                self.wfile.write(crudo)

            def _ruta(self) -> str:
                trozos = urllib.parse.urlsplit(self.path)
                return urllib.parse.unquote(trozos.path[len("/vault/") :])

            def _cuerpo(self) -> str:
                largo = int(self.headers.get("Content-Length", "0"))
                return self.rfile.read(largo).decode("utf-8") if largo else ""

            def _apuntar(self, metodo: str) -> bool:
                plugin.peticiones.append((metodo, self.path))
                if not self._autorizado():
                    self._responder(401, '{"errorCode": 40100}', "application/json")
                    return False
                return True

            # -- verbos ----------------------------------------------------- #

            def do_GET(self) -> None:  # noqa: N802
                if not self._apuntar("GET"):
                    return
                if self.path == "/":
                    self._responder(
                        200,
                        json.dumps({"authenticated": True, "service": "Obsidian Local REST API"}),
                        "application/json",
                    )
                    return
                ruta = self._ruta()
                if ruta not in plugin.notas:
                    self._responder(404, '{"errorCode": 40400}', "application/json")
                    return
                self._responder(200, plugin.notas[ruta], "text/markdown")

            def do_PUT(self) -> None:  # noqa: N802
                if not self._apuntar("PUT"):
                    return
                plugin.notas[self._ruta()] = self._cuerpo()
                self._responder(204)

            def do_POST(self) -> None:  # noqa: N802
                if not self._apuntar("POST"):
                    return
                if self.path.startswith("/search/simple/"):
                    consulta = urllib.parse.parse_qs(
                        urllib.parse.urlsplit(self.path).query
                    ).get("query", [""])[0]
                    hallazgos = [
                        {"filename": nombre, "matches": [{"context": consulta}]}
                        for nombre, texto in sorted(plugin.notas.items())
                        if consulta and consulta in texto
                    ]
                    self._responder(200, json.dumps(hallazgos), "application/json")
                    return
                # En el plugin, POST **añade** al final. Es lo que sostiene la
                # regla de no sobrescribir cuando el vault está detrás de HTTP.
                ruta = self._ruta()
                plugin.notas[ruta] = plugin.notas.get(ruta, "") + self._cuerpo()
                self._responder(204)

        return Manejador


def comprobar_por_el_plugin() -> None:
    print("--- contra un plugin de Obsidian de mentira ---\n")
    plugin = PluginFalso()
    plugin.arrancar()
    vault = memoria.VaultRest(plugin.url, PluginFalso.CLAVE)

    async def guion() -> None:
        # 1. La clave se comprueba antes que nada: sin ella no hay nada que hacer.
        comprobar("El plugin contesta y reconoce la clave", "reconoce" in await vault.comprobar())

        # 2. Crear la nota: PUT una vez, y con la cabecera de Obsidian.
        ruta = await vault.anotar("Reforma bano", CONTENIDO_VIEJO)
        comprobar("Anotar crea la nota en el plugin", ruta in plugin.notas, ruta)
        comprobar("Con su cabecera", plugin.notas[ruta].startswith("---\n"))
        comprobar("Y con lo que se le dijo dentro", CONTENIDO_VIEJO in plugin.notas[ruta])
        comprobar(
            "Se crea con PUT, que es lo unico que reemplaza",
            [m for m, _ in plugin.peticiones].count("PUT") == 1,
            str(plugin.peticiones),
        )

        # 3. Lo que de verdad importa: la segunda vez **añade**, y lo hace con
        #    POST. Un PUT aqui borraria la nota entera sin decir nada.
        await vault.anotar("Reforma bano", CONTENIDO_NUEVO)
        comprobar("Anotar de nuevo NO borra lo anterior", CONTENIDO_VIEJO in plugin.notas[ruta])
        comprobar("Y añade lo nuevo debajo", CONTENIDO_NUEVO in plugin.notas[ruta])
        comprobar(
            "Sin un segundo PUT: se añade con POST",
            [m for m, _ in plugin.peticiones].count("PUT") == 1,
            str([m for m, _ in plugin.peticiones]),
        )

        # 4. Leer, y no leer lo que no hay.
        comprobar("Leer devuelve el contenido", CONTENIDO_NUEVO in await vault.leer(ruta))
        try:
            await vault.leer("Notas/No existe.md")
            comprobar("Una nota que no existe lo dice", False, "no se quejo")
        except FileNotFoundError:
            comprobar("Una nota que no existe lo dice", True)

        # 5. Buscar: lo que devuelve el plugin se traduce a notas.
        encontradas = await vault.buscar("plato de ducha")
        comprobar("Buscar encuentra la nota", len(encontradas) == 1, str([n.ruta for n in encontradas]))
        if encontradas:
            comprobar("La ruta que devuelve es relativa al vault", not Path(encontradas[0].ruta).is_absolute(), encontradas[0].ruta)
        comprobar("Lo que no está no aparece", await vault.buscar("hipopotamo") == [])

        # 6. Y lo que no puede pasar: una ruta con `..` no llega a la red. Sin
        #    disco que resolver, esta comprobacion es la unica que hay.
        antes = len(plugin.peticiones)
        subir_dos = "..\..\secreto.md" if os.name == "nt" else "../../secreto.md"
        for intento in ("../secreto.md", subir_dos, "Notas/../../fuera.md"):
            try:
                await vault.leer(intento)
                comprobar(f"Se niega a leer {intento!r}", False, "no se nego")
            except memoria.FueraDelVault:
                comprobar(f"Se niega a leer {intento!r}", True)
        try:
            await vault.anotar("colada", "texto", carpeta="../fuera")
            comprobar("Se niega a anotar fuera del vault", False, "no se nego")
        except memoria.FueraDelVault:
            comprobar("Se niega a anotar fuera del vault", True)
        comprobar(
            "Y ninguna de esas rutas llego a salir a la red",
            len(plugin.peticiones) == antes,
            f"{len(plugin.peticiones) - antes} peticion(es) de mas",
        )

        await vault.cerrar()

        # 7. Una clave que no vale se dice claramente, no con un 401 pelado.
        malo = memoria.VaultRest(plugin.url, "no es la clave")
        try:
            await malo.anotar("Lo que sea", "texto")
            comprobar("Una clave mala se explica", False, "no se quejo")
        except RuntimeError as e:
            comprobar("Una clave mala se explica", "clave" in str(e).lower(), str(e)[:80])
        finally:
            await malo.cerrar()

        # 8. Y el plugin apagado tampoco es una traza: es un mensaje que dice
        #    que hay que abrir Obsidian.
        apagado = memoria.VaultRest(f"http://127.0.0.1:{puerto_libre()}", PluginFalso.CLAVE)
        try:
            await apagado.leer("Notas/X.md")
            comprobar("El plugin apagado se explica", False, "no se quejo")
        except RuntimeError as e:
            comprobar("El plugin apagado se explica", "Obsidian" in str(e), str(e)[:80])
        finally:
            await apagado.cerrar()

    try:
        asyncio.run(guion())
    finally:
        plugin.parar()
    print()


def comprobar_de_punta_a_punta_por_el_plugin() -> None:
    """El nucleo entero con el vault detras de HTTP, sin tocar disco."""
    plugin = PluginFalso()
    plugin.arrancar()
    nucleo = Nucleo(
        {
            "PERSEO_VAULT": "rest",
            "PERSEO_VAULT_REST": plugin.url,
            "PERSEO_VAULT_CLAVE": PluginFalso.CLAVE,
        }
    )
    try:
        nucleo.arrancar()
        _, trabajo = nucleo.pedir(
            "/trabajos",
            nucleo.token,
            "POST",
            {
                "agente": "memoria",
                "peticion": {
                    "accion": "anotar",
                    "titulo": "Por el plugin",
                    "texto": "Esto lo escribio Obsidian.",
                },
            },
        )
        hecho = nucleo.esperar_estado(int(trabajo["id"]), ("hecho", "fallido"), intentos=60)
        comprobar(
            "Con PERSEO_VAULT=rest el nucleo anota por el plugin",
            hecho.get("estado") == "hecho",
            f"estado={hecho.get('estado')} error={hecho.get('error')}",
        )
        ruta = (hecho.get("resultado") or {}).get("ruta", "")
        comprobar("Y la nota esta en el plugin, no en disco", ruta in plugin.notas, str(list(plugin.notas)))
    finally:
        nucleo.limpiar()
        plugin.parar()
    print()


def main() -> None:
    raiz = Path(tempfile.mkdtemp(prefix="perseo_vault_"))
    try:
        comprobar_en_proceso(raiz)
        comprobar_de_punta_a_punta(raiz)
        comprobar_por_el_plugin()
        comprobar_de_punta_a_punta_por_el_plugin()
    finally:
        shutil.rmtree(raiz, ignore_errors=True)
    resumir()


if __name__ == "__main__":
    main()
