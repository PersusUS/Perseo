"""Agente `memoria`: leer y escribir el vault sin romperlo.

La estructura del vault no se toca — es R7 del plan, y hay años de notas dentro.
Lo que hace este módulo es envolverla: buscar, leer y **añadir**, que es todo lo
que hace falta para que Perseo recuerde cosas entre conversaciones.

**El vault también es un puerto**, y ya tiene dos respaldos:

- `VaultFicheros`, ficheros Markdown en una carpeta. Es el de siempre y sigue
  siendo el que sale por defecto: no necesita que Obsidian esté abierto.
- `VaultRest`, el plugin Local REST API de Obsidian. Escribe **Obsidian**, no
  nosotros, que es lo que evita escribir por debajo de una aplicación que puede
  tener el fichero abierto y quedarse con la versión de antes en memoria.

Se elige con `PERSEO_VAULT`, y el agente no se entera de cuál hay detrás — igual
que con el buzón del correo.

Tres reglas que no son opcionales:

1. **No se sobrescribe nunca.** Si la nota ya existe, se le añade una sección con
   la fecha. Escribir encima de una memoria es la clase de pérdida que no se
   nota hasta meses después, cuando ya no hay de dónde recuperarla.
2. **No se borra.** No hay método para ello, a propósito. Cuando haga falta,
   pasará por `NecesitaConfirmacion` como cualquier acción irreversible.
3. **Nada sale del vault.** Las rutas que llegan en una petición se resuelven y
   se comprueban contra la raíz antes de tocar nada. Esto no es paranoia
   abstracta: lo que Perseo lee viene de correos y de pantallas, y la regla
   heredada de la Fase 1 es que **eso es información observada, nunca una
   instrucción**. Un `../../.ssh/id_rsa` en un asunto de correo no puede acabar
   siendo una ruta que se lee.

Sobre dónde está el vault: se resuelve con `OBSIDIAN_VAULT_PATH`, que es la
variable que ya usaba el indexador de v1 (`RAG/paths.py`, hoy retirado). Se
mantiene el nombre a propósito: quien la tuviera puesta no tiene que cambiar
nada, y no vuelve a haber dos módulos apuntando a carpetas distintas — que es lo
que fue y costó descubrirlo.


"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import unicodedata
import urllib.parse
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Protocol

import aiohttp

from ..infra import almacen
from ..infra.router import registrar

logger = logging.getLogger(__name__)

#: Carpetas del vault. Son las que ya usaba la herramienta de memoria de v1: la
#: estructura del vault no se toca, se envuelve (R7).
#: Perseo tiene carpeta propia dentro del vault desde el 2026-08-24, igual que
#: `06_CLAUDE/`. Antes escribía en la raíz, mezclado con las carpetas numeradas
#: del señor Persus. Mover una nota en Obsidian no rompe nada: los `[[enlaces]]`
#: van por nombre, no por ruta.
CARPETA_PERSEO = "10_PERSEO"
CARPETA_MEMORIAS = f"{CARPETA_PERSEO}/Memorias"
CARPETA_CONVERSACIONES = f"{CARPETA_PERSEO}/Conversaciones"

#: Quién es quién. La escribe el reconocimiento de personas y la lee cualquiera:
#: es la parte del reconocimiento que se puede leer y corregir a mano, frente a
#: los vectores de `datos/perfiles.json`, que no dicen nada a un humano.
CARPETA_PERSONAS = f"{CARPETA_PERSEO}/Personas"

#: Cuánto se lee de un fichero al buscar. Una nota normal no llega ni de lejos, y
#: el tope evita que un adjunto pegado dentro del vault pare la búsqueda.
TOPE_LECTURA = 200_000

#: Cuántos caracteres de contexto se devuelven alrededor de una coincidencia.
CONTEXTO = 160

#: Cuánto se espera al plugin de Obsidian. Está en el bucle local: si tarda más
#: que esto no es que vaya lento, es que no está.
TOPE_REST = 20.0


@dataclass(frozen=True)
class Nota:
    """Una nota del vault, vista desde fuera. La ruta es siempre relativa a la raíz."""

    ruta: str
    titulo: str
    extracto: str = ""
    modificada: str = ""

    def a_dict(self) -> dict[str, Any]:
        return asdict(self)


class Vault(Protocol):
    """Qué se le puede pedir a la memoria. Lo implementa cada respaldo."""

    async def buscar(self, consulta: str, limite: int = 10) -> list[Nota]: ...

    async def leer(self, ruta: str) -> str: ...

    async def anotar(self, titulo: str, texto: str, carpeta: str = CARPETA_MEMORIAS) -> str: ...


class FueraDelVault(Exception):
    """La ruta pedida no cae dentro del vault. Nunca es un caso normal."""


class PluginCaido(RuntimeError):
    """No se pudo hablar con el plugin de Obsidian: cerrado, apagado o sin red.

    Se distingue de los demás errores del plugin —una clave rechazada, un 500—
    porque este sí tiene remedio en el acto: las notas están en el disco y se
    pueden leer sin Obsidian. Ver `VaultRest.respaldo`.
    """


class VaultFicheros:
    """El vault tal y como está hoy: ficheros Markdown en una carpeta."""

    def __init__(self, raiz: Path) -> None:
        self.raiz = raiz.resolve()

    # -- rutas -------------------------------------------------------------- #

    def _resolver(self, ruta: str) -> Path:
        """Convierte una ruta pedida en una ruta real, o se niega.

        `resolve()` deshace los `..` y los enlaces antes de comparar; comparar
        cadenas sin resolver es exactamente como se cuela un `../../`.
        """
        candidata = (self.raiz / ruta).resolve()
        if candidata != self.raiz and self.raiz not in candidata.parents:
            raise FueraDelVault(f"{ruta!r} cae fuera del vault")
        return candidata

    # -- lectura ------------------------------------------------------------ #

    async def buscar(self, consulta: str, limite: int = 10) -> list[Nota]:
        return await asyncio.to_thread(self._buscar, consulta, limite)

    def _buscar(self, consulta: str, limite: int) -> list[Nota]:
        aguja = _plegar(consulta)
        if not aguja:
            return []

        encontradas: list[Nota] = []
        for fichero in sorted(self.raiz.rglob("*.md")):
            if len(encontradas) >= limite:
                break
            try:
                contenido = fichero.read_text(encoding="utf-8", errors="replace")[:TOPE_LECTURA]
            except OSError as e:
                logger.warning("No se pudo leer %s (%s).", fichero, e)
                continue

            plegado = _plegar(contenido)
            posicion = plegado.find(aguja)
            en_el_nombre = aguja in _plegar(fichero.stem)
            if posicion < 0 and not en_el_nombre:
                continue

            encontradas.append(
                Nota(
                    ruta=fichero.relative_to(self.raiz).as_posix(),
                    titulo=fichero.stem,
                    extracto=_recortar(contenido, posicion),
                    modificada=_fecha(fichero),
                )
            )
        return encontradas

    async def leer(self, ruta: str) -> str:
        return await asyncio.to_thread(self._leer, ruta)

    def _leer(self, ruta: str) -> str:
        fichero = self._resolver(ruta)
        if not fichero.is_file():
            raise FileNotFoundError(f"No hay ninguna nota en {ruta!r}")
        return fichero.read_text(encoding="utf-8", errors="replace")

    # -- escritura ---------------------------------------------------------- #

    async def anotar(self, titulo: str, texto: str, carpeta: str = CARPETA_MEMORIAS) -> str:
        return await asyncio.to_thread(self._anotar, titulo, texto, carpeta)

    def _anotar(self, titulo: str, texto: str, carpeta: str) -> str:
        nombre = _nombre_seguro(titulo)
        if not nombre:
            raise ValueError("El título no deja ningún nombre de fichero utilizable.")

        destino = self._resolver(f"{carpeta}/{nombre}.md")
        destino.parent.mkdir(parents=True, exist_ok=True)
        momento = _momento()

        if destino.exists():
            # Añadir, nunca reemplazar. Ver la regla 1 de la cabecera.
            with destino.open("a", encoding="utf-8") as nota:
                nota.write(_seccion(texto, momento))
        else:
            destino.write_text(_nota_nueva(titulo, texto, momento), encoding="utf-8")
        return destino.relative_to(self.raiz).as_posix()


class VaultRest:
    """El vault a través del plugin Local REST API de Obsidian.

    Los mismos tres métodos, otro respaldo. La diferencia que importa es quién
    escribe: aquí el fichero lo toca **Obsidian**, así que no se escribe por
    debajo de una aplicación que puede tenerlo abierto y luego guardar encima lo
    que tenía en memoria.

    Tres cosas que conviene saber antes de tocar esto:

    1. **La regla de no salir del vault se comprueba aquí y solo aquí.** Con
       ficheros hay un `resolve()` contra la raíz; contra el plugin no hay disco
       que resolver, la ruta viaja dentro de la URL y el único sitio donde se
       puede parar un `../` es antes de mandarla. Ver `_ruta_relativa`.
    2. **Añadir es un POST.** En este plugin `PUT` reemplaza el fichero entero y
       `POST` añade al final, así que la regla 1 de la cabecera —no se
       sobrescribe nunca— aquí se traduce en no usar `PUT` sobre algo que ya
       existe. Solo se usa para crear la nota con su cabecera de Obsidian.
    3. **Buscar no dobla las tildes.** La búsqueda la hace el plugin, y la suya
       es literal: buscando `cumpleanos` no aparece `cumpleaños`, que con
       ficheros sí. Es la única diferencia de comportamiento entre los dos
       respaldos, y está aquí anotada para que no se descubra por las malas.

    El plugin sirve HTTPS en el bucle local con un certificado que se firma él
    mismo; ver `_verificar_certificado`.
    """

    def __init__(
        self,
        base: str,
        clave: str,
        tope: float = TOPE_REST,
        respaldo: VaultFicheros | None = None,
    ) -> None:
        self.base = base.rstrip("/")
        self._clave = clave
        self._tope = tope
        #: El vault en disco, para cuando Obsidian está cerrado. Las notas no
        #: se van a ningún sitio porque el programa que las enseña no esté
        #: abierto: buscar y leer siguen funcionando por el disco, y anotar
        #: escribe el fichero que Obsidian recogerá cuando vuelva. Sin esto,
        #: cerrar Obsidian dejaba a Perseo sin memoria entera.
        self.respaldo = respaldo
        self._avisado_caido = False
        self._http: aiohttp.ClientSession | None = None
        # Las anotaciones, una a la vez. El flujo "¿existe? → creo / añado" no
        # es atómico en el plugin: dos `anotar` entrelazados podían verse ambos
        # un 404 y el segundo PUT habría reemplazado la nota que acababa de
        # crear el primero — exactamente la pérdida que prohíbe la regla 1.
        self._cerrojo_escritura = asyncio.Lock()

    # -- transporte --------------------------------------------------------- #

    def _url(self, camino: str, ruta: str = "") -> str:
        # `quote` con `safe="/"`: los espacios y las tildes de un nombre de nota
        # tienen que viajar escapados, pero las barras son la jerarquía.
        return f"{self.base}{camino}{urllib.parse.quote(ruta)}"

    async def _abrir(self) -> aiohttp.ClientSession:
        if self._http is None:
            self._http = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._tope),
                headers={"Authorization": f"Bearer {self._clave}"},
                connector=aiohttp.TCPConnector(
                    ssl=None if _verificar_certificado(self.base) else False
                ),
            )
        return self._http

    async def cerrar(self) -> None:
        if self._http is not None:
            await self._http.close()
            self._http = None

    async def _pedir(
        self,
        metodo: str,
        url: str,
        *,
        cuerpo: str | None = None,
        cabeceras: dict[str, str] | None = None,
        parametros: dict[str, str] | None = None,
    ) -> tuple[int, str]:
        """Una petición al plugin. Devuelve el estado y el texto, sin juzgarlos."""
        http = await self._abrir()
        try:
            async with http.request(
                metodo,
                url,
                data=None if cuerpo is None else cuerpo.encode("utf-8"),
                headers=cabeceras,
                params=parametros,
            ) as respuesta:
                return respuesta.status, await respuesta.text()
        except aiohttp.ClientError as e:
            raise PluginCaido(
                f"No se pudo hablar con el plugin de Obsidian en {self.base}: {e}. "
                "¿Está Obsidian abierto y el plugin Local REST API encendido?"
            ) from None

    def _comprobar(self, estado: int, cuerpo: str) -> None:
        if 200 <= estado < 300:
            return
        if estado in (401, 403):
            raise RuntimeError(
                f"El plugin de Obsidian rechazó la clave ({estado}). "
                "Se saca de los ajustes del plugin; se pone en PERSEO_VAULT_CLAVE."
            )
        raise RuntimeError(f"El plugin de Obsidian respondió {estado}: {cuerpo[:200]}")

    # -- lectura ------------------------------------------------------------ #

    def _al_disco(self, faena: str, caido: PluginCaido) -> VaultFicheros:
        """El vault de disco cuando el plugin no está, o el error si no lo hay."""
        if self.respaldo is None:
            raise caido
        if not self._avisado_caido:
            # Una vez y no en cada búsqueda: el registro no se llena de lo
            # mismo mientras Obsidian siga cerrado.
            logger.warning("El plugin de Obsidian no contesta (%s); se sigue por disco.", faena)
            self._avisado_caido = True
        return self.respaldo

    async def buscar(self, consulta: str, limite: int = 10) -> list[Nota]:
        consulta = consulta.strip()
        if not consulta:
            return []
        try:
            return await self._buscar_por_el_plugin(consulta, limite)
        except PluginCaido as caido:
            return await self._al_disco("buscar", caido).buscar(consulta, limite)

    async def _buscar_por_el_plugin(self, consulta: str, limite: int) -> list[Nota]:
        estado, cuerpo = await self._pedir(
            "POST",
            self._url("/search/simple/"),
            parametros={"query": consulta, "contextLength": str(CONTEXTO)},
        )
        self._comprobar(estado, cuerpo)
        try:
            crudas = json.loads(cuerpo)
        except json.JSONDecodeError:
            raise RuntimeError("El plugin devolvió algo que no es JSON al buscar.") from None
        return _notas_de_busqueda(crudas, limite)

    async def leer(self, ruta: str) -> str:
        relativa = _ruta_relativa(ruta)
        try:
            estado, cuerpo = await self._pedir(
                "GET", self._url("/vault/", relativa), cabeceras={"Accept": "text/markdown"}
            )
        except PluginCaido as caido:
            return await self._al_disco("leer", caido).leer(ruta)
        if estado == 404:
            raise FileNotFoundError(f"No hay ninguna nota en {ruta!r}")
        self._comprobar(estado, cuerpo)
        return cuerpo

    # -- escritura ---------------------------------------------------------- #

    async def anotar(self, titulo: str, texto: str, carpeta: str = CARPETA_MEMORIAS) -> str:
        try:
            return await self._anotar_por_el_plugin(titulo, texto, carpeta)
        except PluginCaido as caido:
            # El fichero se escribe igual y Obsidian lo recoge al abrirse. Lo
            # que no puede pasar es perder lo que había que apuntar.
            return await self._al_disco("anotar", caido).anotar(titulo, texto, carpeta)

    async def _anotar_por_el_plugin(self, titulo: str, texto: str, carpeta: str) -> str:
        nombre = _nombre_seguro(titulo)
        if not nombre:
            raise ValueError("El título no deja ningún nombre de fichero utilizable.")

        relativa = _ruta_relativa(f"{carpeta}/{nombre}.md")
        url = self._url("/vault/", relativa)
        momento = _momento()

        async with self._cerrojo_escritura:
            estado, cuerpo = await self._pedir(
                "GET", url, cabeceras={"Accept": "text/markdown"}
            )
            if estado == 404:
                estado, cuerpo = await self._pedir(
                    "PUT",
                    url,
                    cuerpo=_nota_nueva(titulo, texto, momento),
                    cabeceras={"Content-Type": "text/markdown"},
                )
            else:
                # Un GET que falla por otra cosa —la clave, el plugin caído— no es
                # "la nota no existe". Sin esta comprobación se contestaría creándola
                # de cero, que es exactamente la pérdida que prohíbe la regla 1.
                self._comprobar(estado, cuerpo)
                estado, cuerpo = await self._pedir(
                    "POST",
                    url,
                    cuerpo=_seccion(texto, momento),
                    cabeceras={"Content-Type": "text/markdown"},
                )
            self._comprobar(estado, cuerpo)
        return relativa

    # -- diagnóstico -------------------------------------------------------- #

    async def comprobar(self) -> str:
        """Pregunta al plugin quién es. Para usarlo a mano, ver `_sincrono`."""
        estado, cuerpo = await self._pedir("GET", self._url("/"))
        self._comprobar(estado, cuerpo)
        try:
            datos = json.loads(cuerpo)
        except json.JSONDecodeError:
            datos = {}
        if not datos.get("authenticated"):
            raise RuntimeError(
                "El plugin contesta pero no reconoce la clave: revisa PERSEO_VAULT_CLAVE."
            )
        return f"El plugin de Obsidian responde en {self.base} y reconoce la clave."


def _verificar_certificado(base: str) -> bool:
    """Si hay que verificar el certificado del otro lado.

    El plugin sirve HTTPS con un certificado que se firma él mismo, así que
    verificarlo sin instalar su autoridad es imposible. No verificarlo solo es
    aceptable porque la conexión no sale de la máquina: contra cualquier host que
    no sea el bucle local se verifica, y si el certificado no vale la petición
    falla — que es lo que tiene que pasar.
    """
    trozos = urllib.parse.urlsplit(base)
    if trozos.scheme != "https":
        return True
    return (trozos.hostname or "") not in almacen.LOCALES


def _ruta_relativa(ruta: str) -> str:
    """La misma regla que `VaultFicheros._resolver`, pero sin disco que mirar.

    Aquí no hay `resolve()` que deshaga los `..`: la ruta se mete en una URL y se
    manda. Así que se mira la ruta escrita y se rechaza cualquier cosa que pueda
    apuntar fuera —raíz, letra de unidad, un `..` en cualquier tramo—, con las
    barras invertidas tratadas como separador para que `..\\..\\x` no cuele en
    Linux por ser allí un nombre de fichero válido. Lo que Perseo lee viene de
    correos y de pantallas: fallar por exceso cuesta un error visible.
    """
    limpia = str(ruta).replace("\\", "/").strip()
    if not limpia:
        raise FueraDelVault("Una ruta vacía no apunta a ninguna nota.")
    if limpia.startswith("/") or PureWindowsPath(limpia).drive:
        raise FueraDelVault(f"{ruta!r} no es una ruta dentro del vault")
    partes = [p for p in PurePosixPath(limpia).parts if p not in ("", ".")]
    if any(p == ".." for p in partes):
        raise FueraDelVault(f"{ruta!r} cae fuera del vault")
    if not partes:
        raise FueraDelVault("Una ruta vacía no apunta a ninguna nota.")
    return "/".join(partes)


def _notas_de_busqueda(crudas: Any, limite: int) -> list[Nota]:
    """Traduce lo que devuelve `/search/simple/` a la nota que ve el agente.

    El plugin no dice cuándo se modificó cada nota, así que `modificada` se queda
    vacía: es preferible a inventar una fecha que luego se lee como buena.
    """
    notas: list[Nota] = []
    for cruda in crudas if isinstance(crudas, list) else []:
        if not isinstance(cruda, dict):
            continue
        ruta = str(cruda.get("filename", "")).strip()
        if not ruta:
            continue
        coincidencias = cruda.get("matches") or []
        primera = coincidencias[0] if coincidencias and isinstance(coincidencias[0], dict) else {}
        notas.append(
            Nota(
                ruta=ruta,
                titulo=PurePosixPath(ruta).stem,
                extracto=str(primera.get("context", "")).strip()[:CONTEXTO],
            )
        )
        if len(notas) >= limite:
            break
    return notas


def _momento() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _seccion(texto: str, momento: str) -> str:
    """Lo que se añade a una nota que ya existe. Los dos respaldos escriben esto."""
    return f"\n\n## {momento}\n\n{texto.strip()}\n"


def _nota_nueva(titulo: str, texto: str, momento: str) -> str:
    """Una nota recién creada, con la cabecera que espera Obsidian."""
    return (
        "---\n"
        "tipo: memoria\n"
        f"titulo: {titulo}\n"
        f"fecha_creacion: {momento}\n"
        "tags: [perseo]\n"
        "---\n"
        f"# {titulo}\n\n"
        f"## {momento}\n\n"
        f"{texto.strip()}\n"
    )


def _plegar(texto: str) -> str:
    """Minúsculas y sin tildes, para que buscar 'bano' encuentre 'baño'.

    Buscar con tildes exactas en un vault escrito a mano no encuentra nada la
    mitad de las veces, y quien busca no sabe por qué.
    """
    sin_tildes = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in sin_tildes if not unicodedata.combining(c)).casefold()


def _recortar(contenido: str, posicion: int) -> str:
    if posicion < 0:
        return contenido[:CONTEXTO].strip()
    desde = max(0, posicion - CONTEXTO // 2)
    return contenido[desde : desde + CONTEXTO].strip()


def _fecha(fichero: Path) -> str:
    try:
        return datetime.fromtimestamp(fichero.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return ""


def _nombre_seguro(texto: str) -> str:
    """Deja solo lo que vale como nombre de fichero."""
    return "".join(c for c in texto if c.isalnum() or c in (" ", "_", "-")).strip()


# --------------------------------------------------------------------------- #
# El agente
# --------------------------------------------------------------------------- #

_vault: Vault | None = None


def ruta_vault(cfg: almacen.Configuracion | None = None) -> Path:
    """Dónde está el vault. Una sola variable para todo el sistema

    El respaldo —`<repositorio>/../obsidian_vault`— es para un clon recién
    hecho, sin configurar: se crea al primer apunte y sirve para probar. **El
    vault de verdad se dice en `OBSIDIAN_VAULT_PATH`**, y Perseo escribe
    siempre dentro de su
    propia carpeta, `10_PERSEO/`. La carpeta de pruebas que había en el
    repositorio se vació ese día: sus notas están en el vault grande.
    """
    if cfg is not None and cfg.vault:
        return Path(cfg.vault)
    return Path(os.environ.get("OBSIDIAN_VAULT_PATH", almacen.RAIZ.parent / "obsidian_vault"))


def iniciar(cfg: almacen.Configuracion) -> Vault:
    global _vault
    if _vault is None:
        _vault = _elegir_respaldo(cfg)
    return _vault


def respaldo() -> Vault | None:
    """El vault que está en pie, o `None` si nadie ha llamado a `iniciar`.

    Lo usa la pantalla de estado para preguntarle al plugin de Obsidian por la
    sesión que ya está abierta, en vez de montar una nueva en cada sondeo.
    """
    return _vault


def _elegir_respaldo(cfg: almacen.Configuracion) -> Vault:
    """Qué hay detrás del puerto. Ficheros salvo que se pida el plugin.

    Pedir el plugin sin dar su clave **no es un error**: se avisa y se sigue con
    ficheros, igual que hace el correo con unas credenciales que no están. Un
    vault a medio configurar no puede dejar al núcleo sin memoria.
    """
    if cfg.vault_respaldo == "rest":
        if cfg.vault_rest_clave:
            logger.info("Memoria por el plugin de Obsidian en %s", cfg.vault_rest_url)
            # Con el disco detrás: Obsidian cerrado no puede dejar sin memoria
            # a quien tiene las notas delante, en su carpeta.
            en_disco = ruta_vault(cfg)
            en_disco.mkdir(parents=True, exist_ok=True)
            return VaultRest(
                cfg.vault_rest_url,
                cfg.vault_rest_clave,
                respaldo=VaultFicheros(en_disco),
            )
        logger.warning(
            "PERSEO_VAULT=rest pero no hay clave del plugin (PERSEO_VAULT_CLAVE ni %s). "
            "Se sigue escribiendo en ficheros.",
            cfg.directorio_datos / "obsidian.txt",
        )

    raiz = ruta_vault(cfg)
    raiz.mkdir(parents=True, exist_ok=True)
    logger.info("Memoria sobre el vault en %s", raiz)
    return VaultFicheros(raiz)


async def detener() -> None:
    global _vault
    if isinstance(_vault, VaultRest):
        await _vault.cerrar()
    _vault = None



#: Palabras que no distinguen nada. Se caen al trocear una consulta larga: buscar
#: "de" en un vault devuelve el vault entero.
_VACIAS = frozenset(
    """a al algo ante como con contra cual cuando de del desde donde dos e el ella
    ellos en entre era eres es esa ese eso esta este esto ha hay la las le les lo
    los mas me mi mis muy no nos o os para pero por que se sea segun si sin sobre
    son su sus te tiene todo tu tus un una uno unos y ya""".split()
)


#: Cuántas palabras se prueban como mucho. Cada una es una petición al vault, y
#: una frase larga no mejora por buscar su décima palabra.
TOPE_TERMINOS = 4

#: Cuántos resultados se traen para poder ordenarlos, antes de quedarse con los
#: que se enseñan. Es local: el vault ya los tiene todos.
TOPE_CANDIDATAS = 200


def _terminos(consulta: str) -> list[str]:
    """Las palabras de una consulta que valen la pena, sin tocar las mayúsculas.

    Se conserva el caso original: la búsqueda la hace el vault y no es asunto de
    aquí decidir si distingue mayúsculas.
    """
    palabras = [p.strip(".,;:¿?¡!()[]\"'«»") for p in consulta.split()]
    utiles = [p for p in palabras if len(p) >= 3 and p.lower() not in _VACIAS]
    # Sin duplicados y estable, que `sorted` sobre un set baila entre ejecuciones.
    vistas: list[str] = []
    for p in utiles:
        if p not in vistas:
            vistas.append(p)
    return sorted(vistas, key=len, reverse=True)


def _prioridad(termino: str, nota: Nota) -> int:
    """Cómo de bien encaja una nota con lo que se buscó. Menos es mejor.

    0 — el término **es** una palabra del nombre o de la ruta. Es lo que uno
        quiere decir al preguntar por "el proyecto MAGI": la nota que se llama
        así, no las cuatro que lo mencionan.
    1 — el término aparece dentro de otra palabra del nombre. Cuenta, pero poco:
        buscando `MAGI`, `MagicOCR.md` encajaba aquí y se colaba la primera.
    2 — solo está en el contenido.
    """
    nombre = f"{nota.titulo} {nota.ruta}".lower()
    buscado = termino.lower()
    if re.search(rf"(?<![0-9a-záéíóúñü]){re.escape(buscado)}(?![0-9a-záéíóúñü])", nombre):
        return 0
    return 1 if buscado in nombre else 2


async def buscar_con_reintentos(
    vault: "Vault", consulta: str, limite: int, carpeta: str | None = None
) -> tuple[list[Nota], str]:
    """Busca en el vault preguntando como se habla, y ordena por lo que importa.

    Dos cosas que se descubrieron mirando el vault de verdad el 2026-08-16, y
    que juntas hacían que la memoria pareciera vacía teniendo la nota delante:

    1. **La búsqueda del plugin no ordena por lo que uno espera.** Buscar `MAGI`
       no devolvía `02_PROYECTOS/MAGI/MAGI.md` entre los cinco primeros: salían
       una conversación y un documento que lo mencionaban de pasada. Como el
       corte es por número de resultados, la nota buena se quedaba fuera antes de
       que nadie la viera.
    2. **Una frase entera devuelve ruido.** "qué pone sobre el proyecto MAGI"
       sacaba PersusWeb y MagicOCR: el plugin encuentra algo para casi cualquier
       cosa, así que "no hay resultados" no es la señal de que la consulta era
       mala.

    Así que se hacen tres cosas aquí, y ninguna en el prompt —porque la voz, el
    panel del PC y el móvil preguntan igual de mal—:

    - se pide **más de lo que se va a enseñar**, para tener qué ordenar;
    - se busca también **palabra por palabra**, y manda la que menos devuelve,
      que es la que más distingue;
    - y **una nota cuyo nombre o ruta contiene lo buscado va primera**, que es
      lo que uno quiere decir cuando pregunta por "el proyecto MAGI".

    Si se pasa `carpeta`, se filtran los resultados a esa ruta (prefijo).
    Devuelve las notas y **qué se buscó de verdad**, para poder decirlo en vez de
    dar a entender que encontró justo lo que se pidió.
    """
    # Se pide **mucho más de lo que se va a enseñar**, y este número es medio
    # arreglo. El plugin no acota: devuelve todo lo que encuentra y el corte lo
    # hacíamos aquí. Buscando `MAGI` en el vault real, la nota
    # `02_PROYECTOS/MAGI/MAGI.md` salía en la **posición 19 de 23**, así que
    # pedir diez la tiraba antes de que nadie pudiera ordenarla.
    ancho = TOPE_CANDIDATAS

    # Cada fuente es una búsqueda: la frase entera y luego cada palabra suelta.
    fuentes: list[tuple[str, list[Nota]]] = [(consulta, await vault.buscar(consulta, ancho))]
    for termino in _terminos(consulta)[:TOPE_TERMINOS]:
        if termino != consulta:
            fuentes.append((termino, await vault.buscar(termino, ancho)))

    # **Manda la que menos devuelve.** Es la que más distingue: preguntando "qué
    # pone sobre el proyecto MAGI", `proyecto` sale en dos plantillas cuyo nombre
    # lo lleva y `MAGI` en la nota que se busca. Por longitud o por orden de
    # aparición ganaba `proyecto`, y la respuesta era sobre una plantilla.
    # La frase entera se queda la última: encuentra de todo y no distingue nada.
    palabras = sorted(fuentes[1:], key=lambda f: len(f[1]))
    ordenadas = palabras + [fuentes[0]] if palabras else fuentes

    mejores: dict[str, tuple[int, int, int, Nota]] = {}
    for rango, (termino, halladas) in enumerate(ordenadas):
        for llegada, nota in enumerate(halladas):
            if carpeta and not nota.ruta.startswith(carpeta.rstrip("/") + "/"):
                continue
            clave = (_prioridad(termino, nota), rango, llegada)
            anterior = mejores.get(nota.ruta)
            if anterior is None or clave < anterior[:3]:
                mejores[nota.ruta] = (*clave, nota)

    usados = [termino for termino, halladas in ordenadas if halladas][:1] or [consulta]
    notas = [c[3] for c in sorted(mejores.values(), key=lambda c: c[:3])][:limite]
    return notas, ", ".join(usados)


async def anotar_persona(antes: str, ahora: str) -> str:
    """Apunta en `Personas/` que un perfil anónimo pasó a tener nombre.

    Lo escribe el reconocimiento de personas cuando el señor Persus renombra un
    «Desconocido N». Es la mitad legible de ese reconocimiento: los vectores de
    `datos/perfiles.json` no le dicen nada a nadie, y esta nota sí — y se puede
    corregir a mano, que es lo que hace que sirva.

    Como todo lo de este módulo, **añade**: si la persona ya tenía nota, el
    apunte se apila debajo y no pisa lo que hubiera escrito antes.
    """
    if _vault is None:
        raise RuntimeError("La memoria no está iniciada; falta memoria.iniciar(cfg).")
    texto = (
        f"Reconocido por voz o cara. Antes figuraba como «{antes}».\n"
        "\nLo que se sepa de esta persona va debajo, escrito a mano o dicho en "
        "llamada: Perseo lo respeta y apila lo nuevo al final."
    )
    return await _vault.anotar(ahora, texto, CARPETA_PERSONAS)


@registrar("memoria")
async def _memoria(trabajo: dict[str, Any]) -> dict[str, Any]:
    """Busca, lee o anota en el vault.

    Tres acciones y ninguna destructiva: `buscar`, `leer` y `anotar`. Borrar y
    reemplazar no están, y no es un olvido — cuando hagan falta pasarán por
    `NecesitaConfirmacion`, como cualquier cosa sin vuelta atrás.
    """
    if _vault is None:
        raise RuntimeError("La memoria no está iniciada; falta llamar a memoria.iniciar().")

    peticion = trabajo.get("peticion") or {}
    accion = str(peticion.get("accion", "buscar")).strip().lower()

    if accion == "buscar":
        consulta = str(peticion.get("texto", "")).strip()
        limite = max(1, min(int(peticion.get("limite", 10)), 50))
        carpeta = str(peticion.get("carpeta", "")).strip() or None
        notas, buscado = await buscar_con_reintentos(_vault, consulta, limite, carpeta)
        return {
            "accion": accion,
            "consulta": consulta,
            # Qué se buscó de verdad. Si la frase entera no dio nada y se
            # troceó, quien lea esto tiene que poder decirlo en vez de dar a
            # entender que encontró justo lo que le pidieron.
            "buscado": buscado,
            "notas": [n.a_dict() for n in notas],
            "titular": f"{len(notas)} nota(s) sobre «{buscado}»" if notas else None,
        }

    if accion == "conversacion":
        # Lo que hacía `guardar_conversacion` en la herramienta de v1: la
        # transcripción entera al vault, donde el indexador la recoge. Va aparte
        # de `anotar` porque el título lo pone la fecha y no el que llama: una
        # conversación no tiene nombre hasta que la lees.
        mensajes = peticion.get("mensajes") or []
        lineas = [
            f"**{str(m.get('tipo', '?'))}:** {str(m.get('texto', '')).strip()}"
            for m in mensajes
            if isinstance(m, dict) and str(m.get("texto", "")).strip()
        ]
        if not lineas:
            return {"accion": accion, "ruta": None, "titular": None}
        titulo = f"Conversacion {datetime.now().strftime('%Y-%m-%d %H%M')}"
        ruta = await _vault.anotar(titulo, "\n\n".join(lineas), CARPETA_CONVERSACIONES)
        return {"accion": accion, "ruta": ruta, "mensajes": len(lineas), "titular": None}

    if accion == "leer":
        ruta = str(peticion.get("ruta", "")).strip()
        return {"accion": accion, "ruta": ruta, "contenido": await _vault.leer(ruta)}

    if accion == "anotar":
        titulo = str(peticion.get("titulo", "")).strip()
        texto = str(peticion.get("texto", "")).strip()
        if not titulo or not texto:
            raise ValueError("Para anotar hacen falta `titulo` y `texto`.")
        carpeta = str(peticion.get("carpeta", CARPETA_MEMORIAS)).strip() or CARPETA_MEMORIAS
        ruta = await _vault.anotar(titulo, texto, carpeta)
        logger.info("Anotado en %s", ruta)
        # El titular sale por Telegram: dice qué nota, no lo que pone dentro.
        return {"accion": accion, "ruta": ruta, "titular": f"Anotado en {ruta}"}

    raise ValueError(
        f"Acción desconocida para la memoria: {accion!r}. "
        "Válidas: buscar, leer, anotar, conversacion."
    )


def _sincrono() -> None:  # pragma: no cover - atajo para la línea de comandos
    """`python -m perseo_core.agentes.memoria`: ¿contesta el plugin, y con la clave buena?

    El equivalente de `python -m perseo_core.servicios.google_api`: comprobar lo que hay
    que configurar a mano sin levantar el núcleo entero.
    """
    import sys

    cfg = almacen.cargar_configuracion()
    if not cfg.vault_rest_clave:
        print(
            "No hay clave del plugin. Se pone en PERSEO_VAULT_CLAVE o en "
            f"{cfg.directorio_datos / 'obsidian.txt'}."
        )
        sys.exit(1)

    rest = VaultRest(cfg.vault_rest_url, cfg.vault_rest_clave)

    async def guion() -> str:
        try:
            return await rest.comprobar()
        finally:
            await rest.cerrar()

    try:
        print(asyncio.run(guion()))
    except RuntimeError as e:
        print(f"No: {e}")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    _sincrono()
