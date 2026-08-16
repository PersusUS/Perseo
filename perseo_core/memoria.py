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
que fue H-22 y costó descubrirlo.

Ver bitacora/05_PLAN_PERSEO_V2.md §6 y §9 (Fase D).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import unicodedata
import urllib.parse
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Protocol

import aiohttp

from . import almacen
from .agentes import registrar

logger = logging.getLogger(__name__)

#: Carpetas del vault. Son las que ya usaba la herramienta de memoria de v1: la
#: estructura del vault no se toca, se envuelve (R7).
CARPETA_MEMORIAS = "Memorias_Sistema"
CARPETA_CONVERSACIONES = "Conversaciones"

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

    def __init__(self, base: str, clave: str, tope: float = TOPE_REST) -> None:
        self.base = base.rstrip("/")
        self._clave = clave
        self._tope = tope
        self._http: aiohttp.ClientSession | None = None

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
            raise RuntimeError(
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

    async def buscar(self, consulta: str, limite: int = 10) -> list[Nota]:
        consulta = consulta.strip()
        if not consulta:
            return []
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
        estado, cuerpo = await self._pedir(
            "GET", self._url("/vault/", relativa), cabeceras={"Accept": "text/markdown"}
        )
        if estado == 404:
            raise FileNotFoundError(f"No hay ninguna nota en {ruta!r}")
        self._comprobar(estado, cuerpo)
        return cuerpo

    # -- escritura ---------------------------------------------------------- #

    async def anotar(self, titulo: str, texto: str, carpeta: str = CARPETA_MEMORIAS) -> str:
        nombre = _nombre_seguro(titulo)
        if not nombre:
            raise ValueError("El título no deja ningún nombre de fichero utilizable.")

        relativa = _ruta_relativa(f"{carpeta}/{nombre}.md")
        url = self._url("/vault/", relativa)
        momento = _momento()

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
    """Dónde está el vault. Una sola variable para todo el sistema, ver H-22."""
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
            return VaultRest(cfg.vault_rest_url, cfg.vault_rest_clave)
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
        notas = await _vault.buscar(consulta, limite)
        return {
            "accion": accion,
            "consulta": consulta,
            "notas": [n.a_dict() for n in notas],
            "titular": f"{len(notas)} nota(s) sobre «{consulta}»" if notas else None,
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
    """`python -m perseo_core.memoria`: ¿contesta el plugin, y con la clave buena?

    El equivalente de `python -m perseo_core.google_api`: comprobar lo que hay
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
