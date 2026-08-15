"""Agente `memoria`: leer y escribir el vault sin romperlo.

La estructura del vault no se toca — es R7 del plan, y hay años de notas dentro.
Lo que hace este módulo es envolverla: buscar, leer y **añadir**, que es todo lo
que hace falta para que Perseo recuerde cosas entre conversaciones.

**El vault también es un puerto.** Hoy detrás hay ficheros, y el plan (§6) dice
que mañana habrá el plugin Local REST API de Obsidian, que da parcheo quirúrgico
—apuntar a un encabezado y tocar solo esa parte— y que además evita escribir por
debajo de una aplicación que puede tener el fichero abierto. Cambiar de uno a
otro es escribir una clase, igual que con el buzón del correo.

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
import logging
import os
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

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
        momento = datetime.now().strftime("%Y-%m-%d %H:%M")

        if destino.exists():
            # Añadir, nunca reemplazar. Ver la regla 1 de la cabecera.
            with destino.open("a", encoding="utf-8") as nota:
                nota.write(f"\n\n## {momento}\n\n{texto.strip()}\n")
        else:
            destino.write_text(
                "---\n"
                "tipo: memoria\n"
                f"titulo: {titulo}\n"
                f"fecha_creacion: {momento}\n"
                "tags: [perseo]\n"
                "---\n"
                f"# {titulo}\n\n"
                f"## {momento}\n\n"
                f"{texto.strip()}\n",
                encoding="utf-8",
            )
        return destino.relative_to(self.raiz).as_posix()


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
        raiz = ruta_vault(cfg)
        raiz.mkdir(parents=True, exist_ok=True)
        _vault = VaultFicheros(raiz)
        logger.info("Memoria sobre el vault en %s", raiz)
    return _vault


def detener() -> None:
    global _vault
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
