"""La bóveda: contraseñas y tarjetas que el modelo nunca ve.

Instinct guarda las sesiones y las claves de sus usuarios en sus servidores, y
parte de lo que se le ha reprochado viene de ahí. Aquí la bóveda es un fichero
en `<datos>/boveda.json`, fuera de git, con cada valor cifrado por **DPAPI**, el
cifrado de Windows atado a la cuenta del usuario: copiar el fichero a otra
máquina, u otra cuenta leyéndolo, da bytes que no se descifran.

**El modelo trabaja con referencias, nunca con valores.** Escribe
`{{boveda:resy.clave}}` donde teclearía la contraseña, y el valor de verdad se
pone en el último momento, en el núcleo, justo antes de que llegue al navegador.
Y todo lo que vuelve del navegador —la instantánea de la página, que puede
llevar lo tecleado dentro de un campo— pasa por `tapar` antes de llegar al
modelo o a un registro: el valor se cambia otra vez por su referencia.

**Cada entrada dice en qué sitios vale.** Es la regla que para el robo más
sencillo: una página hostil que convence al modelo de teclear
`{{boveda:gmail.clave}}` en su propio formulario. La referencia solo se sustituye
si la página abierta es de uno de los dominios de la entrada, o un subdominio;
en cualquier otro sitio falla ruidosa, y el modelo se entera de por qué.

Lo que se guarda en claro es lo que hace falta para decidir sin descifrar: el
nombre de cada entrada, su tipo, sus sitios y cómo se llaman sus campos. Eso
dice «tiene cuenta en resy.com», que queda en tu disco y no viaja.

Los valores se escriben desde la línea de comandos —`perseo boveda guardar`—,
nunca por el chat ni por la voz: lo que se dice o se escribe ahí pasa por un
modelo en la nube, y es justo lo que la bóveda existe para evitar.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import sys
import threading
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

NOMBRE_FICHERO = "boveda.json"

#: `cuenta` es usuario y contraseña; `tarjeta`, una forma de pago. La
#: diferencia importa: meter una tarjeta en un formulario es dar un dato de
#: pago a un tercero, y eso pide un sí aunque el sitio esté permitido (ver
#: `agentes/recado.py`). Una contraseña en su propio sitio es entrar en casa.
TIPOS = ("cuenta", "tarjeta")

#: `{{boveda:resy.clave}}`. Minúsculas, cifras, guion y guion bajo: lo que un
#: modelo copia bien y ningún texto normal escribe por casualidad.
PATRON = re.compile(r"\{\{\s*boveda:([a-z0-9_-]+)\.([a-z0-9_]+)\s*\}\}")
_NOMBRE = re.compile(r"^[a-z0-9_-]{1,40}$")
_CAMPO = re.compile(r"^[a-z0-9_]{1,30}$")

#: Por debajo de esto no se tapa: un valor de dos letras aparece en cualquier
#: texto y taparlo todo emborrona la página sin proteger nada. Tres deja
#: dentro el CVC de una tarjeta, que es el que más importa de los cortos.
MINIMO_TAPAR = 3

#: Un número de tarjeta aparece en las páginas partido en grupos. Desde esta
#: longitud, un valor que sea solo cifras se busca también con espacios o
#: guiones entre ellas.
MINIMO_AGRUPADO = 12

#: Entropía que se añade al cifrar. No es un secreto —está aquí escrita— pero
#: hace que otro programa del mismo usuario que llame a DPAPI a pelo, sin ella,
#: no descifre esto por accidente.
_ENTROPIA = b"perseo-boveda-v1"


class ErrorBoveda(Exception):
    """Algo que la bóveda no hace. El mensaje es para el modelo: dice qué y por qué."""


# --------------------------------------------------------------------------- #
# Cifradores
# --------------------------------------------------------------------------- #


class Cifrador(Protocol):
    def cifrar(self, claro: bytes) -> bytes: ...

    def descifrar(self, cifrado: bytes) -> bytes: ...


class CifradorDpapi:
    """DPAPI por `ctypes`: la llave es la cuenta de Windows, y no hay nada que guardar.

    Sin dependencias: `crypt32.dll` está en cualquier Windows. Fuera de Windows no
    existe, y la bóveda lo dice al construirse en vez de guardar algo en claro.
    """

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise ErrorBoveda(
                "La bóveda cifra con DPAPI, que solo existe en Windows. "
                "Sin cifrado de verdad no se guarda nada."
            )
        import ctypes
        from ctypes import wintypes

        class Blob(ctypes.Structure):
            _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

        self._ctypes = ctypes
        self._Blob = Blob
        self._crypt32 = ctypes.windll.crypt32
        self._kernel32 = ctypes.windll.kernel32
        firma = [
            ctypes.POINTER(Blob),
            wintypes.LPCWSTR,
            ctypes.POINTER(Blob),
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(Blob),
        ]
        self._crypt32.CryptProtectData.argtypes = firma
        self._crypt32.CryptProtectData.restype = wintypes.BOOL
        self._crypt32.CryptUnprotectData.argtypes = [
            ctypes.POINTER(Blob),
            ctypes.c_void_p,
            ctypes.POINTER(Blob),
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(Blob),
        ]
        self._crypt32.CryptUnprotectData.restype = wintypes.BOOL
        self._kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        self._kernel32.LocalFree.restype = ctypes.c_void_p

    def _blob(self, datos: bytes):
        ctypes = self._ctypes
        memoria = ctypes.create_string_buffer(datos, len(datos))
        blob = self._Blob(len(datos), ctypes.cast(memoria, ctypes.POINTER(ctypes.c_char)))
        return blob, memoria  # la memoria se devuelve para que viva lo que viva el blob

    def _llamar(self, funcion, datos: bytes) -> bytes:
        ctypes = self._ctypes
        entrada, _m1 = self._blob(datos)
        entropia, _m2 = self._blob(_ENTROPIA)
        salida = self._Blob()
        # 0x1 = CRYPTPROTECT_UI_FORBIDDEN: el núcleo corre sin ventana, y un
        # diálogo de Windows esperando un clic lo dejaría colgado.
        if not funcion(ctypes.byref(entrada), None, ctypes.byref(entropia), None, None, 0x1, ctypes.byref(salida)):
            raise ErrorBoveda(f"DPAPI falló (error {ctypes.GetLastError()}).")
        try:
            return ctypes.string_at(salida.pbData, salida.cbData)
        finally:
            self._kernel32.LocalFree(ctypes.cast(salida.pbData, ctypes.c_void_p))

    def cifrar(self, claro: bytes) -> bytes:
        return self._llamar(self._crypt32.CryptProtectData, claro)

    def descifrar(self, cifrado: bytes) -> bytes:
        return self._llamar(self._crypt32.CryptUnprotectData, cifrado)


class CifradorDePruebas:
    """NO cifra: da la vuelta a los bytes. Existe para las pruebas en Linux.

    Tiene nombre de lo que es a propósito, y la bóveda de verdad no lo elige
    nunca: hay que pasárselo a mano.
    """

    def cifrar(self, claro: bytes) -> bytes:
        return b"prueba:" + claro[::-1]

    def descifrar(self, cifrado: bytes) -> bytes:
        if not cifrado.startswith(b"prueba:"):
            raise ErrorBoveda("Esto no lo cifró el cifrador de pruebas.")
        return cifrado[len(b"prueba:"):][::-1]


# --------------------------------------------------------------------------- #
# La bóveda
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Uso:
    """Una referencia que se acaba de sustituir: qué entrada, qué campo, de qué tipo."""

    nombre: str
    campo: str
    tipo: str
    limite_euros: float | None


def sitio_permitido(anfitrion: str, sitios: list[str]) -> bool:
    """Si `anfitrion` es uno de `sitios` o un subdominio suyo.

    Por el final y con el punto delante: `cuentas.resy.com` vale para `resy.com`,
    y `resy.com.malo.es` o `noresy.com` no.
    """
    anfitrion = (anfitrion or "").strip().lower().rstrip(".")
    if not anfitrion:
        return False
    for sitio in sitios:
        sitio = sitio.strip().lower().rstrip(".")
        if sitio and (anfitrion == sitio or anfitrion.endswith("." + sitio)):
            return True
    return False


class Boveda:
    def __init__(self, ruta: Path, cifrador: Cifrador | None = None) -> None:
        self._ruta = Path(ruta)
        self._cifrador = cifrador
        self._cerrojo = threading.Lock()

    # -- el fichero ---------------------------------------------------------- #

    def _cifrador_o_error(self) -> Cifrador:
        if self._cifrador is None:
            self._cifrador = CifradorDpapi()
        return self._cifrador

    def _leer(self) -> dict[str, Any]:
        try:
            datos = json.loads(self._ruta.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"version": 1, "entradas": {}}
        except (OSError, json.JSONDecodeError) as e:
            # Ilegible no es «vacía»: devolver vacía y guardar encima perdería
            # todas las entradas. Se para aquí, diciendo dónde mirar.
            raise ErrorBoveda(f"No se puede leer {self._ruta}: {e}") from None
        if not isinstance(datos, dict) or not isinstance(datos.get("entradas"), dict):
            raise ErrorBoveda(f"{self._ruta} no tiene la forma de una bóveda.")
        return datos

    def _escribir(self, datos: dict[str, Any]) -> None:
        self._ruta.parent.mkdir(parents=True, exist_ok=True)
        temporal = self._ruta.with_suffix(".tmp")
        temporal.write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(temporal, self._ruta)

    # -- gestión (la usa la línea de comandos) ------------------------------- #

    def guardar(
        self,
        nombre: str,
        tipo: str,
        sitios: list[str],
        campos: dict[str, str],
        limite_euros: float | None = None,
    ) -> None:
        nombre = nombre.strip().lower()
        if not _NOMBRE.match(nombre):
            raise ErrorBoveda("El nombre va en minúsculas, cifras, guion o guion bajo: «resy», «visa_diaria».")
        if tipo not in TIPOS:
            raise ErrorBoveda(f"Tipo desconocido: {tipo!r}. Válidos: {', '.join(TIPOS)}.")
        sitios = [s.strip().lower() for s in sitios if s.strip()]
        if not sitios:
            # Una entrada sin sitios valdría en cualquier web, que es justo lo
            # que la regla de los sitios existe para impedir.
            raise ErrorBoveda("Hace falta al menos un sitio donde valga la entrada (p. ej. resy.com).")
        if not campos:
            raise ErrorBoveda("Una entrada sin campos no guarda nada.")
        cifrador = self._cifrador_o_error()
        cifrados: dict[str, str] = {}
        for campo, valor in campos.items():
            if not _CAMPO.match(campo):
                raise ErrorBoveda(f"Nombre de campo no válido: {campo!r}.")
            cifrados[campo] = base64.b64encode(cifrador.cifrar(valor.encode("utf-8"))).decode("ascii")
        with self._cerrojo:
            datos = self._leer()
            datos["entradas"][nombre] = {
                "tipo": tipo,
                "sitios": sitios,
                "limite_euros": limite_euros,
                "campos": cifrados,
            }
            self._escribir(datos)
        logger.info("Bóveda: guardada la entrada %r (%s, %s).", nombre, tipo, ", ".join(sitios))

    def borrar(self, nombre: str) -> bool:
        with self._cerrojo:
            datos = self._leer()
            if datos["entradas"].pop(nombre.strip().lower(), None) is None:
                return False
            self._escribir(datos)
        return True

    # -- lo que puede ver el modelo ------------------------------------------ #

    def listar(self) -> list[dict[str, Any]]:
        """Las entradas sin un solo valor: nombre, tipo, sitios y nombres de campo."""
        entradas = self._leer()["entradas"]
        return [
            {
                "nombre": nombre,
                "tipo": entrada.get("tipo"),
                "sitios": list(entrada.get("sitios") or []),
                "campos": sorted((entrada.get("campos") or {}).keys()),
                "limite_euros": entrada.get("limite_euros"),
                "referencias": [
                    f"{{{{boveda:{nombre}.{campo}}}}}" for campo in sorted((entrada.get("campos") or {}).keys())
                ],
            }
            for nombre, entrada in sorted(entradas.items())
        ]

    # -- lo que usa el agente ------------------------------------------------ #

    @staticmethod
    def tiene_referencias(texto: str) -> bool:
        return bool(PATRON.search(texto or ""))

    def usos(self, texto: str) -> list[Uso]:
        """Qué entradas nombra un texto, sin descifrar nada. Falla si alguna no existe."""
        entradas = self._leer()["entradas"]
        vistos: list[Uso] = []
        for nombre, campo in PATRON.findall(texto or ""):
            entrada = entradas.get(nombre)
            if entrada is None:
                raise ErrorBoveda(f"No hay ninguna entrada «{nombre}» en la bóveda.")
            if campo not in (entrada.get("campos") or {}):
                raise ErrorBoveda(f"La entrada «{nombre}» no tiene el campo «{campo}».")
            uso = Uso(nombre, campo, str(entrada.get("tipo")), entrada.get("limite_euros"))
            if uso not in vistos:
                vistos.append(uso)
        return vistos

    def sustituir(self, texto: str, anfitrion: str) -> str:
        """Cambia cada referencia por su valor, si la página es de un sitio de la entrada."""
        if not self.tiene_referencias(texto):
            return texto
        entradas = self._leer()["entradas"]
        cifrador = self._cifrador_o_error()

        def poner(encontrado: re.Match[str]) -> str:
            nombre, campo = encontrado.group(1), encontrado.group(2)
            entrada = entradas.get(nombre)
            if entrada is None or campo not in (entrada.get("campos") or {}):
                raise ErrorBoveda(f"«{nombre}.{campo}» no está en la bóveda.")
            sitios = list(entrada.get("sitios") or [])
            if not sitio_permitido(anfitrion, sitios):
                logger.warning(
                    "Bóveda: «%s» pedida en %r, que no es de sus sitios (%s). No se sustituye.",
                    nombre, anfitrion, ", ".join(sitios),
                )
                raise ErrorBoveda(
                    f"«{nombre}» solo vale en {', '.join(sitios)}, y la página abierta es de "
                    f"{anfitrion or 'ningún sitio conocido'}. No se teclea."
                )
            return cifrador.descifrar(base64.b64decode(entrada["campos"][campo])).decode("utf-8")

        return PATRON.sub(poner, texto)

    def tapar(self, texto: str) -> str:
        """Cambia cualquier valor de la bóveda que aparezca en `texto` por su referencia."""
        if not texto:
            return texto
        try:
            entradas = self._leer()["entradas"]
        except ErrorBoveda:
            return texto
        if not entradas:
            return texto
        cifrador = self._cifrador_o_error()
        valores: list[tuple[str, str]] = []
        for nombre, entrada in entradas.items():
            for campo, cifrado in (entrada.get("campos") or {}).items():
                try:
                    valor = cifrador.descifrar(base64.b64decode(cifrado)).decode("utf-8")
                except (ErrorBoveda, ValueError) as e:
                    logger.warning("Bóveda: no se pudo descifrar %s.%s para taparlo: %s", nombre, campo, e)
                    continue
                if len(valor) >= MINIMO_TAPAR:
                    referencia = f"{{{{boveda:{nombre}.{campo}}}}}"
                    # También como viaja en una URL: un formulario que manda por
                    # GET deja la clave en la dirección de la página siguiente.
                    for forma in {valor, urllib.parse.quote(valor, safe=""), urllib.parse.quote_plus(valor)}:
                        valores.append((forma, referencia))
        # Los largos primero: si una clave contiene a otra, se tapa la entera.
        for valor, referencia in sorted(valores, key=lambda par: -len(par[0])):
            if valor.isdigit() and len(valor) >= MINIMO_AGRUPADO:
                agrupado = r"[\s-]?".join(re.escape(c) for c in valor)
                texto = re.sub(agrupado, referencia, texto)
            else:
                texto = texto.replace(valor, referencia)
        return texto


# --------------------------------------------------------------------------- #
# La del proceso
# --------------------------------------------------------------------------- #

_boveda: Boveda | None = None


def iniciar(directorio_datos: Path | str, cifrador: Cifrador | None = None) -> Boveda:
    global _boveda
    _boveda = Boveda(Path(directorio_datos) / NOMBRE_FICHERO, cifrador)
    return _boveda


def actual() -> Boveda | None:
    return _boveda
