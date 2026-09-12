"""Arnés compartido por los scripts de verificación.

Levanta el núcleo de verdad como proceso hijo, sobre un directorio de datos
temporal y un puerto que pide el sistema. Nada se simula: lo que se comprueba
es el proceso que se va a ejecutar en producción.

Vive aquí y no dentro de cada script porque son dos —y serán más— y la parte de
arrancar y parar un proceso en Windows es justo la que no conviene tener
duplicada.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

RAIZ = Path(__file__).resolve().parent.parent

#: Fallos acumulados por `comprobar`. Los scripts lo miran al final.
fallos: list[str] = []


def comprobar(nombre: str, condicion: bool, detalle: str = "") -> None:
    marca = "OK  " if condicion else "FALLO"
    print(f"[{marca}] {nombre}" + (f" -- {detalle}" if detalle else ""))
    if not condicion:
        fallos.append(nombre)


def resumir() -> None:
    """Imprime el recuento y sale con código 1 si algo falló."""
    print()
    if fallos:
        print(f"FALLOS: {len(fallos)} -> {fallos}")
        raise SystemExit(1)
    print("Todo en verde.")


def puerto_libre() -> int:
    """Pide al sistema un puerto sin usar.

    Con un puerto fijo, una ejecución anterior mal cerrada hace fallar la
    siguiente con OSError 10048, que no se parece en nada al problema real.
    """
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ManejadorFalso(BaseHTTPRequestHandler):
    """Un manejador de peticiones que no ensucia la salida del script.

    `BaseHTTPRequestHandler` escribe una línea por petición en stderr, y un
    verificador que levanta un servicio de mentira acaba enterrando sus propios
    [OK] debajo del registro de acceso de un servidor que no existe.
    """

    def log_message(self, *_: Any) -> None:
        pass

    def handle_error(self, *_: Any) -> None:
        pass


class ServidorFalso:
    """Un servicio de fuera, de mentira, en un puerto que da el sistema.

    Cuatro de los verificadores necesitan lo mismo —Telegram, el plugin de
    Obsidian, Google y un sitio web—: un `ThreadingHTTPServer` en un puerto
    libre, un hilo que lo sirve, una URL para dársela al núcleo y una forma de
    pararlo. Eso estaba escrito cuatro veces, palabra por palabra, mientras el
    encabezado de este fichero decía que aquí vive justo lo que no conviene
    tener duplicado.

    Lo que cambia de uno a otro —qué contesta a cada ruta— se escribe en
    `_manejador`, que es lo único que hay que implementar.
    """

    def __init__(self) -> None:
        self.puerto = puerto_libre()
        self._servidor = ThreadingHTTPServer(("127.0.0.1", self.puerto), self._manejador())
        self._servidor.daemon_threads = True

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.puerto}"

    def arrancar(self) -> None:
        threading.Thread(target=self._servidor.serve_forever, daemon=True).start()

    def parar(self) -> None:
        self._servidor.shutdown()

    def _manejador(self) -> type[BaseHTTPRequestHandler]:
        """La clase que atiende las peticiones. La pone cada servicio falso."""
        raise NotImplementedError


class Nucleo:
    """Un núcleo arrancado de verdad, con su directorio de datos aparte.

    El directorio sobrevive a `parar()` a propósito: es lo que permite volver a
    arrancar sobre el mismo estado y comprobar qué sobrevivió al reinicio.
    """

    def __init__(self, entorno_extra: dict[str, str] | None = None) -> None:
        self.datos = Path(tempfile.mkdtemp(prefix="perseo_prueba_"))
        self.puerto = puerto_libre()
        self.base = f"http://127.0.0.1:{self.puerto}"
        self._entorno_extra = entorno_extra or {}
        self._proceso: subprocess.Popen | None = None
        self.salida: list[str] = []

    # -- ciclo de vida ----------------------------------------------------- #

    def arrancar(self) -> None:
        entorno = {
            **os.environ,
            "PERSEO_CORE_DATOS": str(self.datos),
            "PERSEO_CORE_PUERTO": str(self.puerto),
            "PYTHONIOENCODING": "utf-8",
            **self._entorno_extra,
        }
        self._proceso = subprocess.Popen(
            [sys.executable, "-m", "perseo_core"],
            cwd=RAIZ,
            env=entorno,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.salida = []
        threading.Thread(target=self._leer, daemon=True).start()

        for _ in range(60):
            if self._proceso.poll() is not None:
                self.volcar()
                raise SystemExit("El nucleo murio al arrancar")
            try:
                if self.pedir("/salud")[0] == 200:
                    return
            except Exception:
                pass
            time.sleep(0.25)
        self.volcar()
        raise SystemExit("El nucleo no respondio a /salud")

    def _leer(self) -> None:
        assert self._proceso is not None and self._proceso.stdout is not None
        for linea in self._proceso.stdout:
            self.salida.append(linea)

    def parar(self) -> None:
        if self._proceso is None:
            return
        self._proceso.terminate()
        try:
            self._proceso.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proceso.kill()
            self._proceso.wait(timeout=5)
        self._proceso = None

    def reiniciar(self) -> None:
        self.parar()
        print("\n--- nucleo reiniciado ---\n")
        self.arrancar()

    def limpiar(self) -> None:
        self.parar()
        shutil.rmtree(self.datos, ignore_errors=True)

    def volcar(self) -> None:
        """Imprime la salida del hijo sin morir por la consola cp1252 de Windows."""
        texto = "".join(self.salida)
        sys.stdout.write(texto.encode("ascii", "replace").decode("ascii"))
        sys.stdout.flush()

    # -- acceso ------------------------------------------------------------ #

    @property
    def token(self) -> str:
        return (self.datos / "token.txt").read_text(encoding="utf-8").strip()

    def pedir(
        self,
        ruta: str,
        token: str | None = None,
        metodo: str = "GET",
        cuerpo: dict[str, Any] | None = None,
    ) -> tuple[int, Any]:
        datos = json.dumps(cuerpo).encode() if cuerpo is not None else None
        peticion = urllib.request.Request(self.base + ruta, data=datos, method=metodo)
        if datos:
            peticion.add_header("Content-Type", "application/json")
        if token:
            peticion.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(peticion, timeout=10) as r:
                return r.status, json.loads(r.read().decode() or "{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode() or "{}")

    def esperar_estado(
        self, id_trabajo: int, estados: tuple[str, ...], intentos: int = 40
    ) -> dict[str, Any]:
        """Sondea un trabajo hasta que llega a uno de esos estados, o se rinde."""
        trabajo: dict[str, Any] = {}
        for _ in range(intentos):
            _, trabajo = self.pedir(f"/trabajos/{id_trabajo}", self.token)
            if trabajo.get("estado") in estados:
                return trabajo
            time.sleep(0.25)
        return trabajo


class Escucha:
    """Recoge eventos del flujo SSE en segundo plano."""

    def __init__(self, nucleo: Nucleo, cuantos: int) -> None:
        self._nucleo = nucleo
        self._cuantos = cuantos
        self.eventos: list[dict[str, Any]] = []
        self._hilo = threading.Thread(target=self._escuchar, daemon=True)

    def empezar(self) -> None:
        self._hilo.start()
        time.sleep(0.6)  # margen para que la suscripción esté puesta

    def terminar(self, espera: float = 8) -> list[str]:
        self._hilo.join(timeout=espera)
        return [e.get("tipo", "") for e in self.eventos]

    def _escuchar(self) -> None:
        peticion = urllib.request.Request(self._nucleo.base + "/eventos")
        peticion.add_header("Authorization", f"Bearer {self._nucleo.token}")
        try:
            with urllib.request.urlopen(peticion, timeout=15) as r:
                for linea in r:
                    texto = linea.decode("utf-8", "replace")
                    if not texto.startswith("data: "):
                        continue
                    try:
                        self.eventos.append(json.loads(texto[6:]))
                    except json.JSONDecodeError:
                        continue
                    if len(self.eventos) >= self._cuantos:
                        return
        except Exception:
            pass
