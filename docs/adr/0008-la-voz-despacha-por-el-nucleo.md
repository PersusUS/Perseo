# 0008 · La voz despacha por el núcleo lo que Rust no conoce

**Fecha:** 2026-09-25 · **Estado:** vigente

## Contexto

La app de voz ejecuta las herramientas que pide el modelo desde Rust
(`nucleo.rs`): traduce cada una a un trabajo para un agente y lo encola. Las
que no tenían lógica propia iban en una tabla, `DIRECTAS`, a una línea por
herramienta. El fichero vive pegado a su techo de 900 líneas, y en dos días de
Instinct entraron nueve herramientas: cada una costaba una línea de Rust y, sobre
todo, una **reconstrucción de la app** de veinticinco minutos para que la voz la
viera.

Mientras tanto, el chat escrito ya sabía despachar todas esas herramientas desde
Python (`agentes/chat_herramientas.py`), con el mismo catálogo.

## Decisión

Lo que Rust no conoce se lo manda al núcleo: `POST /herramientas/{nombre}`, que
corre el mismo despacho que el chat escrito. La tabla `DIRECTAS` desaparece.

Dos cosas viajan con la herramienta y no se pierden por el camino: **la puerta**
(`voz`) y **quién habla** (el perfil del reconocimiento de voz). El despacho las
lleva en variables de contexto y las pone en cada trabajo que encola, así que
la política ve lo mismo que cuando encolaba Rust: una orden de una visita sigue
siendo de una visita.

Rust se queda con lo suyo: lo que tiene lógica de la cara (aprobar por voz, la
situación del momento, el PC) y lo que ya estaba escrito con sus alias viejos.
Lo que atiende la propia app sin salir de ella (la pantalla, las personas, el
tablero y los hábitos) sigue en TypeScript. Una prueba comprueba que ninguna
herramienta de voz del catálogo se queda sin quien la atienda
(`pruebas/test_voz_por_el_nucleo.py`).

## Consecuencias

- **Una herramienta nueva se declara en el catálogo y ya existe en las dos
  caras**, sin tocar Rust ni reconstruir la app: basta con reiniciar el núcleo.
- La API expone el catálogo por HTTP. No es una consola: cada herramienta lee
  o encola un trabajo con la política de siempre, y la ruta pide token como
  todas. `docs/API.md` lo dice en «Lo que la API no hace».
- La voz depende un poco más del núcleo: si está caído, esas herramientas
  fallan. Ya fallaban igual, porque lo que hacían era encolar en él.

## Qué haría falta para cambiarla

Volver a una tabla en Rust. No se recomienda: era la parte que costaba una
reconstrucción por herramienta, y la que mantenía dos despachos que podían
decir cosas distintas del mismo trabajo.
