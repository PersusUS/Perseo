# 0004 · Una copia deliberada: `ejecutable_real`

**Fecha:** 2026-08-28, escrito aquí el 2026-09-12 · **Estado:** vigente

## Contexto

`ejecutable_real` averigua cuál es el binario de verdad detrás de un envoltorio
`.cmd` de npm. Existe por un fallo medido: en Windows, lanzar el `.cmd` pasa por
`cmd.exe`, que **corta la orden en el primer salto de línea**. Al subagente le
llegaba la primera línea de la tarea y nada más.

Está escrita dos veces:

- `perseo_core/agentes/dev_motores.py` — para el agente `dev`.
- `commands/subagentes_mcp.py` — para el servidor MCP de subagentes.

Y hay una segunda copia con el mismo motivo: la lista de herramientas denegadas.

## Decisión

Se quedan las dos copias.

## Por qué

El servidor MCP de subagentes **corre como proceso suelto y no importa el
núcleo**. Lo lanza el fichero `mcp.json` con su propio intérprete, fuera del
paquete, y hacerle importar `perseo_core` significaría atarlo al núcleo entero
—aiohttp incluido— para usar quince líneas.

La alternativa buena sería un tercer paquete compartido. Para dos funciones
pequeñas y estables, un paquete nuevo cuesta más de lo que ahorra: hay que
instalarlo, versionarlo y acordarse de él.

## Cómo se evita que se separen

Esta página. No hay guardia automática, y conviene decirlo en vez de fingir que
la hay: si alguien cambia una, tiene que acordarse de la otra.

Lo que sí hay es la advertencia escrita en los dos sitios, en el docstring de
cada copia, nombrando a la otra. Es lo mínimo: una copia sin aviso es un fallo
esperando; una copia con el aviso al lado es una decisión.

## Qué haría falta para cambiarla

Que aparezca una tercera copia. Dos se sostienen; a la tercera, el paquete
compartido sale a cuenta.
