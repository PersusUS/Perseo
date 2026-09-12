# Decisiones de arquitectura

Una página por decisión **que alguien podría querer deshacer**. No es
documentación del sistema —eso está en el resto de `docs/`— sino la respuesta a
«¿por qué está esto así?» cuando la respuesta no cabe en un comentario y la
pregunta va a volver.

La regla para escribir una: si alguien que llega de fuera miraría el código y
pensaría *«esto está mal, lo arreglo»*, y arreglarlo rompería algo, hace falta
una página aquí. Si el porqué cabe en tres líneas junto al código, va junto al
código: este repositorio comenta mucho y a propósito, y una carpeta de
decisiones no es una excusa para dejar de hacerlo.

Viven en `docs/` y no en la bitácora por un motivo concreto: la bitácora no se
versiona —es el cuaderno interno, con rutas de esta máquina y decisiones a medio
cocer— y una decisión que solo existe en un disco no gobierna nada.

| # | Decisión |
|---|---|
| [0001](0001-el-nucleo-va-en-capas.md) | El núcleo va en capas, y la regla es una prueba |
| [0002](0002-no-se-parten-la-llamada-y-la-ventana.md) | Por qué `gemini-live.ts` y `App.tsx` siguen pasando del techo |
| [0003](0003-el-catalogo-se-declara-una-vez.md) | El catálogo de herramientas se declara una vez, con copia de respaldo |
| [0004](0004-una-copia-deliberada-de-ejecutable-real.md) | Una copia deliberada: `ejecutable_real` |

## El formato

Corto. Contexto, decisión, consecuencias, y qué haría falta para cambiarla. Una
página que no se lee entera de una sentada no la va a leer nadie.
