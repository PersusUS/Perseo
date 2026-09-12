# 0003 · El catálogo de herramientas se declara una vez, con copia de respaldo

**Fecha:** 2026-09-12 · **Estado:** vigente

## Contexto

Las herramientas que ve el modelo estaban escritas dos veces: enteras en
TypeScript para la llamada de voz, y enteras en Python para el chat escrito. Dos
copias de lo mismo se desincronizan, y se habían desincronizado en algo que no
era cosmético:

- El chat anunciaba una acción `navegar_url` **que el agente `pc` no tiene**. El
  modelo podía pedirla, `pc` la rechazaba como desconocida, la política trata lo
  desconocido como irreversible, y el trabajo se quedaba esperando un sí que
  nadie llegaba a ver.
- `usar_mcp` exigía `argumentos` por voz y no por escrito.
- La receta de poner música decía «no pulses Enter» en un sitio y «Enter lanza el
  resultado» en el otro.

## Decisión

Una fuente: `perseo_core/servicios/catalogo.py`.

La **forma** —nombre, parámetros, tipos, `enum`, obligatorios— existe una sola
vez. Los `Parametro` no llevan variante por cara, así que dos listas de acciones
distintas dejan de ser posibles por construcción, no por disciplina.

El **texto** va por cara, y está bien que vaya: por voz la confirmación se pide
hablando y hay que explicarlo; por escrito se pulsa un botón. Las dos versiones
viven una al lado de la otra en la misma ficha, de forma que escribir dos cosas
que se contradicen deja de poder hacerse sin verlas juntas.

## La copia incrustada, que es el precio

La app de voz pide `GET /herramientas?cara=voz` al conectar. Espera **segundo y
medio** y, si el núcleo no contesta, abre la llamada con una copia incrustada
(`RealTime/src/lib/llamada/catalogo-incrustado.ts`).

Esto es deliberado y va contra la regla de «una sola fuente», así que conviene
decir por qué: **abrir la app antes de que el núcleo termine de levantarse es lo
normal**, no un caso raro. Quedarse sin voz porque el catálogo tardó sería mucho
peor que hablar con la copia de ayer. Y el socket no puede esperar: una llamada
que tarda dos segundos en abrirse se vive como una llamada rota.

Lo que impide que la copia envejezca no es esa petición sino
`pruebas/test_catalogo.py`, que la compara con el núcleo carácter a carácter y
pone el CI en rojo si difieren. Regenerarla es un comando:

```bash
python commands/perseo.py catalogo --incrustar
```

## Lo que NO entra en el catálogo

Los niveles de riesgo. Podría parecer que el nivel de cada herramienta cabe en
su ficha, y sería volver a empezar: `infra/politica.py` ya es la única fuente de
qué necesita confirmación. Una herramienta no tiene nivel — lo tiene la acción
que acaba ejecutando, y eso lo decide la política cuando llega el trabajo.

## Consecuencias

- Dos guardias más salieron gratis: el `enum` de acciones de `controlar_pc` se
  compara con lo que el agente `pc` despacha de verdad, y lo que el chat declara
  se compara con lo que sabe ejecutar. Las dos leyendo el árbol del código, no
  una lista al lado.
- `gemini-live.ts` bajó 280 líneas y `chat.py` 260.

## Qué haría falta para cambiarla

Que la app dejara de poder arrancar sin núcleo. Entonces la copia sobra y se
borra, con su prueba.
