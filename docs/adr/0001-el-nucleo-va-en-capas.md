# 0001 · El núcleo va en capas, y la regla es una prueba

**Fecha:** 2026-09-12 · **Estado:** vigente

## Contexto

`perseo_core/` eran cincuenta ficheros planos. Funcionaba, y tenía dentro un
fallo que costaba entenderlo: `servicios/google_api` construía `Evento` y
`Mensaje`, que vivían dentro de los agentes `agenda` y `correo`, y esos dos
agentes necesitaban `google_api`. Python no admite eso al arrancar, así que los
dos lo importaban **dentro de una función**.

El truco funcionaba. Lo que no funcionaba era lo que enseñaba: que el ciclo se
paga con un apaño local en vez de moverse el tipo a donde le toca. Y como nada
lo medía, nadie sabía cuántos apaños así había.

El mismo día se encontraron a mano otros once fallos de la misma familia: dos
copias del catálogo de herramientas que decían cosas distintas, código que no
llamaba nadie, documentación que describía una API inexistente. Todos eran
fallos de **dónde vive cada cosa**, no de programación.

## Decisión

Cinco capas, de abajo arriba, y nadie importa hacia arriba:

| Capa | Qué entra | Puede importar |
|---|---|---|
| `dominio/` | Los tipos y el vocabulario | nada del paquete |
| `infra/` | Cola, bus, disparadores, prompt, política, router | `dominio` |
| `servicios/` | Google, modelo local, biometría, MCP, tareas, hábitos | `dominio`, `infra` |
| `agentes/` | Los siete que atienden un trabajo de la cola | todo lo de abajo |
| `caras/` | API, Telegram, estado, la web del móvil | todo |

Y —esta es la mitad que importa— **la regla es una prueba**, no una costumbre.
`pruebas/test_arquitectura.py` lee el árbol de importaciones con `ast`, sin
ejecutar nada, y falla si aparece un ciclo o un importe hacia arriba.

## Por qué una prueba y no una nota en `AGENTS.md`

Porque ya había una nota en `AGENTS.md`. «Las caras no piensan» lleva escrito
desde el principio, y el día que se midió había diez componentes de React
llamando al núcleo por su cuenta.

Una carpeta bien puesta se deshace en tres meses. Una prueba que falla en el CI,
no.

## Consecuencias

- Un tipo que necesiten dos capas baja a `dominio/`, y eso deja de ser una
  discusión: es lo que hace falta para que la prueba pase.
- `agentes.py` pasó a `infra/router.py`. No era un agente —es el registro y el
  despachador— y ocupaba el nombre de la capa. El repositorio ya lo llamaba «el
  router» en su propia documentación.
- Los verificadores salieron del paquete a `verificadores/`. El paquete que se
  distribuye ya no lleva su banco de pruebas dentro.
- `__main__.py` se queda en la raíz, sin capa. Es la raíz de composición: el
  único sitio que conoce todas las capas a la vez, porque su trabajo es
  montarlas.

## Qué haría falta para cambiarla

Que las capas dejaran de describir el sistema. Si un día `caras/` tuviera que
importarse desde `servicios/` para algo que no es un rodeo, la respuesta no es
quitar la prueba: es que el orden de las capas está mal y hay que reordenarlo en
`commands/arquitectura.py`, donde el orden **es** la regla.
