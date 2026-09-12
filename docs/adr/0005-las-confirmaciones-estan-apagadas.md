# 0005 · Las confirmaciones están apagadas

**Fecha:** 2026-09-12 · **Estado:** vigente

## Contexto

La política de §7 para lo irreversible y pide un sí. Vive en
`perseo_core/infra/politica.py`, se aplica en un solo sitio —el trabajador— y
tiene cuatro niveles, tabla, modo confianza y regla de visitas. Está bien
construida y sigue entera.

El problema no era la política: era **por dónde le llegaban las preguntas al
dueño**.

Durante una llamada, la confianza se enciende sola y se renueva con su voz en
ventanas de diez minutos (`App.tsx`, `renovarConfianza`). La renovación depende
de que el reconocimiento biométrico le **nombre**, y ese reconocimiento está
sesgado a callar a propósito:

```python
# perseo_core/servicios/biometria.py
#: el umbral se queda en medio, conservador a propósito: mejor un «Desconocido»
UMBRAL_VOZ = 0.55
```

Así que «no sé quién habla» acababa tratándose igual que «no es él». Medido en
la llamada del 2026-09-12 a las 21:31: los tres trabajos que salieron de ella
llegaron al núcleo con `quien = None`, la confianza se encendió una vez y no se
renovó ni una. En una llamada larga eso significa que al minuto diez todo lo
irreversible vuelve a preguntar, aunque el único que haya hablado sea él.

La lección de N-3 decía *«que se apague cuando el que habla deje de ser él»*.
Lo implementado decía *«cuando el micrófono no lo confirme»*. No es lo mismo, y
la diferencia se paga en fricción cada pocos minutos.

## Decisión

El señor Persus las apagó enteras. Con sus palabras: «cualquier usuario puede
decir una tarea y no requerir confirmación no es tan preocupante como para tener
que usarse».

Se apagan con **un interruptor con nombre**, `politica.CONFIRMACIONES`, no
borrando la política.

## Qué se pierde, dicho sin adornos

`CRITICO` también. Borrar ficheros, tocar el registro, matar procesos y
`PowerShell` salen sin preguntar.

Esa es exactamente la puerta por la que el 2026-08-27 salió un `Remove-Item` que
nadie autorizó —el caso que hizo nacer el nivel, contado en la cabecera de
`dominio/niveles.py`—. Queda escrito aquí porque el coste de una decisión se
apunta cuando se toma, no cuando se cobra.

Lo que **no** se pierde: la regla de que lo observado no son órdenes. Esa vive
en el prompt de la voz y en `infra/identidad.py`, es de otra familia, y sigue en
pie.

## Por qué un interruptor y no un borrado

Porque la política no estaba mal, y lo que no está mal no se tira. La tabla de
niveles, el modo confianza, las repeticiones y `pide_confirmacion` siguen
enteros **y siguen probados**: las pruebas del trabajador arman el guardia con
una fixture (`armado`, en `pruebas/test_router.py`) para que el cableado no se
pudra callado, y hay dos pruebas nuevas que fijan lo contrario —que hoy no se
para nada— para que rearmarlo sin querer se ponga rojo.

La separación está en dos funciones:

- `pide_confirmacion(...)` — lo que la política **querría**. Intacta.
- `hay_que_parar(...)` — lo que **pasa**. Es la única que mira el trabajador.

## Qué haría falta para cambiarla

Poner `CONFIRMACIONES = True`, o arrancar con `PERSEO_CONFIRMACIONES=1`. Nada
más: el resto del sistema sigue esperando esa respuesta.

Y si lo que se quiere es el punto medio que esta decisión se saltó —preguntar
solo por lo `CRITICO`, que son tres cosas y en toda la historia de la base de
datos se han pedido dos veces— el sitio es `hay_que_parar`, y son dos líneas.
