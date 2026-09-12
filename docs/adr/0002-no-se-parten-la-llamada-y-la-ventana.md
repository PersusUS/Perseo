# 0002 · Por qué `gemini-live.ts` y `App.tsx` siguen pasando del techo

**Fecha:** 2026-09-12 · **Estado:** vigente, y a propósito temporal

## Contexto

Hay un techo de tamaño: blando a 600 líneas, duro a 900, comprobado en
`pruebas/test_arquitectura.py`. Existe porque un fichero de mil quinientas
líneas es donde se esconde lo duplicado: nadie lo lee entero, así que nadie ve
que dos trozos hacen lo mismo.

El día que se puso, ocho ficheros lo pasaban. Seis se partieron:

| Fichero | Se partió en | Por qué se pudo |
|---|---|---|
| `dev.py` 1543 | el agente, los motores, el motor del SDK | eran tres cosas juntas |
| `Panel.tsx` 1479 | comun, piezas, dos pestañas | eran seis pestañas juntas |
| `almacen.py` 1192 | la cola y la configuración | eran dos cosas juntas |
| `mcp.py` 1053 | orquestador, transportes, argumentos | eran tres cosas juntas |
| `chat.py` 1307 | la conversación y las herramientas | eran dos cosas juntas |
| `Habitos.tsx` 1056 | la pantalla y tres piezas que no saben de hábitos | eran dos cosas juntas |

Ninguna de esas seis particiones cambió una línea de comportamiento. Son
movimientos: un `git diff` que solo mueve se lee en un vistazo.

## Decisión

`RealTime/src/lib/llamada/gemini-live.ts` (1163) y `RealTime/src/App.tsx` (1162)
**no se parten hoy**, y se quedan en la lista de excepciones con este documento
detrás.

## Por qué

Los seis de arriba eran varias cosas viviendo juntas. Estos dos son **una sola
cosa**:

- `gemini-live.ts` es una clase, `GeminiLiveClient`. Ya bajó de 1443 cuando el
  catálogo de herramientas salió a `servicios/catalogo.py` (ver
  [0003](0003-el-catalogo-se-declara-una-vez.md)). Lo que queda —el socket, los
  mensajes, la reconexión y el despacho de herramientas— comparte estado privado
  en cada método. Sacar un trozo obliga a abrir ese estado, y una clase con diez
  campos públicos «para que el otro fichero pueda» es peor que una clase larga.
- `App.tsx` es un componente con diecisiete efectos que comparten estado. La
  partición buena está clara y es la que propuso la consultoría: sacar ganchos
  (`useLlamada`, `useIdentidad`, `useConfianza`). Eso **no es mover**: es
  rediseñar cuándo corre cada efecto.

Y los dos están en el camino de la llamada de voz, que es lo único de este
repositorio que no se puede comprobar sin hablar por el micrófono. Las pruebas
de vitest y los verificadores no lo cubren.

Partir por cuota lo que no tiene costura es cómo se rompe algo de verdad, y se
rompería en el sitio donde nadie lo vería hasta la siguiente llamada.

## Consecuencias

- La lista de excepciones **no crece**: la prueba comprueba que ninguno de los
  dos aumenta una sola línea. Si alguien añade algo, el CI se pone rojo y la
  respuesta es sacar otra cosa, no subir el número.
- El trabajo pendiente está escrito: los ganchos de `App.tsx`. Con la app
  delante y una sesión de voz para probar, es media tarde.

## Qué haría falta para cambiarla

Poder probar una llamada de verdad mientras se parte. Cualquier otra cosa es
mover código en el camino crítico a ciegas.
