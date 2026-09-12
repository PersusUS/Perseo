# 0006 · Dos pantallas para el mismo panel

**Fecha:** decidido en su día, escrito aquí el 2026-09-12 · **Estado:** vigente

## Contexto

El panel —cola, correo, memoria y estado— está escrito dos veces:

- `perseo_core/caras/interfaz/index.html` — la PWA. Un solo fichero sin build,
  servido por el núcleo, que se abre en el móvil por la VPN de casa. Se
  autentica con una cookie de sesión.
- `RealTime/src/components/Panel.tsx` — la misma pantalla en React, dentro de la
  ventana de la app de escritorio, hablando con el núcleo **a través de Rust**
  (`src-tauri/src/panel.rs`).

Las mismas cuatro pestañas, el mismo trabajo, dos implementaciones. Es la
duplicación más grande que queda en el repositorio y la que más invita a que
alguien llegue, la vea y la "arregle".

## Por qué no se puede quitar

La solución obvia es hospedar la PWA dentro de la app de escritorio y borrar la
otra. No funciona, y no por código nuestro:

- **En una ventana aparte**, la CSP de la aplicación bloquea el `<script>` en
  línea de `index.html`. La página sale en blanco.
- **En un `<iframe>`**, la cookie de sesión es `SameSite=Strict` y el navegador
  no la manda desde un contexto embebido. No hay forma de autenticarse.

Las dos son políticas del navegador, no defectos que se parcheen. Aflojarlas
—CSP permisiva, cookie `SameSite=Lax`— es pagar con seguridad real una
duplicación que cuesta bastante menos que eso.

## Lo que se gana, además de la ventana única

En `Panel.tsx` **no se pega ningún token**. Lo lee Rust del disco, igual que
para las herramientas de voz. La PWA necesita su cookie porque vive en un
navegador de verdad; la app de escritorio no necesita ninguna credencial en el
lado de JavaScript, y esa asimetría es una ventaja que se perdería al unificar.

## Decisión

Se quedan las dos. **La del móvil manda**: es la que se usa a diario, y cuando
las dos discrepan, la que está bien es `index.html`.

## Qué las mantiene juntas hoy

Poco, y conviene decirlo en vez de fingir lo contrario.

Lo único automático es `pruebas/test_dev.py::test_las_cuatro_listas_de_modelos_dicen_lo_mismo`,
que comprueba que la lista de modelos de subagentes es la misma —y en el mismo
orden— en el MCP, en `index.html` y en `AgentesTab.tsx`. Nació porque esa lista
ya se desincronizó una vez.

El resto es disciplina: lo que se cambie en una pantalla hay que llevarlo a la
otra, y está escrito en el docstring de `Panel.tsx`, que es donde lo va a leer
quien la toque.

## Qué haría falta para cambiarla

Cualquiera de estas tres:

1. Que Tauri permita servir la PWA con la cookie del núcleo intacta —mismo
   origen efectivo— sin aflojar la CSP.
2. Que la PWA deje de depender de un `<script>` en línea, lo que significa darle
   un build; y un build para la cara de repuesto es justo la complejidad que esa
   cara existe para no tener (ver la excepción de tamaño en
   `commands/arquitectura.py`).
3. Que aparezca una tercera pantalla. Dos se sostienen con disciplina; a la
   tercera, el coste de mantenerlas iguales supera al de inventar el camino
   común.

Mientras tanto, lo que sí se puede hacer sin tocar nada de esto es **subir lo
automático**: cada vez que una discrepancia entre las dos cause un fallo real,
la respuesta correcta es una prueba que las compare, como la de los modelos, y
no un recordatorio más en un comentario.
