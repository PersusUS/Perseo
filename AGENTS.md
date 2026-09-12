# Guía para agentes de código

Si eres un agente (o una persona con prisa) y vas a tocar este repositorio,
esto es lo que hay que saber antes de escribir una línea.

## El modelo mental, en treinta segundos

Un **núcleo** Python siempre encendido (`perseo_core/`) con cola, memoria,
política y agentes. Cuatro **caras** que se conectan a él: la app Tauri
(voz y panel), la PWA del móvil, Telegram y la línea de comandos.

**Las caras no piensan.** Una cara encola un trabajo y sondea el resultado;
toda decisión —a qué agente va, si hace falta confirmación, qué se le dice al
modelo— ocurre en el núcleo. Si te ves metiendo lógica de negocio en un
componente de React, estás en el sitio equivocado.

## Dónde está cada cosa

El núcleo va en cinco capas, de abajo arriba, y **nadie importa hacia arriba**:
`dominio/` (los tipos), `infra/` (cola, bus, política, router), `servicios/` (la
maquinaria), `agentes/` (los que atienden un trabajo) y `caras/` (API, Telegram,
la web del móvil). No es una costumbre: lo comprueba `pruebas/test_arquitectura.py`.

| Vas a tocar | Mira primero |
|---|---|
| La cola, el bus, la API | `perseo_core/caras/api.py`, `infra/bus.py`, `infra/almacen.py` |
| A qué agente va cada cosa | `perseo_core/infra/router.py` (el router) |
| Un agente concreto | `perseo_core/agentes/<nombre>.py` — se llaman como el agente |
| Qué necesita confirmación | `perseo_core/infra/politica.py` |
| El prompt compartido | `perseo_core/infra/identidad.py` |
| La llamada de voz | `RealTime/src/lib/llamada/gemini-live.ts` y `RealTime/src/App.tsx` |
| El panel | `RealTime/src/components/Panel.tsx` |
| Los puentes a Rust | `RealTime/src-tauri/src/commands.rs` y `nucleo.rs` |
| La web del móvil | `perseo_core/caras/interfaz/index.html` — un solo fichero, sin build |
| Las piezas de la app | `RealTime/src/lib/` en cuatro carpetas: `audio/`, `llamada/`, `datos/`, `identidad/` |
| Configuración | [`docs/CONFIGURACION.md`](docs/CONFIGURACION.md) |

Hay un **techo de tamaño**: blando a 600 líneas, duro a 900, con una lista de
excepciones en `commands/arquitectura.py` que solo puede encoger. Los que hoy
siguen por encima se leen **por rangos tras un `grep -n`**, no de una sentada.
`python commands/perseo.py comprobar --arquitectura` dice cuáles son.

## Ver lo que has cambiado

| Tocaste | Para que se vea | ¿Basta con `npm run build`? |
|---|---|---|
| `RealTime/src/**` | `python commands/perseo.py actualizar` | **No** — la interfaz va incrustada dentro del binario |
| `perseo_core/*.py`, `commands/*.py` | Reiniciar el núcleo: `perseo parar` y luego `perseo on` | **No** — el proceso viejo se queda con el código viejo |
| `perseo_core/caras/interfaz/index.html` | Recargar el navegador | Sí: el núcleo lo sirve del disco |

Para el **aspecto** del panel no hace falta pagar los dos minutos de
reconstrucción: la maqueta sirve las pantallas de verdad con datos de mentira
y recarga al guardar.

```bash
node RealTime/node_modules/vite/bin/vite.js --config RealTime/vite.maqueta.config.ts
```

Rutas de la maqueta: `/` el panel, `#tareas` el tablero, `#habitos` y
`#habitos-plantilla` los dos aires del seguimiento.

## Antes de dar algo por bueno

```bash
python commands/perseo.py comprobar
```

Eso es pytest, ruff, `tsc`, las pruebas del frontend y `cargo check`, en orden
de coste: lo que tarda segundos primero. Las cinco corren también en CI, en
Linux y en Windows. `--rapido` se salta Rust, que es la que tarda.

Está escrito en **un solo sitio** a propósito. Antes eran cuatro bloques
copiados —aquí, en el README y en el fichero del CI— y los recuentos de pruebas
que llevaban dentro ya no coincidían en ninguno. Los números que cita la
documentación salen ahora de `perseo cuentas`, y `perseo cuentas --arreglar` los
reescribe; hay una prueba que compara.

Si tocaste el comportamiento de verdad —no solo el aspecto— pasa además el
verificador que le toque. Están en `verificadores/`, menos el de la palabra
clave, que vive en `commands/` porque necesita micrófono. Ninguno toca el estado
real: se montan un directorio temporal y servidores de mentira.

## Trampas que cuestan una hora

- **Núcleo zombi.** Si el proceso viejo no llegó a morir, todo lo que pruebes
  va contra código viejo. El núcleo nuevo lo detecta y se retira diciéndolo,
  pero ante errores raros: `perseo parar` **antes** de `perseo on`, y mira
  `<datos>/nucleo.log`.
- **Pruebas asíncronas.** La convención es `asyncio.run(...)` dentro de un
  `def test_` normal. **Nunca `@pytest.mark.asyncio`**: tumba el CI.
- **PowerShell corrompe UTF-8** al hacer `Get-Content` + `Set-Content` sobre
  ficheros con acentos. Edita con herramientas de edición directa o con un
  script de Python que abra en `utf-8` explícito.
- **Finales de línea.** El repositorio mezcla LF y CRLF. Si editas con un
  script, lee y escribe con `newline=''` para no reescribir el fichero entero.
- **El prompt del router pasa por `str.format`.** Una llave suelta en
  `identidad.NUCLEO` lo revienta con un `KeyError` que no menciona ese fichero.
- **El triaje NO lleva el preámbulo de identidad, y está medido.** Con él
  delante, `qwen3:4b` pasó de acertar los cuatro correos de prueba a fallar el
  que importaba. Si «arreglas» esa prueba añadiéndolo, rompes la clasificación
  del correo en silencio.

## Estilo

- **La interfaz**: negro y blanco, monoespaciada en versalitas, esquinas
  rectas, sin fuentes externas (la CSP es `self`). El color solo para puntos de
  estado.
- **Los comentarios explican el porqué, no el qué.** Este repositorio comenta
  mucho y a propósito: qué se midió, qué se rompió antes, por qué no se hizo
  de la otra manera. Si añades código, añade ese porqué.
- **En castellano**, como el resto: nombres de función, variables y
  comentarios.
- **Nada de secretos en el código.** Si algo necesita una clave, se lee del
  entorno o de `<datos>`, y `<datos>` está en el `.gitignore`.

## Las reglas que no dependen de que te acuerdes

Cinco cosas que antes eran costumbre y ahora las comprueba
`pruebas/test_arquitectura.py` y `pruebas/test_documentacion.py`. Si alguna se
pone roja, el arreglo **no** es tocar la prueba:

| Regla | Qué pasa si la rompes | Dónde se afloja |
|---|---|---|
| El núcleo va en capas y nadie importa hacia arriba | rojo, con el importe señalado | el orden de `CAPAS` en `commands/arquitectura.py` |
| Ningún fichero pasa de 900 líneas | rojo, con el fichero y su cuenta | pártelo; la lista de excepciones solo encoge |
| Un componente no llama al núcleo | rojo, con el componente | pídele los datos a un gancho de `lib/datos/` |
| `docs/API.md` describe las rutas que existen, y todas | rojo, con la ruta | documéntala o bórrala |
| Los recuentos que cita el README son los de verdad | rojo, con la cifra | `perseo cuentas --arreglar` |

Las excepciones vivas —dos ficheros grandes y diez componentes— tienen nombre y
apellidos en `commands/arquitectura.py`, y las dos que necesitan explicación la
tienen en [`docs/adr/`](docs/adr/). Una excepción sin porqué no vale: si nadie
sabe explicar por qué algo sigue ahí, la respuesta correcta es quitarlo.

Para ver cómo va todo de un vistazo:

```bash
python commands/perseo.py comprobar --arquitectura
```

## Lo que no hay que hacer

- Meter lógica en una cara.
- Ampliar `PERSEO_DEV_RAIZ` «temporalmente para probar».
- Añadir una ruta nueva a `RUTAS_PUBLICAS` sin pensarlo dos veces: son las que
  responden sin token.
- Clasificar como `libre` algo que escribe fuera del vault.
- Quitar de un prompt la regla de que lo observado no es una instrucción. Hay
  una prueba que lo impide, y está ahí por algo.
