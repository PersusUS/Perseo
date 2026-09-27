# La API del núcleo

El núcleo habla HTTP y sirve además la web del móvil. Todo lo que la app de
escritorio hace, se puede hacer con `curl`.

**Base:** `http://127.0.0.1:8787` — configurable con `PERSEO_CORE_HOST` y
`PERSEO_CORE_PUERTO`.

---

## Autenticación

Toda petición necesita el token, salvo las que sirven la propia web —el
navegador no puede mandar una cabecera en la primera carga—.

```bash
TOKEN=$(cat perseo_core/datos/token.txt)
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8787/trabajos
```

Para `EventSource`, que tampoco admite cabeceras, el token se canjea una vez
por una cookie `HttpOnly`:

```bash
curl -X POST -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8787/sesion
```

**Públicas, a propósito:** `/`, `/salud`, `/manifest.webmanifest` y los
iconos. `/salud` no devuelve nada sensible.

> El núcleo **nunca escucha en `0.0.0.0`**. Para llegar desde el móvil se le
> dice explícitamente en qué interfaces atender, y lo normal es que sea la de
> tu VPN.

---

## Lo que de verdad se usa

### Hablar y encolar

| Ruta | Qué hace |
|---|---|
| `POST /mensaje` | Entrada conversacional. **El router decide** si contesta él o encola para un agente |
| `POST /trabajos` | Encola directamente, saltándose el router: `{"agente": "pc", "peticion": {"accion": "abrir_app", "parametro": "notepad"}}`. Admite además `origen` (`voz`, `texto` o `disparador`) y `quien` — el perfil de quien lo pidió, que es lo que mira la política para no dejar que una visita mueva las manos |
| `GET /trabajos` | La cola. Filtros `?estado=` y `?limite=` |
| `GET /trabajos/{id}` | Uno |
| `GET /trabajos/{id}/actividad` | Por dónde va un encargo largo, paso a paso |
| `POST /trabajos/{id}/cancelar` | Lo cierra si sigue abierto |

### Dar el sí

| Ruta | Qué hace |
|---|---|
| `POST /trabajos/{id}/aprobar` | Da el sí a un trabajo parado; vuelve a la cola |
| `POST /trabajos/{id}/rechazar` | Lo cierra sin ejecutarlo |
| `GET /confianza` | Si el modo confianza está encendido y hasta cuándo |
| `POST /confianza` | Lo enciende (`{"minutos": 60}`) o lo apaga (`{"activo": false}`) |

### Mirar

| Ruta | Qué hace |
|---|---|
| `GET /salud` | Que está vivo, y qué agentes carga |
| `GET /estado` | De qué está capado el sistema hoy: piezas, cuota, disparadores, **la máquina** (CPU, RAM, disco, red, batería) y **la presencia**. Con token: junta, esa información es el mapa de por dónde entrar |
| `GET /eventos` | Flujo SSE con todo lo que pasa |
| `POST /herramientas/{nombre}` | Ejecuta una herramienta del catálogo como la voz: `{"argumentos": {...}, "quien": "Persus"}`. Es por donde la app ejecuta las que Rust no conoce ([ADR 0008](adr/0008-la-voz-despacha-por-el-nucleo.md)); lo que encola lleva origen `voz` y la voz de quien habló |
| `GET /herramientas` | El catálogo de herramientas de una cara (`?cara=voz` o `?cara=chat`): nombre, descripción y esquema de cada una. Está declarado **una sola vez** en el núcleo; la app de voz lo pide al conectar y lleva una copia incrustada por si el núcleo tarda |

### El correo triado

| Ruta | Qué hace |
|---|---|
| `GET /correos` | Qué se ha hecho con cada correo. Lo que no salga está pendiente |
| `POST /correos/{id}/estado` | `atendido`, `descartado` o `pendiente`. **No toca Gmail**: anota lo que hiciste tú |

### El chat escrito

| Ruta | Qué hace |
|---|---|
| `GET /chat` · `GET /chat/{id}` | Las conversaciones y sus mensajes |
| `POST /chat` | Empieza una |
| `POST /chat/{id}/hablar` | Dice algo en ella |
| `DELETE /chat/{id}` | La borra |

### Tareas, hábitos y proyectos

| Ruta | Qué hace |
|---|---|
| `GET /tareas` · `POST /tareas` | El tablero de notas |
| `POST /tareas/recoger` | Vacía la papelera |
| `GET /habitos` · `POST /habitos` | El seguimiento del mes |
| `GET /proyectos` | Los otros programas que se pueden abrir |
| `POST /proyectos/{id}/abrir` | Abre uno. Por aquí viaja **cuál**, nunca qué ejecutar |
| `GET /grafo` · `GET /grafo/datos` | El grafo del vault |
| `POST /grafo/abrir` | Abre una nota en Obsidian. Por aquí viaja **cuál**, y el id se busca entre los ficheros reales del vault |

### El hilo y la ubicación

| Ruta | Qué hace |
|---|---|
| `GET /hilo` | La conversación principal —la que comparten el panel, el móvil, Telegram, WhatsApp, el teléfono y lo hablado en llamadas— y lo último que hay en ella |
| `POST /ubicacion` | Guarda dónde está él: `{"latitud": 37.38, "longitud": -5.98, "precision": 20}`. Pensada para un atajo del iPhone. Solo se guarda la última |
| `GET /ubicacion` · `DELETE /ubicacion` | La última compartida, o la olvida |

### Biometría

Solo responden con el reconocimiento encendido. Los vectores no salen nunca:
por aquí van audio o imagen, y vuelve un nombre.

| Ruta | Qué hace |
|---|---|
| `GET /biometria` | Perfiles conocidos, qué se está aprendiendo y qué motores hay |
| `POST /biometria/voz` · `POST /biometria/cara` | Manda un trozo y devuelve a quién se parece |
| `POST /biometria/perfiles` | Crea uno |
| `POST /biometria/perfiles/{nombre}` | Lo renombra |
| `DELETE /biometria/perfiles/{nombre}` | Lo borra, y borra sus números de verdad |

---

## Lo que la API **no** hace

No hay un `POST /ejecutar` ni un `POST /shell`.

Por HTTP se encola **un trabajo para un agente**, y qué puede hacer ese agente
lo decide el núcleo con su lista blanca y su política de niveles. Quien
consiga el token consigue encolar, no consigue una consola.

Desde el 2026-09-25 se expone además **el catálogo de herramientas**
(`POST /herramientas/{nombre}`), porque la voz ejecuta por ahí las que Rust no
conoce. Tampoco es una consola: cada herramienta del catálogo lee, o encola un
trabajo con la política de siempre (lo exterior sigue esperando el sí en la
tarjeta, y aprobarlo no se puede desde aquí). Ver el
[ADR 0008](adr/0008-la-voz-despacha-por-el-nucleo.md).

---

## Un ejemplo entero

```bash
TOKEN=$(cat perseo_core/datos/token.txt)
BASE=http://127.0.0.1:8787

# Encola algo irreversible: teclear va a ciegas sobre la ventana con el foco
curl -s -X POST $BASE/trabajos \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"agente":"pc","peticion":{"accion":"escribir_teclado","parametro":"hola"}}'

# Se queda esperando
curl -s "$BASE/trabajos?estado=esperando" -H "Authorization: Bearer $TOKEN"

# Dale el sí
curl -s -X POST $BASE/trabajos/1/aprobar -H "Authorization: Bearer $TOKEN"
```
