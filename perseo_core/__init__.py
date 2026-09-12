"""perseo-core: el núcleo de Perseo v2.

Un solo proceso que decide. Las caras —la app Tauri, la web del móvil, la sesión
de voz— solo transportan entrada y salida; ninguna piensa. Esa separación es lo
que permite mudar el núcleo a la Raspberry Pi más adelante sin reescribir nada.

## Las cinco capas, de abajo arriba

| Capa | Qué vive ahí | Puede importar |
|---|---|---|
| `dominio/` | Los tipos y el vocabulario: `Evento`, `Mensaje`, `Clasificacion`, los niveles de riesgo | nada del paquete |
| `infra/` | La cola, el bus, los disparadores, el prompt, la política, el router | `dominio` |
| `servicios/` | Google, el modelo local, la biometría, MCP, tareas, hábitos, la lista de aplicaciones | `dominio`, `infra` |
| `agentes/` | Los que atienden un trabajo: `correo`, `agenda`, `pc`, `web`, `dev`, `chat`, `memoria` | todo lo de abajo |
| `caras/` | La API, Telegram, la pantalla de estado y la web del móvil | todo |

**Nadie importa hacia arriba**, y eso no es una costumbre: lo comprueba
`pruebas/test_arquitectura.py` leyendo el árbol de importaciones. Antes de las
capas había un ciclo —`google_api` con `agenda` y `correo`— que se pagaba con
dos importes escondidos dentro de funciones. La regla existe para que no vuelva.

`__main__.py` se queda en la raíz a propósito: es la raíz de composición, el
único sitio que conoce todas las capas a la vez porque su trabajo es montarlas.
"""
