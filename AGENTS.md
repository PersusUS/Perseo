# Perseo — guía para agentes

Asistente personal: un núcleo Python siempre encendido (`perseo_core/`) y cuatro
caras que no piensan (voz, panel Tauri, PWA, Telegram). **Las caras no piensan** es
la regla que no se rompe.

## Lo único obligatorio antes de tocar nada

Lee `bitacora/ESTADO.md` (12 KB). Nada más. Todo lo demás se abre **por secciones y
bajo demanda** — abrir un documento entero de esta bitácora cuesta más tokens que el
arreglo que ibas a hacer.

## Qué abrir según lo que vayas a hacer

| Necesitas | Abre | Cómo |
|---|---|---|
| El estado de hoy | `bitacora/ESTADO.md` | Entero (12 KB) |
| Saber qué hace un fichero | `bitacora/01_INVENTARIO.md` | Entero (14 KB): fichero a fichero, con LOC y estado |
| Arreglar un fallo conocido | `bitacora/02_HALLAZGOS.md` | Tabla resumen arriba; luego `grep -n "H-42 ·"` |
| Arrancar, construir, desplegar | `bitacora/06_HANDOFF.md` §3 | `sed -n` de la sección |
| Saber por qué algo es así | `bitacora/06_HANDOFF.md` §4 (decisiones) y §5 (restricciones) | Por sección |
| **Antes de tocar núcleo, build o tests** | `bitacora/06_HANDOFF.md` §6 — las 18 trampas | Por sección |
| Qué pasó un día concreto | `bitacora/11_HISTORIA.md`, `bitacora/04_SESIONES.md` | **Solo `grep -n`**, nunca enteros |
| Los subagentes de código | `bitacora/10_SUBAGENTES.md` | Entero (8 KB) |
| La PWA del móvil | `bitacora/07_PWA.md` | Entero (12 KB) |

Presupuesto de lectura, para elegir con conocimiento: `11_HISTORIA.md` 117 KB ·
`02_HALLAZGOS.md` 119 KB · `04_SESIONES.md` 85 KB · `06_HANDOFF.md` 48 KB ·
`05_PLAN_PERSEO_V2.md` 26 KB · `03_ROADMAP.md` 20 KB. Los tres primeros **no se
leen enteros nunca**.

Lo mismo con el código: `01_INVENTARIO.md` dice qué hay en cada fichero antes de
abrirlo. Los que pasan de 1.000 líneas —`Panel.tsx`, `dev.py`, `chat.py`,
`almacen.py`, `gemini-live.ts`, `mcp.py`, `api.py`— se leen por rangos tras un
`grep -n`, no de una sentada.

## Ver los resultados (lo que el usuario VE)

| Tocaste | Para que se vea | Basta `npm run build` / pytest |
|---|---|---|
| `RealTime/src/**` | `python commands/perseo.py actualizar` | NO — la interfaz va incrustada en el binario (H-55) |
| `perseo_core/*.py`, `commands/*.py` | Reiniciar el núcleo (`perseo parar` + `perseo on`; el vigilante lo revive con el código nuevo) | NO |

`perseo actualizar` hace TODO: cierra la app, `tauri build --no-bundle`, sella la
marca en `perseo_core/datos/version.json`, vacía la caché de WebView2 (trampa §6.16)
y reabre la app. Tarda ~2 min.

Para el ASPECTO del panel no hace falta pagar esos dos minutos por vuelta: la maqueta
sirve el panel de verdad con datos de mentira y recarga al guardar.

```
node RealTime/node_modules/vite/bin/vite.js --config RealTime/vite.maqueta.config.ts
```

## Verificaciones antes de dar algo por bueno

- Python: `python -m pytest` (convención: `asyncio.run(...)` dentro de un `def test_`
  normal — NUNCA `@pytest.mark.asyncio`, tumba el CI, §6.11).
- Frontend (`RealTime/`): `npx tsc --noEmit` y `npm test` (Vitest).
- Rust: `cargo check --locked`.
- CI: `gh run list --limit 3` — dos veces estuvo rojo sin saberlo (H-40).

## Trampas que cuestan una hora

Las dieciocho están en `bitacora/06_HANDOFF.md` §6. Las cuatro que más se repiten:

- **Núcleo zombi tras reiniciar**: si el viejo (`pythonw -m perseo_core`) no llegó a
  morir, TODO lo que pruebes va contra código viejo. Desde el 2026-08-24 el núcleo
  nuevo lo detecta —pregunta por `/salud`— y se retira diciéndolo (H-66), en vez del
  bucle silencioso de `OSError 10048`. Ante errores raros tras tocar el núcleo:
  `nucleo.log` + `Get-NetTCPConnection -LocalPort 8787`, y `perseo parar` ANTES de `on`.
- **PowerShell corrompe UTF-8** al hacer round-trip `Get-Content` + `Set-Content` en
  ficheros con acentos (§6.13): ediciones SIEMPRE quirúrgicas directas o script Python.
- Los hijos spawned no heredan stdin del canal JSON-RPC ni entorno completo (§6.14).
- **Los modelos gratuitos de los subagentes se retiran sin avisar** (H-75): si un
  encargo del agente `dev` o del MCP de subagentes falla raro, lo primero es
  `opencode models | grep free`, no leer código. El modelo se hereda de
  `~/.config/opencode/opencode.jsonc`; el porqué, en `bitacora/10_SUBAGENTES.md`.

El testigo de Google caduca cada 7 días: si el triaje muere,
`python -m perseo_core.autorizar_google`.

## Estilo

- Negro y blanco, monoespaciada en versalitas, esquinas rectas, sin fuentes externas
  (CSP `self`). El color solo para puntos de estado (§4.8).
- Las caras no piensan: la vista encola y sondea; lo irreversible pide su sí por la
  política (§7 del plan).

## Al terminar

Entrada en `bitacora/04_SESIONES.md`, hallazgo nuevo en `bitacora/02_HALLAZGOS.md`
(tabla arriba **y** ficha), y si cambió el estado del sistema, `bitacora/ESTADO.md`.
La crónica larga va a `bitacora/11_HISTORIA.md`, no al handoff.
