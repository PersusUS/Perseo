# Perseo — memoria para sesiones

Lee `bitacora/06_HANDOFF.md` antes de tocar nada: manda lo último (§ numeradas).

## Ver los resultados (lo que el usuario VE)

| Tocaste | Para que se vea | Basta `npm run build` / pytest |
|---|---|---|
| `RealTime/src/**` | `python commands/perseo.py actualizar` | NO — la interfaz va incrustada en el binario (H-55) |
| `perseo_core/*.py`, `commands/*.py` | Reiniciar el núcleo (`perseo parar` + `perseo on`; el vigilante lo revive con el código nuevo) | NO |

`perseo actualizar` hace TODO: cierra la app, `tauri build --no-bundle`,
sella la marca en `perseo_core/datos/version.json`, vacía la caché de
WebView2 (trampa §6.16) y reabre la app. Tarda ~2 min.

Para el ASPECTO del panel no hace falta pagar esos dos minutos por vuelta:
la maqueta sirve el panel de verdad con datos de mentira y recarga al
guardar.

```
node RealTime/node_modules/vite/bin/vite.js --config RealTime/vite.maqueta.config.ts
```

## Verificaciones antes de dar algo por bueno

- Python: `python -m pytest` (convención: `asyncio.run(...)` dentro de
  `def test_` normal — NUNCA `@pytest.mark.asyncio`, tumba el CI, §6.11).
- Frontend (`RealTime/`): `npx tsc --noEmit` y `npm test` (Vitest).
- Rust: `cargo check --locked`.
- CI: `gh run list --limit 3` — dos veces estuvo rojo sin saberlo (H-40).

## Trampas que cuestan una hora

- **Núcleo zombi tras reiniciar**: si el viejo (`pythonw -m perseo_core`)
  no llegó a morir, TODO lo que pruebes va contra código viejo. Desde el
  2026-08-24 el núcleo nuevo lo detecta —pregunta por `/salud` antes de
  tocar nada— y se retira diciéndolo (H-66), en vez del bucle silencioso de
  `OSError 10048`. Ante errores raros tras tocar el núcleo: `nucleo.log` +
  `Get-NetTCPConnection -LocalPort 8787`, y `perseo parar` ANTES de `on`.
- **PowerShell corrompe UTF-8** al hacer round-trip `Get-Content` +
  `Set-Content` en ficheros con acentos (§6.13): ediciones SIEMPRE
  quirúrgicas directas o script Python.
- Los hijos spawned no heredan stdin del canal JSON-RPC ni entorno completo.
- El testigo de Google caduca cada 7 días: si el triaje muere,
  `python -m perseo_core.autorizar_google`.

## Estilo

- Negro y blanco, monoespaciada en versalitas, esquinas rectas, sin fuentes
  externas (CSP `self`). El color solo para puntos de estado (§4.8).
- Las caras no piensan: la vista encola y sondea; lo irreversible pide su sí
  por la política (§7 del plan).
