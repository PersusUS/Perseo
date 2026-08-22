import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// @ts-expect-error process is a nodejs global
const host = process.env.TAURI_DEV_HOST;

// https://vite.dev/config/
export default defineConfig(async () => ({
  plugins: [react()],

  // La marca de construcción, incrustada en el binario. La pone `perseo
  // actualizar`, que sella con el mismo valor `<datos>/version.json` para el
  // móvil; sin variable —o sea, en `npm run dev`— vale 'dev'. Ver
  // src/lib/version.ts.
  define: {
    // @ts-expect-error process es global de node
    __PERSEO_BUILD__: JSON.stringify(process.env.PERSEO_BUILD || 'dev'),
  },

  // Vite options tailored for Tauri development and only applied in `tauri dev` or `tauri build`
  //
  // 1. prevent Vite from obscuring rust errors
  clearScreen: false,
  // 2. tauri expects a fixed port, fail if that port is not available
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host
      ? {
          protocol: "ws",
          host,
          port: 1421,
        }
      : undefined,
    watch: {
      // 3. tell Vite to ignore watching `src-tauri`
      ignored: ["**/src-tauri/**"],
    },
  },
}));
