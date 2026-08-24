import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

/**
 * La configuración de la maqueta del panel (`maqueta/`). Cambia dos cosas de la
 * de siempre: la raíz —para que el `index.html` que se sirva sea el de la
 * maqueta— y el `@tauri-apps/api/core`, que aquí es un `invoke` de mentira
 * porque en un navegador no hay Rust con quien hablar.
 *
 * No afecta a la app: `vite.config.ts` sigue igual y construye lo de siempre.
 */
export default defineConfig({
  plugins: [react()],
  root: resolve(__dirname, "maqueta"),
  resolve: {
    alias: {
      "@tauri-apps/api/core": resolve(__dirname, "maqueta/tauri.ts"),
    },
  },
  // La marca de construcción la inyecta la configuración de siempre; aquí es
  // «maqueta» para que se vea de un vistazo que esto no es la app.
  define: { __PERSEO_BUILD__: JSON.stringify("maqueta") },
  server: { port: 1421, strictPort: true },
});
