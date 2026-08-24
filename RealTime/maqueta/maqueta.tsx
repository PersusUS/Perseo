/**
 * La maqueta del panel: el panel de verdad, con datos de mentira.
 *
 * Existe porque ver un cambio de CSS en la app cuesta dos minutos —hay que
 * reconstruir el binario, la interfaz va incrustada dentro (H-55)— y eso hace
 * que nadie itere sobre el aspecto. Aquí el cambio se ve al guardar.
 *
 *     node RealTime/node_modules/vite/bin/vite.js --config RealTime/vite.maqueta.config.ts
 *
 * Lo que se ve aquí es EXACTAMENTE el componente que va en la app: el `invoke`
 * de Rust se sustituye por uno de mentira en la configuración de vite, y nada
 * más. Si algo se ve bien aquí y mal en la app, es que la app lleva otra cosa.
 */
import React from "react";
import ReactDOM from "react-dom/client";

import { Panel } from "../src/components/Panel";
import "../src/styles/globals.css";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <div style={{ position: "fixed", inset: 0, background: "var(--bg)" }}>
      <Panel onCerrar={() => {}} />
    </div>
  </React.StrictMode>,
);
