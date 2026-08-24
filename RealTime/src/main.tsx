import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { Corteza } from "./components/Corteza";
import { getCurrentWindow } from "@tauri-apps/api/window";
import "./styles/globals.css";

/**
 * Esta misma build viste dos casas: la ventana de Perseo (App, la llamada) y
 * las cortezas — las ventanas de proyecto y del grafo, que solo tienen barra
 * e iframe (ver components/Corteza.tsx). Quién es quién lo dice la ETIQUETA
 * de la ventana: `main` es Perseo; `proyecto-*` y `grafo-*` son cortezas. La
 * etiqueta la inyecta Tauri de forma síncrona en páginas locales, así que el
 * reparto no espera a ninguna llamada. Fuera de Tauri (npm run dev en el
 * navegador) no hay etiqueta y se cae al lado de siempre.
 */
function queVentanaSoy(): "corteza" | "perseo" {
  try {
    const etiqueta = getCurrentWindow().label;
    return etiqueta.startsWith("proyecto-") || etiqueta.startsWith("grafo-")
      ? "corteza"
      : "perseo";
  } catch {
    return "perseo";
  }
}

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    {queVentanaSoy() === "corteza" ? <Corteza /> : <App />}
  </React.StrictMode>,
);
