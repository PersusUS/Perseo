/**
 * La maqueta: las pantallas de verdad, con datos de mentira.
 *
 * Existe porque ver un cambio de CSS en la app cuesta dos minutos —hay que
 * reconstruir el binario, la interfaz va incrustada dentro— y eso hace
 * que nadie itere sobre el aspecto. Aquí el cambio se ve al guardar.
 *
 *     node RealTime/node_modules/vite/bin/vite.js --config RealTime/vite.maqueta.config.ts
 *
 * Lo que se ve aquí es EXACTAMENTE el componente que va en la app: el `invoke`
 * de Rust se sustituye por uno de mentira en la configuración de vite, y nada
 * más. Si algo se ve bien aquí y mal en la app, es que la app lleva otra cosa.
 *
 * Qué pantalla se sirve lo decide la almohadilla de la dirección, no un cambio
 * de fichero: el panel en `/`, el corcho de tareas en `#tareas`, los hábitos en
 * `#habitos` y `#habitos-plantilla`
 * —que son la misma pantalla en sus dos aires y por eso están las dos, para
 * poder comprobar de un vistazo que un color nuevo se declaró en los dos
 * bloques de variables y no solo en el negro—.
 */
import React from "react";
import ReactDOM from "react-dom/client";

import { Habitos } from "../src/components/Habitos";
import { Panel } from "../src/components/Panel";
import { Tareas } from "../src/components/Tareas";
import "../src/styles/globals.css";

/** Datos de mentira para los hábitos: un mes a medio hacer.
 *
 * Un almacén vacío enseña la pantalla en su caso más fácil —todo a cero, todas
 * las barras al suelo— que es justo el que no hay que mirar. Este siembra un
 * mes irregular: hábitos que van bien, hábitos abandonados, huecos de fin de
 * semana y rachas de distinta longitud. Solo se siembra si no hay nada guardado,
 * para no pisar lo que uno esté probando a mano.
 */
function sembrar() {
  const CLAVE = "perseo.habitos.v1";
  if (localStorage.getItem(CLAVE)) return;

  const hoy = new Date();
  const k = `${hoy.getFullYear()}-${String(hoy.getMonth() + 1).padStart(2, "0")}`;
  const dias = new Date(hoy.getFullYear(), hoy.getMonth() + 1, 0).getDate();

  const habitos = [
    { id: "h1", nombre: "Levantarse a las 06:00" },
    { id: "h2", nombre: "Meditar" },
    { id: "h3", nombre: "Gimnasio" },
    { id: "h4", nombre: "Ducha fría" },
    { id: "h5", nombre: "Trabajo" },
    { id: "h6", nombre: "Leer 10 páginas" },
    { id: "h7", nombre: "Aprender algo nuevo" },
    { id: "h8", nombre: "Sin azúcar" },
    { id: "h9", nombre: "Sin alcohol" },
    { id: "h10", nombre: "Una hora de redes" },
    { id: "h11", nombre: "Planificar el día" },
    { id: "h12", nombre: "Dormir antes de las 23:00" },
  ];

  // Cada hábito con su propia constancia, del que no falla al que se dejó.
  const constancia = [0.95, 0.4, 0.75, 0.6, 0.98, 0.5, 0.35, 0.8, 0.9, 0.25, 0.7, 0.55];
  const marcas: Record<string, boolean> = {};
  const animo: Record<string, number> = {};
  const motivacion: Record<string, number> = {};

  for (let d = 1; d <= Math.min(dias, hoy.getDate()); d++) {
    const finde = [0, 6].includes(new Date(hoy.getFullYear(), hoy.getMonth(), d).getDay());
    habitos.forEach((h, i) => {
      // Un pseudoazar estable: la misma maqueta enseña siempre el mismo mes, y
      // así un cambio de CSS se compara con el de antes en vez de con otro mes.
      const ruido = ((d * 37 + i * 91) % 100) / 100;
      if (ruido < constancia[i] * (finde ? 0.6 : 1)) marcas[`${h.id}|${d}`] = true;
    });
    animo[d] = 4 + ((d * 13) % 7);
    motivacion[d] = 3 + ((d * 29) % 8);
  }

  localStorage.setItem(CLAVE, JSON.stringify({ habitos, meses: { [k]: { marcas, animo, motivacion } } }));
}

function Maqueta() {
  const [ruta, setRuta] = React.useState(location.hash);
  React.useEffect(() => {
    const cambio = () => setRuta(location.hash);
    addEventListener("hashchange", cambio);
    return () => removeEventListener("hashchange", cambio);
  }, []);

  if (ruta.startsWith("#tareas")) return <Tareas onCerrar={() => {}} />;
  if (ruta.startsWith("#habitos")) {
    sembrar();
    return <Habitos onCerrar={() => {}} estilo={ruta === "#habitos-plantilla" ? "plantilla" : "perseo"} />;
  }
  return <Panel onCerrar={() => {}} />;
}

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <div style={{ position: "fixed", inset: 0, background: "var(--bg)" }}>
      <Maqueta />
    </div>
  </React.StrictMode>,
);
