/**
 * El marco de la ventana: la barra de arriba y el acceso a los proyectos.
 *
 * La ventana va **sin decoración de Windows** (`decorations: false` en
 * `tauri.conf.json`): la barra gris del sistema, con su tipografía y sus
 * esquinas redondeadas, era lo único de la pantalla que no seguía el estilo del
 * puesto de mando. A cambio, minimizar, pantalla completa y cerrar hay que
 * ponerlos aquí — y arrastrar la ventana también, que es lo que hace el
 * `data-tauri-drag-region` del riel.
 *
 * **Cerrar esconde, no mata.** Es la misma regla que ya tenía la ventana desde
 * que Perseo vive en la bandeja (`src-tauri/src/lib.rs`): el proceso sigue vivo
 * para que la palabra clave tenga a quien llamar. Por eso el botón dice
 * «esconder» al pasar por encima y no tiene el rojo de un cierre de verdad.
 *
 * Los botones son de texto, en versalitas espaciadas, y no círculos de cristal:
 * los redondos con desenfoque eran de otra interfaz —la de antes de T-11— y
 * junto a las lecturas monoespaciadas cantaban.
 */
import React, { useCallback, useEffect, useState } from 'react';
import { getCurrentWindow } from '@tauri-apps/api/window';

interface Props {
  onPanel: () => void;
  onHabitos: () => void;
  onTareas: () => void;
  onAjustes: () => void;
  onProyectos: () => void;
  proyectosAbiertos: boolean;
}

const IconMinimizar = () => (
  <svg viewBox="0 0 16 16" aria-hidden><line x1="3" y1="8" x2="13" y2="8" /></svg>
);

const IconPantallaCompleta = () => (
  <svg viewBox="0 0 16 16" aria-hidden>
    <path d="M3 6V3h3" /><path d="M13 6V3h-3" />
    <path d="M3 10v3h3" /><path d="M13 10v3h-3" />
  </svg>
);

const IconVentana = () => (
  <svg viewBox="0 0 16 16" aria-hidden><rect x="3.5" y="3.5" width="9" height="9" /></svg>
);

const IconEsconder = () => (
  <svg viewBox="0 0 16 16" aria-hidden><line x1="4" y1="4" x2="12" y2="12" /><line x1="12" y1="4" x2="4" y2="12" /></svg>
);

/** La ventana, o `null` si esto no corre dentro de Tauri.
 *
 *  `getCurrentWindow()` **lanza** cuando no hay Tauri detrás —en el navegador,
 *  que es donde se prueban los estilos con `npm run dev`—, y sin este envoltorio
 *  ese fallo tumbaba la aplicación entera y dejaba la página en blanco. Los
 *  botones de la ventana no funcionan ahí, que es lo correcto: no hay ventana
 *  que minimizar. */
function ventanaActual() {
  try {
    return getCurrentWindow();
  } catch {
    return null;
  }
}

export const Marco: React.FC<Props> = ({
  onPanel, onHabitos, onTareas, onAjustes, onProyectos, proyectosAbiertos,
}) => {
  const [completa, setCompleta] = useState(true);

  // Se pregunta al arrancar en vez de suponerlo: la ventana nace en pantalla
  // completa por configuración, pero eso puede cambiar y el icono tiene que
  // decir la verdad.
  useEffect(() => {
    ventanaActual()?.isFullscreen().then(setCompleta).catch(() => {});
  }, []);

  const alternarCompleta = useCallback(async () => {
    const ventana = ventanaActual();
    if (!ventana) return;
    try {
      const ahora = await ventana.isFullscreen();
      await ventana.setFullscreen(!ahora);
      setCompleta(!ahora);
    } catch {
      // Sin permiso: no hay nada que hacer y menos que decir.
    }
  }, []);

  return (
    <>
      {/* El riel de arriba. Ocupa el ancho entero porque es la zona de
          arrastre: una franja estrecha en una esquina se falla con el ratón. */}
      <div className="marco-riel" data-tauri-drag-region>
        <div className="marco-acciones">
          <button className="marco-boton" onClick={onPanel}>Panel</button>
          {/* Va pegado al Panel y no al final del riel: las dos son pantallas
              que tapan la llamada, y Ajustes es otra cosa —un cajón de
              preferencias—. Agruparlas por lo que hacen y no por cuándo se
              añadieron. */}
          <button className="marco-boton" onClick={onHabitos}>Hábitos</button>
          {/* El corcho va pegado a Hábitos por lo mismo: las dos son pantallas
              suyas —lo que él lleva a mano— y no ventanas al núcleo. */}
          <button className="marco-boton" onClick={onTareas}>Tareas</button>
          <button className="marco-boton" onClick={onAjustes}>Ajustes</button>
        </div>

        <div className="marco-ventana">
          <button
            className="marco-icono"
            title="Minimizar"
            onClick={() => ventanaActual()?.minimize().catch(() => {})}
          >
            <IconMinimizar />
          </button>
          <button
            className="marco-icono"
            title={completa ? 'Salir de pantalla completa' : 'Pantalla completa'}
            onClick={alternarCompleta}
          >
            {completa ? <IconVentana /> : <IconPantallaCompleta />}
          </button>
          <button
            className="marco-icono"
            title="Esconder en la bandeja — Perseo sigue escuchando"
            onClick={() => ventanaActual()?.close().catch(() => {})}
          >
            <IconEsconder />
          </button>
        </div>
      </div>

      {/* Los proyectos. Antes eran un cohete redondo entre los controles de la
          llamada, donde parecía otro botón de la llamada y no lo es: esto no
          toca la voz, abre trabajo. Ahora es una pestaña abajo a la izquierda,
          con el mismo aire que las lecturas, y la tira sigue saliendo donde
          salía. */}
      <button
        className={`marco-pestana ${proyectosAbiertos ? 'abierta' : ''}`}
        onClick={onProyectos}
      >
        <span className="marco-pestana-marca" />
        Proyectos
      </button>
    </>
  );
};
