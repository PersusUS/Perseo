/**
 * La corteza de las ventanas de proyecto y del grafo.
 *
 * Las ventanas que Perseo abre de otras apps — Armario, CVScraper, MAGI, el
 * grafo del vault — cargan páginas remotas, y a una página remota no se le
 * puede colgar una barra propia: lo intentado y desmontado el 2026-08-24 (ni
 * el puente IPC llega hasta ella ni hay forma honesta de pulsar un botón).
 * Así que ESTA ventana carga una página nuestra —esta misma, que es mínima—
 * y la app del proyecto va dentro de un iframe.
 *
 * Lo que gana el señor Persus con el rodeo:
 *
 *   - La barra gris de Windows desaparece: arriba manda el riel de la casa,
 *     que se arrastra (`data-tauri-drag-region`) y lleva minimizar, pantalla
 *     completa y cerrar.
 *   - Todo en blanco y negro, como mandó el señor Persus al ver los colores
 *     por proyecto («déjalo de nuevo en blanco y negro»). El mecanismo queda:
 *     si un proyecto declara `color` en `proyectos.json`, una línea fina bajo
 *     la barra lo recoge — sin declararlo, monocromo de la casa.
 *   - Cerrar cierra DE VERDAD. No como la ventana principal, que se esconde en
 *     la bandeja porque la palabra clave necesita a quien llamar: una ventana
 *     de proyecto no es Perseo, y nadie espera que su X deje un proceso vivo.
 *
 * Los parámetros no viajan en la URL de la ventana: los guarda Rust por
 * etiqueta y esta página se los pregunta al arrancar (`corteza_parametros`).
 * Es lo que mantiene el token del grafo fuera de `location`, donde cualquier
 * JS remoto podría leerlo.
 */
import React, { useCallback, useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { getCurrentWindow } from '@tauri-apps/api/window';

type DatosCorteza = {
  url: string;
  titulo: string;
  /** #rrggbb o cadena vacía = monocromo de la casa. */
  color: string;
};

/** La ventana de esta página, o `null` fuera de Tauri (pruebas en navegador).
 *  Mismo envoltorio defensivo que `Marco.tsx`: sin él, el fallo tumba todo. */
function ventanaActual() {
  try {
    return getCurrentWindow();
  } catch {
    return null;
  }
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

const IconCerrar = () => (
  <svg viewBox="0 0 16 16" aria-hidden><line x1="4" y1="4" x2="12" y2="12" /><line x1="12" y1="4" x2="4" y2="12" /></svg>
);

export const Corteza: React.FC = () => {
  const [datos, setDatos] = useState<DatosCorteza | null>(null);
  const [fallo, setFallo] = useState('');
  // Como en Marco.tsx: se pregunta, no se supone — el icono tiene que decir
  // la verdad aunque la ventana nazca, se maximice o vuelva por su cuenta.
  const [completa, setCompleta] = useState(false);

  useEffect(() => {
    invoke<DatosCorteza | null>('corteza_parametros')
      .then(d => {
        if (d) setDatos(d);
        else setFallo('Esta ventana no tiene ficha asignada.');
      })
      .catch(e => setFallo(`No se pudo preguntar a Rust: ${e}`));
  }, []);

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
      // Sin permiso no hay nada que hacer ni nada que contar.
    }
  }, []);

  // El color solo se usa si llega limpio; Rust ya lo valida, pero esto es lo
  // que evita pintar basura si algún día cambia el contrato.
  const color =
    datos && /^#[0-9a-fA-F]{6}$/.test(datos.color.trim())
      ? datos.color.trim()
      : '';

  // Al cargar la app, el foco pasa al iframe: si no, teclear dentro del
  // proyecto no hacía nada hasta pulsarlo con el ratón.
  const alCargar = useCallback((e: React.SyntheticEvent<HTMLIFrameElement>) => {
    try {
      (e.currentTarget.contentWindow as Window | null)?.focus();
    } catch {
      // Cruzado y sin permiso: el clic manual siempre queda.
    }
  }, []);

  return (
    <div className={`corteza${completa ? ' completa' : ''}`}>
      {/* El riel de arriba. Ancho entero por lo mismo que el de la ventana
          principal: la zona de arrastre estrecha se falla con el ratón. */}
      <div className="corteza-riel" data-tauri-drag-region style={color ? { '--color-proyecto': color } as React.CSSProperties : undefined}>
        <span className="corteza-marca" />
        <span className="corteza-titulo">{datos?.titulo ?? ''}</span>

        <div className="corteza-botones">
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
            title="Cerrar — esta ventana sí se cierra de verdad"
            onClick={() => ventanaActual()?.close().catch(() => {})}
          >
            <IconCerrar />
          </button>
        </div>

        {/* La línea de identidad del proyecto, bajo la barra entera. Va como
            elemento y no como border-bottom para poder medir 2 px exactos. */}
        <span className="corteza-linea" />
      </div>

      {fallo ? (
        <div className="corteza-aviso">{fallo}</div>
      ) : !datos ? (
        <div className="corteza-aviso">Preparando…</div>
      ) : (
        <iframe
          className="corteza-marco"
          src={datos.url}
          title={datos.titulo}
          onLoad={alCargar}
        />
      )}
    </div>
  );
};

export default Corteza;
