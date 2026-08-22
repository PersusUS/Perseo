import React from 'react';
import type { AspectoLive } from '../lib/config';

interface Props {
  isSpeaking: boolean;
  isListening: boolean;
  isConnecting: boolean;
  /** Qué se dibuja alrededor de la cara. Ver components/Escenografia.tsx. */
  aspecto?: AspectoLive;
}

/**
 * Los anillos que rodean la cara (T-11).
 *
 * Tres velocidades y dos sentidos: es lo que hace que parezcan un instrumento y
 * no una animación pegada encima. Se paran cuando no hay llamada, porque un
 * instrumento que gira solo miente sobre lo que está pasando.
 *
 * El de dentro es el anillo de volumen: su trazo crece con la voz de Perseo,
 * así que también dice de un vistazo que el audio está llegando.
 */
const Anillos: React.FC = () => (
  <>
    <div className="anillo g1">
      <svg viewBox="0 0 340 340" aria-hidden>
        <circle className="anillo-traz" cx="170" cy="170" r="168" strokeWidth="1" strokeDasharray="2 10" opacity=".5" />
      </svg>
    </div>
    <div className="anillo g2">
      <svg viewBox="0 0 340 340" aria-hidden>
        <g className="anillo-traz" strokeWidth="1.2" opacity=".7">
          <path d="M170 24 a146 146 0 0 1 126 73" />
          <path d="M296 243 a146 146 0 0 1 -126 73" />
          <path d="M44 243 a146 146 0 0 1 0 -146" />
        </g>
      </svg>
    </div>
    <div className="anillo g3">
      <svg viewBox="0 0 340 340" aria-hidden>
        <g className="anillo-traz" strokeWidth="2" opacity=".55">
          <path d="M170 46 l0 12" />
          <path d="M170 282 l0 12" />
          <path d="M46 170 l12 0" />
          <path d="M282 170 l12 0" />
        </g>
      </svg>
    </div>
    <div className="anillo-vol">
      <svg viewBox="0 0 340 340" aria-hidden>
        <circle cx="170" cy="170" r="85" />
      </svg>
    </div>
  </>
);

export const PerseoFace: React.FC<Props> = ({ isSpeaking, isListening, isConnecting, aspecto = 'mira' }) => {
  const cls = isSpeaking ? 'speaking'
    : isConnecting ? 'connecting'
    : isListening ? 'listening'
    : 'idle';

  // La escala la pone el CSS a partir de `--vol`, que ya viene suavizada
  // (App.tsx). Antes se calculaba aquí, con la medida cruda y multiplicada por
  // 0,8: la cara llegaba a hincharse casi al doble y temblaba con cada sílaba.
  //
  // Va en la foto y no en el contenedor: si la llevara el contenedor, los
  // anillos —que cuelgan de él— se hincharían igual y dejarían de parecer un
  // instrumento fijo alrededor de la cara.
  return (
    <div className={`perseo-face ${cls}`}>
      {aspecto === 'cartel' ? (
        // En «cartel» la cara no lleva anillos: lleva la esquina cortada y dos
        // marcas. Ahí el movimiento lo pone la cartela, no el instrumental.
        <>
          <span className="cara-esquina e1" />
          <span className="cara-esquina e2" />
        </>
      ) : (
        <Anillos />
      )}
      <img src="/perseo-avatar.jpg" alt="Perseo" className="perseo-avatar-img" draggable={false} />
      {/* Concentric ripples when speaking */}
      {isSpeaking && (
        <>
          <div className="ripple r1" />
          <div className="ripple r2" />
          <div className="ripple r3" />
        </>
      )}
    </div>
  );
};
