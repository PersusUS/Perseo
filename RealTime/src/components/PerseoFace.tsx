import React from 'react';

interface Props {
  isSpeaking: boolean;
  isListening: boolean;
  isConnecting: boolean;
  volume: number;
}

export const PerseoFace: React.FC<Props> = ({ isSpeaking, isListening, isConnecting, volume }) => {
  const cls = isSpeaking ? 'speaking' 
    : isConnecting ? 'connecting' 
    : isListening ? 'listening' 
    : 'idle';

  const scale = isSpeaking ? 1 + volume * 0.8 : 1; // Oscilación más visible 

  return (
    <div className={`perseo-face ${cls}`} style={{ transform: `scale(${scale})` }}>
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
