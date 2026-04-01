import React, { useEffect, useRef } from 'react';

interface Props {
  stream: MediaStream | null;
}

export const CameraPreview: React.FC<Props> = ({ stream }) => {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
    }
  }, [stream]);

  if (!stream) return null;

  return (
    <div style={{ 
      position: 'absolute', 
      top: '20px', 
      right: '20px', 
      width: '160px', 
      height: '120px', 
      borderRadius: '12px', 
      overflow: 'hidden', 
      border: '2px solid var(--border)', 
      boxShadow: '0 8px 24px rgba(0,0,0,0.4)', 
      zIndex: 10,
      background: 'var(--bg-color)'
    }}>
      <video 
        ref={videoRef} 
        autoPlay 
        playsInline 
        muted 
        style={{ width: '100%', height: '100%', objectFit: 'cover' }} 
      />
      <div style={{
          position: 'absolute',
          bottom: 4, right: 4,
          background: 'rgba(0,0,0,0.5)',
          padding: '2px 6px',
          borderRadius: 4,
          fontSize: '10px',
          color: '#fff'
      }}>
          REC 🔴
      </div>
    </div>
  );
};
