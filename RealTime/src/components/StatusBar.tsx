import React from 'react';

interface Props {
  state: string;
}

export const StatusBar: React.FC<Props> = ({ state }) => {
  return (
    <div className="status-bar">
      <div className={`status-indicator ${state}`} />
      <span>{state.charAt(0).toUpperCase() + state.slice(1)}</span>
    </div>
  );
};
