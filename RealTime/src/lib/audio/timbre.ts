/**
 * El timbre de llamada entrante. Sintetizado con WebAudio y no un fichero de
 * audio: la CSP del binario solo deja `self`, y un WAV incrustado es un
 * recurso más que mantener. Dos tonos alternos, suaves, en bucle hasta que
 * alguien atienda o mande callar.
 *
 * El AudioContext necesita un gesto previo del usuario para sonar; si el
 * navegador lo suspende, se reintenta al arrancar cada ciclo — por eso el
 * `resume()` dentro del intervalo.
 */

let contexto: AudioContext | null = null;
let temporizador: number | null = null;

function pitido(frecuencia: number, cuando: number) {
  if (!contexto) return;
  const oscilador = contexto.createOscillator();
  const ganancia = contexto.createGain();
  oscilador.type = 'sine';
  oscilador.frequency.value = frecuencia;
  ganancia.gain.setValueAtTime(0.0001, contexto.currentTime + cuando);
  ganancia.gain.exponentialRampToValueAtTime(0.14, contexto.currentTime + cuando + 0.03);
  ganancia.gain.exponentialRampToValueAtTime(0.0001, contexto.currentTime + cuando + 0.32);
  oscilador.connect(ganancia).connect(contexto.destination);
  oscilador.start(contexto.currentTime + cuando);
  oscilador.stop(contexto.currentTime + cuando + 0.35);
}

/** Empieza a sonar. Idempotente: dos avisos seguidos no duplican el bucle. */
export function sonar(): void {
  if (temporizador !== null) return;
  contexto = contexto ?? new AudioContext();
  const ciclo = () => {
    void contexto?.resume().catch(() => {});
    pitido(932, 0);
    pitido(699, 0.38);
  };
  ciclo();
  temporizador = window.setInterval(ciclo, 1250);
}

/** Calla. */
export function callar(): void {
  if (temporizador !== null) {
    clearInterval(temporizador);
    temporizador = null;
  }
}
