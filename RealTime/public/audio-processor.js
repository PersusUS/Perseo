/**
 * El micrófono, convertido a PCM de 16 bits para la sesión de voz.
 *
 * El worklet se despierta cada cuanto de render —128 muestras, que a 16 kHz son
 * 8 ms—, y mandar cada cuanto por su cuenta salían 125 mensajes por segundo:
 * 125 copias al hilo principal, 125 conversiones a base64 y 125 escrituras en
 * el socket, cada una con su cabecera. Con el hilo principal ocupado
 * (transcripción, cara, capturas) esa cola se acumulaba y el modelo recibía la
 * voz tarde, que desde fuera se ve como que Perseo tarda en enterarse.
 *
 * Se juntan 1024 muestras —64 ms— antes de mandar: ocho veces menos mensajes,
 * y 64 ms de retraso que no se notan al lado de lo que costaba la cola.
 */

/** Muestras por envío. 1024 a 16 kHz = 64 ms. */
const MUESTRAS_POR_ENVIO = 1024;

class AudioCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.acumulado = new Int16Array(MUESTRAS_POR_ENVIO);
    this.escritas = 0;
  }

  process(inputs) {
    const input = inputs[0][0]; // mono channel
    if (input) {
      for (let i = 0; i < input.length; i++) {
        // Convert Float32 [-1, 1] to Int16 PCM
        this.acumulado[this.escritas++] = Math.max(-32768, Math.min(32767, input[i] * 32768));

        if (this.escritas === MUESTRAS_POR_ENVIO) {
          // Se manda una copia y se sigue con el mismo hueco: transferir el
          // buffer lo dejaría desprendido y habría que crear uno nuevo cada
          // vez, que es basura para el recolector 15 veces por segundo.
          const trozo = this.acumulado.slice();
          this.port.postMessage(trozo.buffer, [trozo.buffer]);
          this.escritas = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor('audio-capture', AudioCaptureProcessor);
