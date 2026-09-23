/**
 * Qué hacer cuando llega un aviso de llamada: timbre, contarlo en vivo o guardarlo.
 *
 * El aviso lo deja quien quiere que Perseo llame —el detector de palabra clave
 * sin motivo; los subagentes, los encargos del núcleo y los recordatorios con
 * él— en el marcador que vigila Rust (`autollamada.rs`).
 *
 * Lo que se arregló el 2026-09-23, porque los encargos «funcionaban mal»: un
 * aviso con motivo que llegaba **en plena llamada se tiraba**. La regla de «si
 * ya está en llamada no se hace nada» era para la palabra clave —que sirve para
 * empezar una conversación, no para cortar la que hay— y se estaba aplicando
 * también a «tu encargo ha terminado». Ahora un motivo en plena llamada se le
 * cuenta al modelo para que lo diga; uno que llega mientras conecta, se guarda
 * para contarlo al abrir.
 */

type EstadoLlamada = 'disconnected' | 'connecting' | 'connected' | 'error';

type AccionAviso =
  /** Entrar en llamada sin más: la palabra clave o el aplauso. */
  | 'llamar'
  /** Sonar el timbre con el motivo: si contesta, llamada saliente. */
  | 'timbre'
  /** Ya hay llamada: decírselo al modelo para que lo cuente. */
  | 'contar-en-vivo'
  /** Está conectando: que se cuente en cuanto abra. */
  | 'guardar'
  | 'nada';

/**
 * `recienAbierta` es la ventana de los primeros segundos tras abrir la app: si
 * el vigilante de Rust gana la carrera al marcador del detector, un marcador
 * vacío ahí sería una llamada al abrirse disfrazada de evento.
 */
export function decidirAviso(motivo: string, estado: EstadoLlamada, recienAbierta: boolean): AccionAviso {
  const conMotivo = motivo.trim().length > 0;
  if (estado === 'connected') return conMotivo ? 'contar-en-vivo' : 'nada';
  if (estado === 'connecting') return conMotivo ? 'guardar' : 'nada';
  if (conMotivo) return 'timbre';
  return recienAbierta ? 'nada' : 'llamar';
}

/**
 * Cómo se le cuenta al modelo un aviso que llega en plena llamada.
 *
 * Va con turno pedido, para que lo diga ya y no cuando al señor Persus le
 * toque hablar: un recordatorio de las cinco no vale a las cinco y cuarto. Y
 * marcado como lo que es —un aviso del sistema, no algo que alguien acabe de
 * decir—, para que no lo lea en voz alta tal cual.
 */
export function avisoEnLlamada(motivo: string): string {
  const lineas = motivo
    .split('\n')
    .map(l => l.trim())
    .filter(Boolean);
  return (
    '[AVISO DEL SISTEMA — no lo ha dicho nadie: cuéntaselo ahora, con tus palabras y en cuanto ' +
    'termines la frase, y sin leer esta marca]\n' +
    lineas.map(l => `- ${l}`).join('\n')
  );
}
