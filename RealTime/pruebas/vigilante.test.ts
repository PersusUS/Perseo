/**
 * El vigilante de identidad, con el núcleo simulado.
 *
 * Fija los dos arreglos que más importaban para no llamar «señor Persus» a
 * quien no es: cuando el núcleo duda, la etiqueta anterior se retira y se avisa
 * de la duda —antes se quedaba puesta y el modelo seguía creyendo que hablaba
 * él—; y mientras suena Perseo no se manda nada, para que su propia voz no
 * acabe tomada por la de alguien.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';

const invocar = vi.fn();
vi.mock('@tauri-apps/api/core', () => ({ invoke: (...args: unknown[]) => invocar(...args) }));

const { vigilante } = await import('../src/lib/identidad/identidad');

const TROZO = 1024;

/** Tres segundos de voz en trozos de 64 ms, como los manda el worklet. */
function tresSegundosDeVoz(): ArrayBuffer[] {
  const trozos: ArrayBuffer[] = [];
  for (let k = 0; k < 47; k++) {
    const trozo = new Int16Array(TROZO);
    for (let i = 0; i < TROZO; i++) {
      const n = k * TROZO + i;
      const envolvente = 0.2 + 0.8 * Math.abs(Math.sin((2 * Math.PI * 3 * n) / 16000));
      trozo[i] = Math.round(8000 * envolvente * Math.sin(n * 0.06));
    }
    trozos.push(trozo.buffer);
  }
  return trozos;
}

async function hablar(): Promise<void> {
  for (const trozo of tresSegundosDeVoz()) vigilante.consumirAudio(trozo);
  // Deja que la petición al núcleo simulado conteste.
  await new Promise(r => setTimeout(r, 0));
  await new Promise(r => setTimeout(r, 0));
}

describe('VigilanteIdentidad', () => {
  let hablantes: (string | null)[];
  let dudas: number;

  beforeEach(() => {
    invocar.mockReset();
    hablantes = [];
    dudas = 0;
    // El vigilante es uno para toda la app: se vacía lo que dejó la prueba
    // anterior, incluida la cola del eco de 300 ms.
    vigilante.desactivar();
    (vigilante as unknown as { ecoHastaMs: number }).ecoHastaMs = 0;
    vigilante.activa = true;
    vigilante.hablanteActual = null;
    vigilante.hablaPerseo = () => false;
    vigilante.onHablante = nombre => hablantes.push(nombre);
    vigilante.onDuda = () => { dudas++; };
  });

  it('cuando el núcleo duda, deja de decir que habla Persus y avisa', async () => {
    invocar.mockResolvedValueOnce({ nombre: 'Persus', confianza: 0.8 });
    await hablar();
    expect(hablantes).toEqual(['Persus']);

    invocar.mockResolvedValueOnce({ nombre: null, dudoso: 'Persus', confianza: 0.45 });
    await hablar();
    expect(hablantes).toEqual(['Persus', null]);
    expect(vigilante.hablanteActual).toBeNull();
    expect(dudas).toBe(1);
  });

  it('un desconocido llega con su etiqueta provisional', async () => {
    invocar.mockResolvedValueOnce({ nombre: 'Desconocido 1', provisional: true });
    await hablar();
    expect(hablantes).toEqual(['Desconocido 1']);
  });

  it('mientras suena Perseo no se manda nada al núcleo', async () => {
    vigilante.hablaPerseo = () => true;
    await hablar();
    expect(invocar).not.toHaveBeenCalled();
  });

  it('lo que viaja es la ventana entera, no picos sueltos', async () => {
    invocar.mockResolvedValueOnce({ nombre: null, voz: 0 });
    await hablar();
    expect(invocar).toHaveBeenCalledTimes(1);
    const [comando, { audio }] = invocar.mock.calls[0] as [string, { audio: string }];
    expect(comando).toBe('biometria_voz');
    // 3 s de PCM de 16 bits son ~96 kB; en base64, unos 128 kB.
    expect(audio.length).toBeGreaterThan(120_000);
  });
});
