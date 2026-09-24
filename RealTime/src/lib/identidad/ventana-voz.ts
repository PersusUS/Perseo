/**
 * Qué trozos de micrófono merecen viajar al núcleo para saber quién habla.
 *
 * **Esto no decide qué es voz.** Eso lo decide el núcleo
 * (`perseo_core/servicios/biometria_senal.py`), que es donde se decide todo lo
 * demás. Aquí solo se corta el micrófono en **ventanas seguidas**, sin
 * recortarles nada por dentro, y se tira lo que es obvio que no lleva a ningún
 * sitio: minutos de silencio o una tos suelta.
 *
 * Por qué se cambió, el 2026-09-23. Antes esto era un detector de voz de
 * verdad, con un suelo de ruido que se recalculaba *también mientras
 * hablabas*: a los ~300 ms de frase el suelo alcanzaba a la voz, el umbral
 * —tres veces el suelo— se quedaba por encima de casi todo, y solo pasaban
 * picos de sílaba pegados unos a otros. Ningún envío llegaba a los 0,4 s que el
 * núcleo exigía, así que casi todo se tiraba al llegar y el reconocimiento de
 * voz apenas acertaba.
 *
 * Ahora el suelo sigue al mínimo —baja enseguida, sube muy despacio, por si
 * enciendes un ventilador—, y una ventana se cierra:
 *
 *  · al acabar la frase —400 ms callado tras haber oído algo—, o
 *  · a los tres segundos si se sigue hablando, y entonces empieza otra.
 *
 * Una ventana con menos de 0,8 s de sonido no se manda: el núcleo pide un
 * segundo de voz útil y no llegaría.
 */

export const MUESTRAS_SEGUNDO = 16000;

/** Tope de una ventana: tres segundos, y se abre la siguiente. */
export const VENTANA_MAXIMA = MUESTRAS_SEGUNDO * 3;

/** Sonido mínimo dentro de una ventana para que merezca el viaje. */
export const SONIDO_MINIMO = MUESTRAS_SEGUNDO * 0.8;

/** Silencio que da la frase por acabada. */
const SILENCIO_CIERRE = MUESTRAS_SEGUNDO * 0.4;

/** Lo que se guarda de antes de la primera sílaba, para no cortarle el ataque. */
const PREVIO = MUESTRAS_SEGUNDO * 0.2;

/** Por debajo de esto no hay nadie, por limpio que esté el micrófono (int16). */
const SONIDO_ABSOLUTO = 300;

function energiaRms(muestras: Int16Array): number {
  let suma = 0;
  for (let i = 0; i < muestras.length; i++) {
    suma += muestras[i] * muestras[i];
  }
  return Math.sqrt(suma / Math.max(1, muestras.length));
}

export class VentanaVoz {
  private trozos: Int16Array[] = [];
  private muestras = 0;
  /** Muestras de los trozos con sonido. */
  private sonido = 0;
  /** Muestras seguidas sin sonido al final de la ventana. */
  private silencioFinal = 0;
  private suelo = SONIDO_ABSOLUTO;

  /**
   * Mete un trozo del micrófono. Devuelve la ventana entera si se acaba de
   * cerrar y vale la pena mandarla; si no, `null`.
   */
  empujar(trozo: Int16Array): Int16Array | null {
    const rms = energiaRms(trozo);
    const hay = rms >= Math.max(SONIDO_ABSOLUTO, this.suelo * 2);
    // El suelo sigue al MÍNIMO: baja enseguida con cualquier trozo más callado
    // y sube muy despacio. Si siguiera a la media —como antes—, o aprendiera
    // rápido de lo que no pasa el umbral, los valles entre sílabas lo irían
    // subiendo hasta alcanzar a la voz a mitad de frase, que es exactamente
    // el fallo que hubo. Lo poco que sube es para un ruido que llega y se
    // queda —un ventilador—, que sin ello contaría como voz para siempre.
    this.suelo = rms < this.suelo ? (this.suelo + rms) / 2 : this.suelo * 0.995 + rms * 0.005;

    if (!hay && this.sonido === 0) {
      // Todavía no ha empezado nadie a hablar: solo se guarda un poco de antes.
      this.trozos.push(trozo);
      this.muestras += trozo.length;
      while (this.trozos.length > 1 && this.muestras - this.trozos[0].length >= PREVIO) {
        this.muestras -= this.trozos.shift()!.length;
      }
      return null;
    }

    this.trozos.push(trozo);
    this.muestras += trozo.length;
    if (hay) {
      this.sonido += trozo.length;
      this.silencioFinal = 0;
    } else {
      this.silencioFinal += trozo.length;
    }

    if (this.muestras >= VENTANA_MAXIMA || this.silencioFinal >= SILENCIO_CIERRE) {
      return this.cerrar();
    }
    return null;
  }

  /** Tira lo acumulado: se empieza de cero con el trozo siguiente. */
  descartar(): void {
    this.trozos = [];
    this.muestras = 0;
    this.sonido = 0;
    this.silencioFinal = 0;
  }

  private cerrar(): Int16Array | null {
    const vale = this.sonido >= SONIDO_MINIMO;
    const ventana = vale ? unir(this.trozos, this.muestras) : null;
    this.descartar();
    return ventana;
  }
}

function unir(trozos: Int16Array[], total: number): Int16Array {
  const junta = new Int16Array(total);
  let cursor = 0;
  for (const trozo of trozos) {
    junta.set(trozo, cursor);
    cursor += trozo.length;
  }
  return junta;
}
