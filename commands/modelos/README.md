# El modelo de la palabra clave

Aquí vive el modelo que decide si has dicho "Perseo". El detector
(`../palabra_clave.py`) busca **`perseo.onnx` en esta carpeta**; si no lo
encuentra, carga un modelo de repuesto y lo dice por registro cada vez que
arranca.

```
commands/modelos/perseo.onnx            <- el modelo propio, cuando exista
commands/modelos/perseo.yml             <- la configuración con la que se entrena
commands/modelos/entrenar_colab.ipynb   <- el notebook que lo entrena, listo para subir
commands/modelos/construir_notebook.py  <- cómo se generó ese notebook
```

Se puede apuntar a otro sitio con `PERSEO_MODELO_PALABRA=C:\ruta\al\modelo.onnx`,
y mover el listón de sensibilidad con `PERSEO_UMBRAL_PALABRA` (0.5 por defecto).

---

## Estado

**El modelo propio no está entrenado todavía.** Mientras tanto el detector carga
`hey_jarvis`, uno de los pre-entrenados que trae openWakeWord. Eso significa que
hoy el circuito funciona de punta a punta —aplaudes, habla, decide en local en
menos de 200 ms— pero **la palabra que reconoce es "hey jarvis", no "Perseo"**.

Se comprueba en cualquier momento con:

```bash
python commands/verificar_palabra_clave.py
```

El verificador dice qué modelo tiene cargado en su primera línea, y adapta la
frase de prueba a lo que haya.

---

## El problema del idioma

Esta es la parte que hay que entender antes de ponerse, porque es lo que separa
"un rato de Colab" de "un fin de semana".

openWakeWord no se entrena con grabaciones de gente: se entrena con miles de
muestras sintéticas. Quien las genera es
[piper-sample-generator](https://github.com/dscripka/piper-sample-generator), y
ese generador **habla inglés**. Trae un solo modelo, `en-us-libritts-high.pt`, y
la documentación no explica cómo usar una voz de otro idioma — no es el `.onnx`
de Piper que se descarga por ahí, sino un punto de control de entrenamiento del
que no hay equivalente español publicado.

Consecuencia: si le pides al generador la palabra `perseo`, mil voces inglesas
dirán algo parecido a *"pur-SEE-oh"*, y el modelo aprenderá **eso**. Luego dices
"Perseo" en español y no se entera.

Hay tres salidas, en orden de lo que cuestan:

1. **Escribir la palabra como suena en inglés.** Es lo que hace `perseo.yml`:
   pide las tres grafías `perseo`, `pair say oh` y `per say oh`. El modelo es
   binario y se activa con cualquiera de ellas, así que lo que acaba aprendiendo
   es un sonido muy cercano al "Perseo" español. Es el camino barato y el que
   conviene probar primero.
2. **Grabar muestras propias y mezclarlas.** openWakeWord admite clips reales
   junto a los sintéticos. Con unos cientos de "Perseo" dichos por ti, en sitios
   y tonos distintos, el modelo se ajusta a tu voz. Es la mejor opción para un
   asistente de una sola persona, y también la más aburrida de conseguir.
3. **Portar una voz Piper española al generador.** La más limpia y la que más
   trabajo tiene: hay que convertir un `es_ES` de Piper al formato que el
   generador espera. No hay receta publicada.

Si después de entrenar con la vía 1 el modelo responde mal a tu voz, la vía 2 se
puede añadir encima sin volver a empezar.

---

## Entrenar

**No en este portátil.** La RTX 4050 tiene 6 GB y están comprometidos con el
modelo de 4B del router del núcleo (restricción 1 del handoff). Va a Colab con
GPU.

### 0. El notebook oficial no vale — usa `entrenar_colab.ipynb`

Esto es lo primero que hay que saber, porque cuesta una sesión entera
descubrirlo: el
[notebook oficial de openWakeWord](https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb)
**está roto** desde noviembre de 2025
([issue #296](https://github.com/dscripka/openWakeWord/issues/296)). Falla en
cuatro sitios encadenados:

| Error | Qué es |
|---|---|
| `ModuleNotFoundError: No module named 'piper'` | La dependencia ya no se instala sola |
| `generate_samples() missing 1 required positional argument: 'model'` | El notebook no pasa el argumento que la función pide |
| `ValueError: Error! Clip does not have the correct sample rate!` | Piper devuelve 22.050 Hz y la augmentación quiere 16.000 |
| `FileNotFoundError: positive_features_test.npy` | Consecuencia del anterior: sin augmentar no hay características que entrenar |

En esta carpeta hay un notebook que sí funciona: **`entrenar_colab.ipynb`**. Se
sube a Colab, se pone el entorno de ejecución en GPU y se ejecuta entero. Solo
tiene dos números que tocar, arriba de la celda 10 (`N_MUESTRAS` y `N_PASOS`), y
vienen puestos para una **T4 gratuita**: unas dos horas de principio a fin.

Sale del notebook de
[alfiedennen/openwakeword-colab-2026](https://github.com/alfiedennen/openwakeword-colab-2026)
(MIT, © 2026 Alfie Dennen), que aplica seis parches conocidos y sustituye el
entrenador de openWakeWord por un bucle de PyTorch propio. **Sus celdas están tal
cual**; lo único que cambia es la configuración, que aquí lee `perseo.yml` — sin
eso se perderían los negativos en español y los pesos.

Descarga solo lo que hace falta (`mit_rirs`, FMA, las características de
ACAV100M y el conjunto de validación de falsos positivos), así que no hay nada
que preparar a mano.

Para regenerarlo cuando el original cambie:

```bash
python commands/modelos/construir_notebook.py
```

**Ojo con una cosa:** `perseo.yml` va **incrustado** dentro del notebook, porque
el repositorio es privado y desde Colab no se puede leer. Si cambias el `.yml`,
vuelve a ejecutar ese guion.

### 1. Traerse el modelo

La última celda exporta `perseo.onnx` y lo baja. **Bájalo en cuanto salga**: si
Colab corta la sesión, se pierde lo que no esté en tu disco. El `.tflite` no se
genera a propósito — en Windows no hay `tflite-runtime`, y por eso el detector
pide `onnx` explícitamente.

El fichero se copia a esta carpeta y ya está: no hay que tocar código.

### 2. Comprobarlo

```bash
python commands/verificar_palabra_clave.py
```

Al ver `perseo.onnx`, el verificador cambia solo la frase de prueba a "Perseo" con
voz española. Si sale en verde, queda lo único que no se puede automatizar:
aplaudir dos veces y hablarle.

---

## Si sale mal

| Síntoma | Dónde mirar |
|---|---|
| No reconoce tu voz nunca | Es lo esperable de la vía 1 si tu acento se aleja del inglés. Baja `PERSEO_UMBRAL_PALABRA` a 0.3 y vuelve a probar antes de reentrenar |
| Se activa con cualquier cosa | Sube el umbral a 0.7, o añade lo que lo dispara a `custom_negative_phrases` y reentrena |
| El verificador falla al cargar | El `.onnx` está a medio copiar o es el `.tflite` renombrado |
| Una celda del notebook revienta | Copia el error tal cual. **Nada de esto se ha podido probar desde el repositorio** —aquí no hay GPU—, así que el mensaje concreto vale más que cualquier suposición. Los seis parches de la celda 3 son idempotentes: volver a ejecutarla no rompe nada |
| Colab corta la sesión a media generación | Las celdas son idempotentes y se saltan lo ya hecho: vuelve a ejecutar desde el principio y retomará donde estaba, salvo que se haya reiniciado la máquina |
