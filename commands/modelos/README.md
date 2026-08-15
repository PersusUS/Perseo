# El modelo de la palabra clave

Aquí vive el modelo que decide si has dicho "Perseo". El detector
(`../palabra_clave.py`) busca **`perseo.onnx` en esta carpeta**; si no lo
encuentra, carga un modelo de repuesto y lo dice por registro cada vez que
arranca.

```
commands/modelos/perseo.onnx     <- el modelo propio, cuando exista
commands/modelos/perseo.yml      <- la configuración con la que se entrena
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
GPU, o a la RunPod.

### 1. Preparar la máquina

```bash
pip install openwakeword
git clone https://github.com/dscripka/piper-sample-generator
```

Hacen falta además, en el mismo directorio de trabajo:

| Qué | De dónde | Para qué |
|---|---|---|
| `mit_rirs/` | Respuestas al impulso del MIT | Simular habitaciones |
| `audioset_16k/`, `fma/` | AudioSet y Free Music Archive | Ruido y música de fondo |
| `validation_set_features.npy` | [openwakeword_features](https://huggingface.co/datasets/davidscripka/openwakeword_features) en Hugging Face | Medir falsos positivos contra ~11 h de habla, ruido y música |
| `openwakeword_features_ACAV100M_2000_hrs_16bit.npy` | El mismo sitio | Los negativos del entrenamiento |

El
[notebook oficial](https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb)
descarga todo eso solo; es la vía cómoda y la que conviene usar. `perseo.yml`
sustituye a la configuración que el notebook trae de ejemplo.

### 2. Lanzar

```bash
python -m openwakeword.train --training_config perseo.yml --generate_clips
python -m openwakeword.train --training_config perseo.yml --augment_clips
python -m openwakeword.train --training_config perseo.yml --train_model
```

Con `n_samples: 50000` la generación es lo que más tarda: entre una y dos horas
de GPU. El entrenamiento en sí son minutos.

### 3. Traerse el modelo

Sale `perseo.onnx` (y un `.tflite` que aquí no se usa: en Windows no hay
`tflite-runtime`, y por eso el detector pide `onnx` explícitamente). El `.onnx` se
copia a esta carpeta y ya está — no hay que tocar código.

### 4. Comprobarlo

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
