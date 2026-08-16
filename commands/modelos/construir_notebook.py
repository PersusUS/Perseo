"""Construye commands/modelos/entrenar_colab.ipynb a partir del notebook de
alfiedennen/openwakeword-colab-2026 (MIT), sustituyendo su celda de configuración
por una que lea `perseo.yml`.

Se hace así, y no copiando el notebook a mano, para que las catorce celdas que no
tocamos —instalación, parches, descargas, entrenador y exportación— queden
**byte a byte** como las del original. Si mañana hay que actualizarlas, se vuelve
a ejecutar este guion contra su versión nueva.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

AQUI = Path(__file__).resolve().parent
DESTINO = AQUI / "entrenar_colab.ipynb"
YML = AQUI / "perseo.yml"

#: De dónde sale el notebook base. Se descarga en cada ejecución en vez de
#: guardarse aquí: lo que este repositorio mantiene es la celda de
#: configuración, no las catorce que no tocamos.
ORIGEN = (
    "https://raw.githubusercontent.com/alfiedennen/openwakeword-colab-2026/"
    "main/train_wakeword.ipynb"
)

RESPALDO_MD = """\
## 11-bis. Guardar lo caro en Drive (recomendado en la T4 gratuita)

Colab corta la sesión por inactividad y, al reciclar la máquina, **borra
`/content` entero**. Si te vas a dormir con la generación en marcha, al volver no
queda nada: ni los clips, ni el `train.py` clonado, ni la configuración. Pasó el
2026-08-17, y costó una hora de generación.

Esta celda copia a Drive lo que cuesta tiempo y **restaura** lo que encuentre
allí al volver a ejecutarla. Es idempotente: lánzala después de cada fase.

Si prefieres no montar Drive, sáltala y ejecuta el notebook de una sentada.
"""

RESPALDO = """
import glob
import os
import shutil

import yaml

CARPETA_DRIVE = "/content/drive/MyDrive/perseo_entrenamiento"

from google.colab import drive
drive.mount("/content/drive")

with open("/content/my_model.yaml") as f:
    cfg = yaml.safe_load(f)
SALIDA = cfg["output_dir"]
os.makedirs(CARPETA_DRIVE, exist_ok=True)

# La configuración es barata de copiar y es lo primero que se echa en falta.
shutil.copy("/content/my_model.yaml", CARPETA_DRIVE + "/my_model.yaml")

def cuantos(ruta):
    return len(glob.glob(ruta + "/**/*", recursive=True))

aqui, alli = cuantos(SALIDA), cuantos(CARPETA_DRIVE + "/salida")
if aqui > alli:
    print("  Guardando en Drive lo generado...")
    shutil.copytree(SALIDA, CARPETA_DRIVE + "/salida", dirs_exist_ok=True)
    print("  OK", cuantos(CARPETA_DRIVE + "/salida"), "ficheros a salvo")
elif alli:
    print("  Restaurando desde Drive lo de la sesion anterior...")
    shutil.copytree(CARPETA_DRIVE + "/salida", SALIDA, dirs_exist_ok=True)
    print("  OK", cuantos(SALIDA), "ficheros recuperados en", SALIDA)
else:
    print("  Nada que guardar todavia: ejecutala cuando haya clips.")
"""


PORTADA = """\
# Entrenar `perseo.onnx`

Este notebook entrena el modelo de la palabra clave de **Perseo** y devuelve un
`perseo.onnx` que se copia a `commands/modelos/` del repositorio. No hay que tocar
código: el detector lo encuentra solo.

## De dónde sale, y por qué no es el oficial

El notebook oficial de openWakeWord **está roto** desde noviembre de 2025
([issue #296](https://github.com/dscripka/openWakeWord/issues/296)): falta `piper`,
`generate_samples()` pide un argumento que nadie le pasa, y la augmentación falla con
*"Clip does not have the correct sample rate!"*, que tumba en cascada el entrenamiento.

Así que este notebook es el de
[alfiedennen/openwakeword-colab-2026](https://github.com/alfiedennen/openwakeword-colab-2026)
(MIT, © 2026 Alfie Dennen), que aplica seis parches conocidos y sustituye el entrenador de
openWakeWord por un bucle de PyTorch propio. **Todas sus celdas están tal cual**; lo único
que cambia es la de configuración, que aquí lee `perseo.yml` en vez de dos variables
sueltas — sin eso perderíamos los negativos en español y los pesos que ya están decididos.

## Antes de darle a "ejecutar todo"

- **Entorno de ejecución → GPU.** Con la T4 gratuita hay que recortar dos números de
  `perseo.yml`, y por eso están arriba de la celda 10: `N_MUESTRAS` baja a 20.000 (el mínimo
  que recomienda openWakeWord: ~1 h de generación en vez de ~2,5 h) y `N_PASOS` a 20.000.
  Contando todo, la sesión entera va sobre las **2 h**. Con Colab Pro y una L4, los dos a
  50000, que es lo que pide `perseo.yml`.
- **Descarga el `.onnx` en cuanto salga.** Si la sesión se corta, se pierde todo lo que no
  esté bajado.
- **Nada de esto se ha probado desde el repositorio**: no hay GPU aquí. Si una celda falla,
  el error concreto vale más que cualquier suposición — cópialo tal cual.

## El problema del idioma

El generador de muestras **habla inglés**, así que `perseo.yml` pide tres grafías —`perseo`,
`pair say oh`, `per say oh`— para que una voz inglesa produzca algo cercano al "Perseo"
español. Está explicado entero en `README.md`, en esta misma carpeta. Si al terminar el
modelo no reconoce tu voz, lo primero es bajar `PERSEO_UMBRAL_PALABRA` a 0.3 **antes** de
volver a entrenar.
"""

CONFIG = '''\
import os

import yaml

# ── Lo único que se toca aquí ─────────────────────────────────────────────
# Los dos valores de `perseo.yml` que no caben en una T4 gratuita. 20.000
# muestras es el mínimo que recomienda openWakeWord (~1 h de generación en vez
# de ~2,5 h), y 20.000 pasos es lo que trae el notebook del que sale este.
# Con una L4 de Colab Pro, los dos a 50000, que es lo que pide `perseo.yml`.
N_MUESTRAS = 20000
N_PASOS = 20000
# ──────────────────────────────────────────────────────────────────────────

# `perseo.yml` va incrustado y no se descarga: el repositorio es privado, así que
# desde Colab no hay forma de leerlo. Es una copia literal de
# commands/modelos/perseo.yml — si allí cambia algo, hay que traerlo aquí.
PERSEO_YML = """
{yml}
"""

perseo = yaml.safe_load(PERSEO_YML)

# Lo que decide Perseo, y no se toca: la palabra, los negativos en español y los
# pesos. Es justamente lo que se perdería usando el notebook original.
config = {{
    "target_phrase": perseo["target_phrase"],
    "model_name": perseo["model_name"],
    "custom_negative_phrases": perseo["custom_negative_phrases"],
    "max_negative_weight": perseo["max_negative_weight"],
    "target_false_positives_per_hour": perseo["target_false_positives_per_hour"],
    "steps": N_PASOS,
    "model_type": perseo["model_type"],
    "layer_size": perseo["layer_size"],
    "tts_batch_size": perseo["tts_batch_size"],
    "augmentation_batch_size": perseo["augmentation_batch_size"],
    "augmentation_rounds": perseo["augmentation_rounds"],
    "background_paths_duplication_rate": perseo["background_paths_duplication_rate"],
    "n_samples": N_MUESTRAS,
    "n_samples_val": perseo["n_samples_val"],

    # Claves que el entrenador de este notebook necesita y `perseo.yml` no tiene:
    # el YAML de openWakeWord no las lleva porque su entrenador las trae por
    # dentro. Los valores son los del notebook original salvo `layer_dim`, que se
    # iguala a nuestro `layer_size`.
    "target_accuracy": 0.7,
    "target_recall": 0.5,
    "batch_size": 128,
    "learning_rate": 1e-4,
    "layer_dim": perseo["layer_size"],
    "n_blocks": 1,
    "model_input_shape": [16, 96],
    "n_classes": 1,
    "batch_n_per_class": perseo["batch_n_per_class"],
    "tflite_export": False,   # en Windows no hay tflite-runtime; solo hace falta el .onnx
    "onnx_export": True,

    # Rutas de Colab. Las de `perseo.yml` son relativas a la máquina donde se
    # entrene, y aquí esa máquina es esta.
    "piper_sample_generator_path": "/content/piper-sample-generator",
    "background_paths": ["/content/fma_wav"],
    "rir_paths": ["/content/mit_rirs"],
    "false_positive_validation_data_path": "/content/acav_val_subset.npy",
    "feature_data_files": {{"ACAV100M_sample": "/content/acav_train_subset.npy"}},
}}

NOMBRE = config["model_name"]
SALIDA = f"/content/{{NOMBRE}}_output"
config.update({{
    "output_dir": SALIDA,
    "positive_clips_train_dir": f"{{SALIDA}}/{{NOMBRE}}/positive_train",
    "positive_clips_test_dir": f"{{SALIDA}}/{{NOMBRE}}/positive_test",
    "negative_clips_train_dir": f"{{SALIDA}}/{{NOMBRE}}/negative_train",
    "negative_clips_test_dir": f"{{SALIDA}}/{{NOMBRE}}/negative_test",
    "feature_save_dir": f"{{SALIDA}}/{{NOMBRE}}",
}})

os.makedirs(config["output_dir"], exist_ok=True)
with open("/content/my_model.yaml", "w") as f:
    yaml.dump(config, f, sort_keys=False, allow_unicode=True)

print(f'  palabra:        {{config["target_phrase"]}}')
print(f'  negativos:      {{len(config["custom_negative_phrases"])}} en español')
print(f'  muestras:       {{config["n_samples"]}} (perseo.yml pide {{perseo["n_samples"]}})')
print(f'  pasos:          {{config["steps"]}} (perseo.yml pide {{perseo["steps"]}})')
print(f'  peso negativo:  {{config["max_negative_weight"]}}')
print(f'  falsos pos./h:  {{config["target_false_positives_per_hour"]}}')
print(f'  salida:         {{SALIDA}}')
'''


def main() -> None:
    print(f"descargando {ORIGEN}")
    with urllib.request.urlopen(ORIGEN, timeout=60) as respuesta:
        notebook = json.loads(respuesta.read().decode("utf-8"))
    celdas = notebook["cells"]

    # La celda 0 es su portada; la 20, su configuración. Se localizan por su
    # contenido y no por el índice, para que este guion siga valiendo si el
    # original añade celdas por delante.
    i_portada = 0
    i_config = next(
        i for i, c in enumerate(celdas) if "EDIT THESE TWO LINES" in "".join(c["source"])
    )

    yml = YML.read_text(encoding="utf-8").strip()
    if '"""' in yml:
        raise SystemExit("perseo.yml lleva comillas triples; romperían la cadena incrustada")

    celdas[i_portada] = {
        "cell_type": "markdown",
        "metadata": {},
        "source": PORTADA.splitlines(keepends=True),
    }
    celdas[i_config] = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": CONFIG.format(yml=yml).splitlines(keepends=True),
    }

    # Su celda de markdown anterior a la configuración anuncia "no example-yaml
    # dependency", que aquí ya no es verdad.
    anterior = "".join(celdas[i_config - 1]["source"])
    if celdas[i_config - 1]["cell_type"] == "markdown" and "training config" in anterior.lower():
        celdas[i_config - 1]["source"] = [
            "## 10. La configuración, desde `perseo.yml`\n",
            "\n",
            "La palabra, los negativos en español y los pesos salen del repositorio.\n",
            "Las rutas y las claves que pide el entrenador de este notebook se añaden aquí.\n",
        ]

    # Y su título de sección del entrenador dice L4/A100; con T4 tarda más.
    for celda in celdas:
        if celda["cell_type"] != "markdown":
            continue
        texto = "".join(celda["source"])
        if "Hand-rolled trainer" in texto:
            celda["source"] = [
                re.sub(r"~30-40 min on L4 / A100", "~30-40 min en L4, ~1 h en T4", texto)
            ]

    # La red de seguridad va justo despues de la generacion, que es la fase que
    # cuesta una hora y la que se pierde entera si Colab recicla la maquina.
    i_generar = next(
        i for i, c in enumerate(celdas) if "--generate_clips" in "".join(c["source"])
    )
    celdas.insert(
        i_generar + 1,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": RESPALDO_MD.splitlines(keepends=True),
        },
    )
    celdas.insert(
        i_generar + 2,
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": RESPALDO.strip().splitlines(keepends=True),
        },
    )

    DESTINO.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"escrito {DESTINO} ({DESTINO.stat().st_size / 1024:.0f} KB, {len(celdas)} celdas)")
    print(f"portada en la celda {i_portada}, configuración en la {i_config}")


if __name__ == "__main__":
    main()
