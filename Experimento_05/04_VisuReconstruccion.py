"""
+-------------------------------------------------------------------------------------------+ 
| Visualización para comparar la reconstrucción obtenida y los datos originales de 12       |
| muestras aleatorias.                                                                      |
|                                                                                           |
| Autor: Jorge Menéndez Lagunilla                                                           |
| Fecha: 02/2024                                                                            |
|                                                                                           |
+-------------------------------------------------------------------------------------------+
"""


# =========================================================================================== #
# LIBRERIAS

import sys

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, './') 
from libs.generales import obtener_dataset, reset_seeds


# =========================================================================================== #
# PARÁMETROS

NOMBRE_ARCHIVO = "Deep_01_proyecciones_3_K1.hdf5"
PATH_DATOS = "./Experimento_05/proyecciones/"
PATH_FIGURES = "./Experimento_05/figures/"

SEED = 42

SIZE_FIG = 8
N_CHANNELS = 5

PATH_MODELOS = "./Experimento_05/models/"
MODELO_ENCODER = "encoder_Deep_01_3_K1.keras"
MODELO_DECODER = "decoder_Deep_01_3_K1.keras"

encoder = load_model(PATH_MODELOS + MODELO_ENCODER)
decoder = load_model(PATH_MODELOS + MODELO_DECODER)


# =========================================================================================== #
# FUNCIONES




# =========================================================================================== #
# MAIN

reset_seeds(SEED)

# Cargamos la proyeccion
datos = obtener_dataset(PATH_DATOS+NOMBRE_ARCHIVO, "muestras")
scaler = MinMaxScaler((-1,1))
datos_visu = scaler.fit_transform(datos.reshape(-1,N_CHANNELS)).reshape(datos.shape)
print("Shape muestras: ", datos.shape)


N_MUESTRAS = datos.shape[0]
indices = np.random.randint(low=0, high=N_MUESTRAS, size=12)

datosDecodificados = decoder.predict((encoder.predict((datos[indices]))))
datosDecodificados_visu = scaler.transform(datosDecodificados.reshape(-1,N_CHANNELS)).reshape(datosDecodificados.shape)

# ------------------------------------------------------------------------------------------- #

# Figuras de los datos
f, axes = plt.subplots(3, 4, figsize=(12, SIZE_FIG), sharex=True, sharey=True)
f.tight_layout()

for j, row in enumerate(axes):
    for i, ax in enumerate(row):
        ax.plot(datos_visu[indices[j*4+i]], label="Original")
        ax.plot(datosDecodificados_visu[j*4+i], label="Reconstruccion")

ax.set_ylim((-1,1))
ax.set_xticks(np.arange(0, 5))
ax.set_xticklabels([str(f) for f in range(1, 6, 1)])
ax.set_yticks(np.arange(-1, 1.5, 0.5))
ax.legend()

# f.suptitle("Entradas - Reconstrucción")

plt.show()

f.savefig(PATH_FIGURES + "ReconstruccionDatos.png")