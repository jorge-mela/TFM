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

sys.path.insert(0, "./")
from libs.generales import obtener_dataset, reset_seeds


# =========================================================================================== #
# PARÁMETROS

NOMBRE_ARCHIVO = "CAE1D_01_proyecciones_15_K3.hdf5"
PATH_DATOS = "./Experimento_01/proyecciones/"
PATH_FIGURES = "./Experimento_01/figures/"

SIZE_FIG = 8
N_CHANNELS = 7

PATH_MODELOS = "./Experimento_01/models/"
MODELO_ENCODER = "encoder_CAE1D_01_15_K3.keras"
MODELO_DECODER = "decoder_CAE1D_01_15_K3.keras"

encoder = load_model(PATH_MODELOS + MODELO_ENCODER)
decoder = load_model(PATH_MODELOS + MODELO_DECODER)


# =========================================================================================== #
# FUNCIONES


def generar_plots(data: np.ndarray, indice: int, vmin: int, vmax: int):

    norm = Normalize(vmin=vmin, vmax=vmax)
    plt.colorbar(ScalarMappable(norm=norm, cmap="coolwarm"), ticks=(-1, 0, 1))
    plt.imshow(data[indice], cmap="coolwarm")  # vmin=vmin, vmax=vmax
    plt.xticks(())

    return None


# =========================================================================================== #
# MAIN

reset_seeds(42)

# Cargamos la proyeccion
datos = obtener_dataset(PATH_DATOS + NOMBRE_ARCHIVO, "muestras")
scaler = MinMaxScaler((-1, 1))
datos_visu = scaler.fit_transform(datos.reshape(-1, N_CHANNELS)).reshape(datos.shape)


N_MUESTRAS = datos.shape[0]
indicesVecinos = indiceCentro = np.random.randint(low=0, high=N_MUESTRAS, size=12)

# ------------------------------------------------------------------------------------------- #

# Figuras de los datos
f = plt.figure(figsize=(SIZE_FIG, SIZE_FIG))
plt.suptitle("Datos Brutos")
for i, indx in enumerate(indicesVecinos):

    ax = plt.subplot(3, 4, i + 1)
    if i == 0:
        ax.title.set_text(f"Muestra {indx}")
    else:
        titulo = f"Muestra {indx}"
        ax.title.set_text(titulo)

    generar_plots(datos_visu, indx, -1, 1)

f.tight_layout()
f.savefig(PATH_FIGURES + "ReconstruccionDatosBrutos.png")

# ------------------------------------------------------------------------------------------- #

datosDecodificados = decoder.predict((encoder.predict((datos[indicesVecinos, :, :]))))
datosDecodificados_visu = scaler.transform(datosDecodificados.reshape(-1, N_CHANNELS))
datosDecodificados_visu = datosDecodificados_visu.reshape(datosDecodificados.shape)

f = plt.figure(figsize=(SIZE_FIG, SIZE_FIG))
plt.suptitle("Datos Reconstruidos")
for indx in range(len(indicesVecinos)):

    ax = plt.subplot(3, 4, indx + 1)

    if i == 0:
        ax.title.set_text(f"Muestra {indicesVecinos[indx]}")
    else:
        titulo = f"Muestra {indicesVecinos[indx]}"
        ax.title.set_text(titulo)

    generar_plots(datosDecodificados, indx, -1, 1)

f.tight_layout()
f.savefig(PATH_FIGURES + "ReconstruccionDatosDecodificados.png")

# ------------------------------------------------------------------------------------------- #

plt.show()
