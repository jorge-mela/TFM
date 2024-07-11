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

NOMBRE_ARCHIVO = "ROT_CAE2D_04_proyecciones_15_K2.hdf5"
PATH_DATOS = "./Experimento_02/proyecciones/"
PATH_FIGURES = "./Experimento_02/figures/"

SIZE_FIG = 8
N_CHANNELS = 1

PATH_MODELOS = "./Experimento_02/models/"
MODELO_ENCODER = "encoder_ROT_CAE2D_04_15_K1.keras"
MODELO_DECODER = "decoder_ROT_CAE2D_04_15_K1.keras"

encoder = load_model(PATH_MODELOS + MODELO_ENCODER)
decoder = load_model(PATH_MODELOS + MODELO_DECODER)


# =========================================================================================== #
# FUNCIONES

def generar_plots(data: np.ndarray, indice:int, vmin:int, vmax:int):

    norm = Normalize(vmin=vmin, vmax=vmax)
    plt.colorbar(ScalarMappable(norm=norm, cmap="coolwarm"), ticks=(-1, 0, 1))
    plt.imshow(data[indice], cmap="coolwarm", norm=norm)
    plt.xticks(())

    return None


# =========================================================================================== #
# MAIN

reset_seeds(42)

# Cargamos la proyeccion
datos = obtener_dataset(PATH_DATOS+NOMBRE_ARCHIVO, "muestras")
scaler = MinMaxScaler((-1,1))
datos_visu = scaler.fit_transform(datos.reshape(-1,N_CHANNELS)).reshape(datos.shape)
print("Shape muestras: ", datos.shape)


N_MUESTRAS = datos.shape[0]
indices = np.random.randint(low=0, high=N_MUESTRAS, size=12)

# ------------------------------------------------------------------------------------------- #

# Figuras de los datos
f = plt.figure(figsize=(SIZE_FIG, SIZE_FIG))
plt.suptitle("Datos Brutos")
for i, indx in enumerate(indices):

    ax = plt.subplot(3, 4, i+1)
    if i==0:
        ax.title.set_text(f"Muestra {indx}")
    else:
        titulo = f"Muestra {indx}"
        ax.title.set_text(titulo)

    generar_plots(datos_visu, indx, -1, 1)

f.tight_layout()
f.savefig(PATH_FIGURES + "ReconstruccionDatosBrutos.png")

# ------------------------------------------------------------------------------------------- #

datosDecodificados = decoder.predict((encoder.predict((datos[indices, :, :]))))
datosDecodificados_visu = scaler.transform(datosDecodificados.reshape(-1,N_CHANNELS)).reshape(datosDecodificados.shape)

f = plt.figure(figsize=(SIZE_FIG, SIZE_FIG))
plt.suptitle("Datos Decodificados")
for indx in range(len(indices)):

    ax = plt.subplot(3, 4, indx+1)

    if i==0:
        ax.title.set_text(f"Muestra {indices[indx]}")
    else:
        titulo = f"Muestra {indices[indx]}"
        ax.title.set_text(titulo)

    generar_plots(datosDecodificados_visu, indx, -1, 1)

f.tight_layout()
f.savefig(PATH_FIGURES + "ReconstruccionDatosDecodificados.png")

# ------------------------------------------------------------------------------------------- #

plt.show()