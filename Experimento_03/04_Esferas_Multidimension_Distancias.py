"""
# =========================================================================================== #
# Visualización no interactiva que genera el scatter plot en 2 dimensiones de los datos y     #
# las muestras que dieron lugar a los 11 puntos más cercanos al punto seleccionado en la      #
# proyección, saleccionado por índice o por coordenadas (baja dimensión).                     #
#                                                                                             #
# Autor: Jorge Menéndez Lagunilla                                                             #
# Fecha: 12/2023                                                                              #
#                                                                                             #
# =========================================================================================== #
"""


# =========================================================================================== #
# LIBRERIAS

import sys

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from matplotlib.colors import Normalize, Colormap
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, "./")
from libs.generales import obtener_dataset


# =========================================================================================== #
# PARÁMETROS

NOMBRE_ARCHIVO = "ROT_CAE2D_04_proyecciones_15_K2.hdf5"
PATH_DATOS = "./Experimento_03/proyecciones/"
PATH_FIGURES = "./Experimento_03/figures/"

SIZE_FIG = 8
N_CHANNELS = 1

# puntos = ((-8,4), (-22,4), (0,4))

ORIGEN = (-8, -2)
METRICA = "cityblock" # Dado que se suele trabajar con dimensiones altas se recomienda usar distancia Manhattan = Cityblock
# Salen (N_ESFERAS-1) esferas porque la primera siempre tiene radio 0
N_ESFERAS = 10  # No se recomienda pasar de 12 (no hay más colores en el colormap)

# Generamos 12 colores para poder seleccionar hasta 12 zonas distitas
colormap = get_cmap("Set3")
aux = np.linspace(0, 1, 12)
colores =  colormap(aux) # Generamos los colores como tal


# =========================================================================================== #
# FUNCIONES



# =========================================================================================== #
# ERRORES




# =========================================================================================== #
# MAIN

# Cargamos la proyeccion y sus datos
proyeccion = obtener_dataset(PATH_DATOS + NOMBRE_ARCHIVO, "proyeccion")
N_MUESTRAS = proyeccion.shape[0]

datosLS = obtener_dataset(PATH_DATOS+ NOMBRE_ARCHIVO, "espacioLatente")

datosAS = obtener_dataset(PATH_DATOS+ NOMBRE_ARCHIVO, "muestras")
datosAS = datosAS.reshape(N_MUESTRAS, -1) # Para calcular las distancias fácilmente

# Seleccionamos el punto en función de las coordenadas introducidas por el usuario
cercanosPR = np.argsort(cdist(np.expand_dims(ORIGEN, 0), proyeccion))
indiceCentro = cercanosPR[...,0]


# Calculamos los radios de las esferas en función de la máxima distancia
# Espacio Latente
distanciasLS = cdist(datosLS[indiceCentro], datosLS, metric=METRICA)
maxdistLS = distanciasLS.max()
radiosLS = np.linspace(0, maxdistLS, N_ESFERAS)

# Espacio Ambiente
distanciasAS = cdist(datosAS[indiceCentro], datosAS, metric=METRICA)
maxdistAS = distanciasAS.max()
radiosAS = np.linspace(0, maxdistAS, N_ESFERAS)


# ------------------------------------------------------------------------------------------- #


# # Generamos el scatter del espacio latente
# fig, ax = plt.subplots(1, 1, figsize=(SIZE_FIG, SIZE_FIG))
# fig.tight_layout()

# for i, r in enumerate(radiosLS):
#     if i != 0:
#         indicesMarcados = np.squeeze((radiosLS[i-1] <= distanciasLS) & (distanciasLS <= r))
#         ax.scatter(proyeccion[indicesMarcados, 0], proyeccion[indicesMarcados, 1], color=colores[i], label=r"d$\leq$"+ f"{np.round(r, 2)}", alpha=0.3)

# ax.scatter(proyeccion[indiceCentro, 0], proyeccion[indiceCentro, 1], marker='x', color=colores[0], label="Centro")
# ax.legend()
# ax.set_title("Espacio Latente")

# fig.savefig(PATH_FIGURES + "ScatterEsferasLS.png")

# ------------------------------------------------------------------------------------------- #

# Generamos el scatter del espacio ambiente
fig, ax = plt.subplots(1, 1, figsize=(SIZE_FIG, SIZE_FIG))
fig.tight_layout()

for i, r in enumerate(radiosAS):
    if i != 0:
        indicesMarcados = np.squeeze((radiosAS[i-1] <= distanciasAS) & (distanciasAS <= r))
        ax.scatter(proyeccion[indicesMarcados, 0], proyeccion[indicesMarcados, 1], color=colores[i], label=r"d$\leq$"+ f"{np.round(r, 2)}", alpha=0.3)

ax.scatter(proyeccion[indiceCentro, 0], proyeccion[indiceCentro, 1], marker='x', color=colores[0], label="Centro")
ax.legend()
ax.set_title("Espacio Ambiente")

fig.savefig(PATH_FIGURES + "ScatterEsferasAS.png")

# ------------------------------------------------------------------------------------------- #

# # Generamos el scatter con ambos
# fig, axes = plt.subplots(1, 2, figsize=(2*SIZE_FIG, SIZE_FIG), sharey=True, sharex=True)
# fig.tight_layout()

# for i, r in enumerate(radiosLS):
#     if i != 0:
#         indicesMarcados = np.squeeze((radiosLS[i-1] <= distanciasLS) & (distanciasLS <= r))
#         axes[0].scatter(proyeccion[indicesMarcados, 0], proyeccion[indicesMarcados, 1], color=colores[i], label=r"d$\leq$"+ f"{np.round(r, 2)}", alpha=0.3)

# axes[0].scatter(proyeccion[indiceCentro, 0], proyeccion[indiceCentro, 1], marker='x', color=colores[0], label="Centro")
# axes[0].legend()
# axes[0].set_title("Espacio Latente")
# axes[0].set_facecolor('k')

# for i, r in enumerate(radiosAS):
#     if i != 0:
#         indicesMarcados = np.squeeze((radiosAS[i-1] <= distanciasAS) & (distanciasAS <= r))
#         axes[1].scatter(proyeccion[indicesMarcados, 0], proyeccion[indicesMarcados, 1], color=colores[i], label=r"d$\leq$"+ f"{np.round(r, 2)}", alpha=0.3)

# axes[1].scatter(proyeccion[indiceCentro, 0], proyeccion[indiceCentro, 1], marker='x', color=colores[0], label="Centro")
# axes[1].legend()
# axes[1].set_title("Espacio Ambiente")
# axes[1].set_facecolor('k')

# fig.savefig(PATH_FIGURES + "ScatterEsferasLS_AS.png")



plt.show()
