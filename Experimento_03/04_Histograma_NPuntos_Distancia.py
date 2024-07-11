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

ORIGEN = (-2, 1)
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


# Determinamos los límites para los índices de los puntos
limites = np.arange(0, N_MUESTRAS, N_MUESTRAS//N_ESFERAS)

# # Espacio Latente
# distanciasLS = np.squeeze(cdist(datosLS[indiceCentro], datosLS, metric=METRICA))
# indicesMarcadosLS = np.argsort(distanciasLS)
# alturasLS = distanciasLS[indicesMarcadosLS[limites[1:]]]

# Espacio Ambiente
# Por puntos
distanciasAS = np.squeeze(cdist(datosAS[indiceCentro], datosAS, metric=METRICA))
indicesMarcadosAS = np.argsort(distanciasAS)
alturasDistancia = distanciasAS[indicesMarcadosAS[limites[1:]]]

# Por distancias
maxdistAS = distanciasAS.max()
radiosAS = np.linspace(0, maxdistAS, N_ESFERAS)
alturasPuntos = []

for i, r in enumerate(radiosAS):
    indicesMarcados = np.squeeze((radiosAS[i-1] <= distanciasAS) & (distanciasAS <= r))
    alturasPuntos.append(len(distanciasAS[indicesMarcados]))

alturasPuntos = np.asarray(alturasPuntos)*(100) / N_MUESTRAS # Lo pasamos a porcentaje
alturasPuntosAcum = np.cumsum(alturasPuntos) # La acumulada


# ------------------------------------------------------------------------------------------- #


# Generamos el barplot a # de puntos constante
fig, ax = plt.subplots(1, 1, figsize=(SIZE_FIG, SIZE_FIG))
# fig.tight_layout()

ax.bar(x=limites[1:], height=alturasDistancia, align="edge", width=N_MUESTRAS//N_ESFERAS)

ax.set_xticks(limites[1:])
ax.set_title("Espacio Ambiente")
ax.set_xlabel("Numero de puntos")
ax.set_ylabel("Distancia")

fig.savefig(PATH_FIGURES + "BarplotDistancia.png")

# ------------------------------------------------------------------------------------------- #

# Generamos el barplot a distancia constante
fig, ax = plt.subplots(1, 1, figsize=(SIZE_FIG, SIZE_FIG))
# fig.tight_layout()

ax.bar(x=radiosAS, height=alturasPuntos, align="edge", width=radiosAS[1]-radiosAS[0])

ax.set_xticks(radiosAS)
ax.set_title("Espacio Latente")
ax.set_xlabel("Radio de la estructura")
ax.set_ylabel("% de puntos añadidos")

fig.savefig(PATH_FIGURES + "BarplotNPuntos.png")

# ------------------------------------------------------------------------------------------- #

# Generamos el barplot a distancia constante acumulado
fig, ax = plt.subplots(1, 1, figsize=(SIZE_FIG, SIZE_FIG))
# fig.tight_layout()

ax.bar(x=radiosAS, height=alturasPuntosAcum, align="edge", width=radiosAS[1]-radiosAS[0])

ax.set_xticks(radiosAS)
ax.set_title("Espacio Latente")
ax.set_xlabel("Radio de la estructura")
ax.set_ylabel("% acumulado de puntos")

fig.savefig(PATH_FIGURES + "BarplotNPuntosAcum.png")


plt.show()
