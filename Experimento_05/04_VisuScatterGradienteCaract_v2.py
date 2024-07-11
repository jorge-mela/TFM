"""
# =========================================================================================== #
# Genera dos mapas coloreados en función del valor de una característica, en este caso se     #
# usan el rotacional y la divergencia.                                                        #
#                                                                                             #
# Autor: Jorge Menéndez Lagunilla                                                             #
# Fecha: 02/2024                                                                              #
#                                                                                             #
# =========================================================================================== #
"""

# =========================================================================================== #
# LIBRERÍAS

import os

import h5py
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler

from libs.generales import obtener_dataset

# =========================================================================================== #
# PARÁMETROS

PATH_PROYECCION = "./temporal/1Drotdiv/proyecciones/CAE1D_proyecciones_TEMPORAL.hdf5"
PATH_DATOS = "./temporal/1Drotdiv/proyecciones/CAE1D_proyecciones_TEMPORAL.hdf5"

norm = Normalize(vmin=-1, vmax=1, clip=False)


# =========================================================================================== #
# FUNCIONES




# =========================================================================================== #
# MAIN

datos = obtener_dataset(PATH_DATOS, "muestras")
proyeccion = obtener_dataset(PATH_PROYECCION, "proyeccion")


N_CANALES = datos.shape[-1]

# Como los datos son bidimensionales, reducimos cada característica al valor medio temporal
datos = datos.mean(axis=1)

labels = ["rotacional", "divergencia", "Ex", "Ey", "Txy", "bx", "by"]

s = [slice(None)] * len(datos.shape)

# fig = plt.figure()
# axes=[]
# for i in range(N_CANALES):
#     plt.subplot(2,4,i+1)
#     plt.scatter(proyeccion[:,0], proyeccion[:,1], c=datos[:,i], cmap="seismic", s=1)
#     plt.title(labels[i])

# plt.subplot(2,4,i+2)
# plt.axis("off")

# fig.colorbar(cm.ScalarMappable(norm=norm, cmap="seismic")) # ax=axes[i]

fig, ax = plt.subplots(2,4,sharex=True, sharey=True)
for i in range(N_CANALES):
    ax[i//4, i%4].scatter(proyeccion[:,0], proyeccion[:,1], c=datos[:,i], cmap="seismic", s=1)
    ax[i//4, i%4].set_title(labels[i])

ax[-1,-1].set_axis_off()

plt.colorbar(cm.ScalarMappable(norm=norm, cmap="seismic")) # ax=axes[i]

plt.show() 