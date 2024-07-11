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
from sklearn.manifold import TSNE

from libs.generales import obtener_dataset

# =========================================================================================== #
# PARÁMETROS

PATH_PROYECCION = "./atemporal/fft/proyecciones/relu_proyecciones_3.hdf5"
PATH_DATOS = "./atemporal/fft/proyecciones/relu_proyecciones_3.hdf5"

SEED = 42

norm = Normalize(vmin=-1, vmax=1, clip=False)


# =========================================================================================== #
# FUNCIONES




# =========================================================================================== #
# MAIN

datos = obtener_dataset(PATH_DATOS, "muestras")
espacioLatente = obtener_dataset(PATH_PROYECCION, "espacioLatente")

tsne = TSNE(n_components=2, perplexity=80, init="pca", random_state=SEED, early_exaggeration=25,
            n_iter=2000, n_iter_without_progress=500)

proyeccion = tsne.fit_transform(espacioLatente)

N_CANALES = datos.shape[-1]

labels = ["rotacional", "divergencia", "Ex", "Ey", "Txy", "bx", "by"]

s = [slice(None)] * len(datos.shape)

fig = plt.figure()
axes=[]
for i in range(N_CANALES):
    plt.subplot(2,4,i+1)
    plt.scatter(proyeccion[:,0], proyeccion[:,1], c=datos[:,i], cmap="seismic", s=1)
    plt.title(labels[i])

plt.subplot(2,4,i+2)
plt.axis("off")

fig.colorbar(cm.ScalarMappable(norm=norm, cmap="seismic")) # ax=axes[i]

plt.show() 