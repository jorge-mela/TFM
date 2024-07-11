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

import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.manifold import TSNE

sys.path.insert(0, "./")
from libs.generales import obtener_dataset


# =========================================================================================== #
# PARÁMETROS
NOMBRE_ARCHIVO = "CAE1D_01_proyecciones_15_K3.hdf5"
PATH_DATOS = "./Experimento_01/proyecciones/"
PATH_FIGURES = "./Experimento_01/figures/"

PATH_PROYECCION = "./atemporal/1Drotdiv/proyecciones/relu_proyecciones_3.hdf5"
PATH_DATOS = "./atemporal/1Drotdiv/proyecciones/relu_proyecciones_3.hdf5"

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

fig.savefig(PATH_FIGURES + "GradienteCaract.png")

plt.show() 