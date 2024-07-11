"""
+-------------------------------------------------------------------------------------------+ 
| Visualización de las sparklines dentro de la proyección 2D para observar la ordenación    |
| general conseguida por el embedding.                                                      |
|                                                                                           |
| Autor: Jorge Menéndez Lagunilla                                                           |
| Fecha: 02/2024                                                                            |
|                                                                                           |
+-------------------------------------------------------------------------------------------+
"""

# =========================================================================================== #
# LIBRERIAS

import sys

import matplotlib.colors as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage, DrawingArea
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, "./")
from libs.generales import obtener_dataset, reset_seeds


# =========================================================================================== #
# PARÁMETROS

NOMBRE_ARCHIVO = "ROT_CAE2D_01_proyecciones_5_K3.hdf5"
PATH_DATOS = "./Experimento_03/proyecciones/"
PATH_FIGURES = "./Experimento_03/figures/"
PATH_FLECHAS = "./Experimento_03/data/01_muestras.hdf5"

SEED = 42

N_CHANNELS = 1
SIZE_FIG = 8
ZOOM_IMGS = 0.3
REDUCCION_FLECHAS=8

# Para colorear las imágenes
colorMap = cm.Colormap("coolwarm", 256)
norm = cm.Normalize(vmin=-1, vmax=1)


# =========================================================================================== #
# FUNCIONES


def poisson_disc_centres(r: float, coords: np.ndarray):

    available = coords.copy()
    centros_ = []

    # Para controlar que no se quede pinzado
    k = 0

    while available.shape[0] != 0 and k < 30:
        shapeAnterior = available.shape[0]
        pt = np.expand_dims(available[np.random.randint(0, available.shape[0])], 0)
        centros_.append(pt)
        dist = cdist(pt, available)

        # Nos quedamos solo con los elementos que están suficientemente alejados
        elim = dist > (2 * r)
        available = available[elim[0]]

        # Si no se reduce la longitud de available, terminamos
        if available.shape[0] == shapeAnterior:
            k = k + 1
        else:
            k = 0
            shapeAnterior = available.shape[0]

    centros = np.asarray(centros_)
    centros = np.squeeze(centros)

    return centros


def poisson_disc_indices(r: float, coords: np.ndarray):

    available = coords.copy()
    indx = np.arange(len(available))
    indices_ = []

    # Para controlar que no se quede pinzado
    k = 0

    while available.shape[0] != 0 and k < 30:
        shapeAnterior = available.shape[0]
        i = np.random.randint(0, available.shape[0])
        indices_.append(indx[i])
        pt = np.expand_dims(available[i], 0)
        dist = cdist(pt, available)

        # Nos quedamos solo con los elementos que están suficientemente alejados
        elim = dist > (2 * r)
        available = available[elim[0]]
        indx = indx[elim[0]]

        # Si no se reduce la longitud de available, terminamos
        if available.shape[0] == shapeAnterior:
            k = k + 1
        else:
            k = 0
            shapeAnterior = available.shape[0]

    return indices_


# =========================================================================================== #
# MAIN

reset_seeds(SEED)

# Cargamos la proyeccion
print("Cargamos los datos")
proyeccion = obtener_dataset(PATH_DATOS + NOMBRE_ARCHIVO, "proyeccion")
# campoXY = obtener_dataset(PATH_FLECHAS, "campoXY")
campoUV = obtener_dataset(PATH_FLECHAS, "flechas")
campoUV = campoUV[:, ::REDUCCION_FLECHAS, ::REDUCCION_FLECHAS, :] # Para que sea comprensible
campoX, campoY = np.meshgrid(
    np.linspace(0, 0.5, campoUV.shape[-3]), np.linspace(0, 0.5, campoUV.shape[-2])
)
print("Datos cargados")

# Generamos unos indices aleatorios para plotear
indices = poisson_disc_indices(0.4, proyeccion)

# ------------------------------------------------------------------------------------------- #

fig, ax = plt.subplots(1, 1, figsize=(SIZE_FIG, SIZE_FIG))
fig.tight_layout()
ax.scatter(proyeccion[:, 0], proyeccion[:, 1], alpha=0.3)
ax.set_title("Campos de Flechas")
ax.set_aspect('equal', 'box')

for i in indices:
    ax.quiver(
        campoX + proyeccion[i, 0],
        campoY + proyeccion[i, 1],
        campoUV[i, ..., 0] * ZOOM_IMGS,
        -campoUV[i, ..., 1] * ZOOM_IMGS,
        color="black",
        width=0.001,
        headwidth=2,
        scale=150,
    )

fig.savefig(PATH_FIGURES + "scatter_Flechas.png")

# ------------------------------------------------------------------------------------------- #


plt.show()
