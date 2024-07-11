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
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, "./")
from libs.generales import obtener_dataset, reset_seeds


# =========================================================================================== #
# PARÁMETROS

NOMBRE_ARCHIVO = "CAE1D_01_proyecciones_15_K3.hdf5"
PATH_DATOS = "./Experimento_01/proyecciones/"
PATH_FIGURES = "./Experimento_01/figures/"

N_CHANNELS = 7
SIZE_FIG = 8
ZOOM_IMGS = SIZE_FIG / 3.6

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

reset_seeds(42)

# Cargamos la proyeccion
proyeccion = obtener_dataset(PATH_DATOS + NOMBRE_ARCHIVO, "proyeccion")
print(proyeccion.shape)

inputs = obtener_dataset(PATH_DATOS + NOMBRE_ARCHIVO, "muestras")
datosLS = obtener_dataset(PATH_DATOS + NOMBRE_ARCHIVO, "espacioLatente")

datosLS = MinMaxScaler((-1, 1)).fit_transform(datosLS)
inputs_s = MinMaxScaler((-1, 1)).fit_transform(inputs.reshape((-1, N_CHANNELS)))
inputs = inputs_s.reshape(inputs.shape)


# Generamos unos indices aleatorios para plotear
indices = poisson_disc_indices(0.4, proyeccion)

xLS = np.linspace(0, 1, datosLS.shape[-1]) * 0.3
yLS = (datosLS[indices, :] * 0.3).reshape(len(indices), -1)

# ------------------------------------------------------------------------------------------- #

# Comparación Sparklines Espacio Latente - Espacio Ambiente
fig, axes = plt.subplots(1, 2, figsize=(2*SIZE_FIG, SIZE_FIG), sharey=True, sharex=True)
fig.tight_layout()
axes[0].set_aspect('equal')
axes[1].set_aspect('equal')

axes[0].scatter(proyeccion[:, 0], proyeccion[:, 1], alpha=0.3)
axes[0].set_title("Espacio Ambiente")
axes[1].scatter(proyeccion[:, 0], proyeccion[:, 1], alpha=0.3)
axes[1].set_title("Espacio Latente")

for i in range(len(indices)):
    axes[1].plot(
        xLS + proyeccion[indices[i], 0],
        yLS[i] + proyeccion[indices[i], 1],
        color="black",
    )

    im = OffsetImage(inputs[indices[i]], zoom=ZOOM_IMGS, cmap=colorMap, norm=norm)
    img = im.get_children()[-1]
    img.set_cmap("coolwarm")

    artist = []
    ab = AnnotationBbox(
        im,
        (proyeccion[indices[i], 0], proyeccion[indices[i], 1]),
        xycoords="data",
        frameon=False,
    )

    artist.append(axes[0].add_artist(ab))

fig.savefig(PATH_FIGURES + "scatterSparklines.png")

# ------------------------------------------------------------------------------------------- #

# Figura del espacio ambiente
fig, ax = plt.subplots(1, 1, figsize=(SIZE_FIG, SIZE_FIG))
fig.tight_layout()
ax.scatter(proyeccion[:, 0], proyeccion[:, 1], alpha=0.3)
ax.set_title("Espacio ambiente")
ax.set_aspect('equal')

for i in indices:
    im = OffsetImage(inputs[i], zoom=ZOOM_IMGS, cmap=colorMap, norm=norm)
    img = im.get_children()[-1]
    img.set_cmap("coolwarm")

    artist = []
    ab = AnnotationBbox(
        im, (proyeccion[i, 0], proyeccion[i, 1]), xycoords="data", frameon=False
    )
    artist.append(ax.add_artist(ab))

fig.savefig(PATH_FIGURES + "scatterSparklines_AS.png")

# ------------------------------------------------------------------------------------------- #

# Figura del espacio latente
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
fig.tight_layout()
ax.set_aspect('equal')

ax.scatter(proyeccion[:, 0], proyeccion[:, 1], alpha=0.3)
ax.set_title("Espacio Latente")

for i in range(len(indices)):
    ax.plot(
        xLS + proyeccion[indices[i], 0],
        yLS[i] + proyeccion[indices[i], 1],
        color="black",
    )

fig.savefig(PATH_FIGURES + "scatterSparklines_LS.png")

# ------------------------------------------------------------------------------------------- #

plt.show()