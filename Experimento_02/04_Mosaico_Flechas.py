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
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, "./")
from libs.generales import obtener_dataset, reset_seeds


# =========================================================================================== #
# PARÁMETROS

NOMBRE_ARCHIVO = "ROT_CAE2D_04_proyecciones_15_K1.hdf5"
PATH_DATOS = "./Experimento_02/proyecciones/"
PATH_FIGURES = "./Experimento_02/figures/"
PATH_FLECHAS = "./Experimento_02/data/imagenes_muestras.hdf5"

N_CHANNELS = 1
SIZE_FIG = 8
N_CUADROS = 16
N_ROWS = 4
N_COLS = 4

SEED = 42


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
inputs = obtener_dataset("./Experimento_02/data/01_muestras.hdf5", "muestras")[...,0]
campoXY = obtener_dataset(PATH_FLECHAS, "campoXY")
campoUV = obtener_dataset(PATH_FLECHAS, "campoUV")
imagenes = obtener_dataset(PATH_FLECHAS, "imagenes")
print("Datos cargados")

# Generamos unos indices aleatorios para plotear
indices = poisson_disc_indices(0.4, proyeccion)

# ------------------------------------------------------------------------------------------- #

# Solo para saber dónde han caído
fig, ax = plt.subplots(1, 1, figsize=(SIZE_FIG, SIZE_FIG))
fig.tight_layout()
ax.scatter(proyeccion[:, 0], proyeccion[:, 1], alpha=0.3)
ax.scatter(proyeccion[indices[:N_CUADROS], 0], proyeccion[indices[:N_CUADROS], 1], marker='x')
ax.set_title("Puntos usados")

# ------------------------------------------------------------------------------------------- #
# Mostramos los tensores con sus coordenadas

inputs_s = MinMaxScaler((-1,1)).fit_transform(inputs.reshape((-1,1)))
inputs = inputs_s.reshape(inputs.shape).squeeze()

fig, ax = plt.subplots(N_ROWS, N_COLS, figsize=(SIZE_FIG, SIZE_FIG), sharex=True, sharey=True)
fig.tight_layout()
# fig.suptitle("Tensores de entrada")

for i, indx in enumerate(indices[:N_CUADROS]):
    ax[i//N_COLS,i%N_COLS].imshow(inputs[indx], cmap="coolwarm", norm=norm)

    titulo = f"{np.round(proyeccion[indx,0], 2), np.round(proyeccion[indx,1], 2)}"
    ax[i//N_COLS,i%N_COLS].set_title(titulo)


fig.savefig(PATH_FIGURES + "MosaicoInputs.png")

# ------------------------------------------------------------------------------------------- #
# Mostramos los campos de flechas con sus coordenadas
fig, ax = plt.subplots(N_ROWS, N_COLS, figsize=(SIZE_FIG, SIZE_FIG), sharex=True, sharey=True)
fig.tight_layout()
# fig.suptitle("Campos de Flechas")
    
for i, indx in enumerate(indices[:N_CUADROS]):
    ax[i//N_COLS,i%N_COLS].imshow(imagenes[indx], cmap="Greys", vmin=0, vmax=255)
    ax[i//N_COLS,i%N_COLS].quiver(campoXY[...,0], campoXY[...,1], campoUV[indx, ..., 0], -campoUV[indx, ..., 1], color="yellow")

    # Para ajustar los ejes a la imagen
    ax[i//N_COLS,i%N_COLS].set_xlim((0, 320))
    ax[i//N_COLS,i%N_COLS].set_ylim((0, 320))

    titulo = f"{np.round(proyeccion[indx,0], 2), np.round(proyeccion[indx,1], 2)}"
    ax[i//N_COLS,i%N_COLS].set_title(titulo)


fig.savefig(PATH_FIGURES + "MosaicoFlechas.png")

# ------------------------------------------------------------------------------------------- #
# Mostramos los tensores sobre las imágenes
fig, ax = plt.subplots(N_ROWS, N_COLS, figsize=(SIZE_FIG, SIZE_FIG), sharex=True, sharey=True)
fig.tight_layout()
# fig.suptitle("Campos de Flechas")


    
for i, indx in enumerate(indices[:N_CUADROS]):
    ax[i//N_COLS,i%N_COLS].imshow(imagenes[indx], cmap="Greys", vmin=0, vmax=255)

    ampliada = inputs[indx].repeat(40, axis=-2).repeat(40, axis=-1)
    ax[i//N_COLS,i%N_COLS].imshow(ampliada, cmap="coolwarm", norm=norm, alpha=0.6)

    # Para ajustar los ejes a la imagen
    ax[i//N_COLS,i%N_COLS].set_xlim((0, 320))
    ax[i//N_COLS,i%N_COLS].set_ylim((0, 320))

    titulo = f"{np.round(proyeccion[indx,0], 2), np.round(proyeccion[indx,1], 2)}"
    ax[i//N_COLS,i%N_COLS].set_title(titulo)


fig.savefig(PATH_FIGURES + "MosaicoHeatmap.png")

# ------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------- #
# Mostramos los tensores sobre las imágenes
fig, ax = plt.subplots(N_ROWS, N_COLS, figsize=(SIZE_FIG, SIZE_FIG), sharex=True, sharey=True)
fig.tight_layout()
# fig.suptitle("Campos de Flechas")


    
for i, indx in enumerate(indices[:N_CUADROS]):
    ax[i//N_COLS,i%N_COLS].imshow(imagenes[indx], cmap="Greys", vmin=0, vmax=255)

    ampliada = inputs[indx].repeat(40, axis=-2).repeat(40, axis=-1)
    ax[i//N_COLS,i%N_COLS].imshow(ampliada, cmap="coolwarm", norm=norm, alpha=0.4)

    ax[i//N_COLS,i%N_COLS].quiver(campoXY[...,0], campoXY[...,1], campoUV[indx, ..., 0], -campoUV[indx, ..., 1], color="yellow")

    # Para ajustar los ejes a la imagen
    ax[i//N_COLS,i%N_COLS].set_xlim((0, 320))
    ax[i//N_COLS,i%N_COLS].set_ylim((0, 320))

    titulo = f"{np.round(proyeccion[indx,0], 2), np.round(proyeccion[indx,1], 2)}"
    ax[i//N_COLS,i%N_COLS].set_title(titulo)


fig.savefig(PATH_FIGURES + "MosaicoHeatmapFlechas.png")

# ------------------------------------------------------------------------------------------- #



plt.show()
