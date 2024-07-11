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

NOMBRE_ARCHIVO = "ROT_CAE2D_04_proyecciones_15_K1.hdf5"
PATH_DATOS = "./Experimento_03/proyecciones/"
PATH_FIGURES = "./Experimento_03/figures/"

SIZE_FIG = 8
N_CHANNELS = 1

modo = "coordenadas"

puntos = ((-22,4), (-8,4), (-5,3))


# Generamos 20 colores para poder seleccionar hasta 20 puntos distintos
colormap = get_cmap("Set3")
aux = np.linspace(0, 1, 13)
colores =  colormap(aux) # Generamos los colores como tal


# =========================================================================================== #
# FUNCIONES

def get_image(path, zoom=1):
    return OffsetImage(plt.imread(path), zoom=zoom)


def generar_plots(data: np.ndarray, indice:int, vmin:int=None, vmax:int=None):
    
    if (vmin is not None) and (vmax is not None):
        plt.imshow(data[indice], cmap="coolwarm", vmin=vmin, vmax=vmax, origin="lower")
    else:
        plt.imshow(data[indice], cmap="coolwarm", vmin=np.percentile(data, 5), vmax=np.percentile(data, 95), origin="lower")
    
    plt.xticks(())
    norm = Normalize(vmin=vmin, vmax=vmax)
    plt.colorbar(ScalarMappable(norm=norm, cmap="coolwarm"), ticks=(-1, 0, 1))

    # plt.plot(data[indice])
    # plt.xticks(np.arange(data.shape[-1]))
    # plt.yticks(np.arange(vmin, vmax))

    return None


def generar_quivers(img: np.ndarray, quiver: np.ndarray, index:tuple):

    xWin, yWin = np.meshgrid(np.arange(0, 75*2, 4), np.arange(0, 75*2, 4)) # SALE 38x38
    plt.imshow(img[index], cmap="Greys", vmin=0, vmax=1)
    plt.quiver(xWin, yWin, quiver[index, :, :, 0], -quiver[index, :, :, 1], color="yellow")
    plt.xticks([])
    plt.yticks([])

    return None


def generar_plots_ls(data: np.ndarray, indice:int, ndims:int):
    
    plt.imshow(data[indice].reshape(1,ndims), cmap="coolwarm", vmin=-1, vmax=1)
    plt.xticks([])
    plt.yticks([])

    return None    


def generar_grafica_ls(data: np.ndarray, index:tuple, vmin:float=None, vmax:float=None):

    # nombres = ["Original"]
    if (vmin is not None) and (vmax is not None):
        bottomY = vmin
        topY = vmax
    nombres = []
    # plt.plot(np.arange(data[index[0]].size), data[index[0]])
    # for i, indx in enumerate(index[1::2]):
    for i, indx in enumerate(index):
        plt.plot(np.arange(data[indx].size), data[indx])
        nombres.append("Muestra "+ str(indx))
    # plt.legend(nombres) # La leyenda no es tan importante, no hace falta saber cuál se extravió
    plt.ylim((bottomY, topY))
    plt.xticks(np.arange(1,20,2))

    return None


def plotCollection(ax:plt.Axes, xs:np.ndarray, ys:np.ndarray=None, *args, **kwargs):

    if ys is not None:
        ax.plot(xs,ys, *args, **kwargs)
    else:
        ax.plot(xs, *args, **kwargs)

    if "label" in kwargs.keys():
        #remove duplicates
        handles, labels = ax.get_legend_handles_labels()
        newLabels, newHandles = [], []
        for handle, label in zip(handles, labels):
            if label not in newLabels:
                newLabels.append(label)
                newHandles.append(handle)

        ax.legend(newHandles, newLabels, loc="best")


# =========================================================================================== #
# MAIN

# Cargamos la proyeccion y sus datos
proyeccion = obtener_dataset(PATH_DATOS + NOMBRE_ARCHIVO, "proyeccion")

datosLS = obtener_dataset(PATH_DATOS+ NOMBRE_ARCHIVO, "espacioLatente")
print(datosLS.shape)
datosLS = MinMaxScaler((-1,1)).fit_transform(datosLS)

N_MUESTRAS = proyeccion.shape[0]

fig, axes = plt.subplots(1, 2, figsize=(2*SIZE_FIG, SIZE_FIG))
fig.tight_layout() 

axes[0].scatter(proyeccion[:,0], proyeccion[:,1], alpha=0.3, color=get_cmap("Set1")(0.222))
axes[1].set_ylim(-1, 1)
axes[1].set_xticks(np.arange(datosLS.shape[-1]))
axes[1].set_xticklabels(np.arange(1, datosLS.shape[-1]+1))

# Seleccionamos las muestras más cercanas a los puntos marcados
for i, punto in enumerate(puntos):
    centroP = np.asarray(punto).reshape(1,2)
    distancias = cdist(centroP, proyeccion).squeeze()
    indicesVecinos = np.argsort(distancias)[:30] # 30 puntos más cercanos

    axes[0].scatter(centroP[:,0], centroP[:,1], color=colores[i], marker='x')
    axes[0].scatter(proyeccion[indicesVecinos, 0], proyeccion[indicesVecinos, 1], color=colores[i], marker='x')
    plotCollection(ax=axes[1], xs=datosLS[indicesVecinos].T,  color=colores[i], label=str(punto), alpha=0.7)
    


fig.savefig(PATH_FIGURES + "PtsCercanosComparacionLS.png")



plt.show()
