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

puntos = ((-8,4), (-22,4), (0,4))

# Generamos 20 colores para poder seleccionar hasta 20 puntos distintos
colormap = get_cmap("tab20")
aux = np.linspace(0, 1, 20)
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


# =========================================================================================== #
# ERRORES

if len(puntos) != 3:
    exStr = "Solo puede introducirse grupos de 3 puntos"
    raise(exStr)


# =========================================================================================== #
# MAIN

# Cargamos la proyeccion y sus datos
proyeccion = obtener_dataset(PATH_DATOS + NOMBRE_ARCHIVO, "proyeccion")

datosLS = obtener_dataset(PATH_DATOS+ NOMBRE_ARCHIVO, "espacioLatente")
print(datosLS.shape)
datosLS = MinMaxScaler((-1,1)).fit_transform(datosLS)

N_MUESTRAS = proyeccion.shape[0]

distribucionAxes = [['A', 'B', 'B'],
                    ['A', 'C', 'C']]
fig, axes = plt.subplot_mosaic(distribucionAxes, figsize=(2*SIZE_FIG, SIZE_FIG))
fig.tight_layout() 

# Seleccionamos las muestras más cercanas a los puntos marcados
centroP0 = np.asarray(puntos[0]).reshape(1,2)
distancias0 = cdist(centroP0, proyeccion).squeeze()
indicesVecinos0 = np.argsort(distancias0)[:30] # 30 puntos más cercanos

centroP1 = np.asarray(puntos[1]).reshape(1,2)
distancias1 = cdist(centroP1, proyeccion).squeeze()
indicesVecinos1 = np.argsort(distancias1)[:30] # 30 puntos más cercanos

centroP2 = np.asarray(puntos[2]).reshape(1,2)
distancias2 = cdist(centroP2, proyeccion).squeeze()
indicesVecinos2 = np.argsort(distancias2)[:30] # 30 puntos más cercanos

# Para el scatter
i = 1
for centro, vecinos in zip((centroP0, centroP1, centroP2), (indicesVecinos0, indicesVecinos1, indicesVecinos2)):
    axes['A'].scatter(centro[:,0], centro[:,1], color=colores[i], marker='x')
    axes['A'].scatter(proyeccion[vecinos, 0], proyeccion[vecinos, 1], color=colores[i], marker='x')
    i = i+1


# Coloreamos las cajas pertenecientes al punto común
boxplot0 = axes['B'].boxplot(datosLS[indicesVecinos0], patch_artist=True, positions=(np.arange(datosLS.shape[-1])-0.125), widths=0.25)
for patch in boxplot0['boxes']:
    patch.set_facecolor(colores[1])

boxplot0 = axes['C'].boxplot(datosLS[indicesVecinos0], patch_artist=True, positions=(np.arange(datosLS.shape[-1])-0.125), widths=0.25)
for patch in boxplot0['boxes']:
    patch.set_facecolor(colores[1])


# Coloreamos las cajas del segundo punto
boxplot1 = axes['B'].boxplot(datosLS[indicesVecinos1], patch_artist=True, positions=(np.arange(datosLS.shape[-1])+0.125), widths=0.25)
for patch in boxplot1['boxes']:
    patch.set_facecolor(colores[2])


# Coloreamos las cajas del tercer punto
boxplot2 = axes['C'].boxplot(datosLS[indicesVecinos2], patch_artist=True, positions=(np.arange(datosLS.shape[-1])+0.125), widths=0.25)
for patch in boxplot2['boxes']:
    patch.set_facecolor(colores[3])


# Dejamos bonitos los ejes
axes['A'].scatter(proyeccion[:,0], proyeccion[:,1], alpha=0.3, color=get_cmap("tab20")(0))
axes['B'].set_ylim(-1, 1)
axes['B'].set_xticks(np.arange(datosLS.shape[-1]))
axes['B'].set_xticklabels(np.arange(1, datosLS.shape[-1]+1))
axes['C'].set_ylim(-1, 1)
axes['C'].set_xticks(np.arange(datosLS.shape[-1]))
axes['C'].set_xticklabels(np.arange(1, datosLS.shape[-1]+1))

fig.savefig(PATH_FIGURES + "PtsCercanosComparacionLS.png")



plt.show()
