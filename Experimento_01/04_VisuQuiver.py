"""
# =========================================================================================== #
# Script que permite la visualización de los campos de flechas (tanto de muestras con         #
# temporalidad como sin ella) sobre las imágenes de las que provienen.                        #
# Se proporciona el número de muestra en bucle hasta que se introduzca un -1, que se cerrará  #
# el programa.                                                                                #
#                                                                                             #
# Autor: Jorge Menéndez Lagunilla                                                             #
# Fecha: 02/2024                                                                              #
#                                                                                             #
# =========================================================================================== #
"""


# =========================================================================================== #
# LIBRERIAS

import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

sys.path.insert(0, "./")
from libs.generales import obtener_dataset


# =========================================================================================== #
# PARÁMETROS

PATH_MUESTRAS = "./Experimento_01/data/"
NOMBRE_ARCHIVO = "imagenes_muestras.hdf5"
PATH_PROYECCION = "./Experimento_01/proyecciones/CAE1D_01_proyecciones_15_K3.hdf5"
PATH_FIGURES = "./Experimento_01/figures/"

COORDENADA_X = 8
COORDENADA_Y = 6


# =========================================================================================== #
# FUNCIONES

def muestra_quiver_temporal(campos:np.ndarray, idx:int=0):
    # Está muy hard-coded

    D = 150
    xWin, yWin = np.meshgrid(np.arange(0, D*2, 16), np.arange(0, D*2, 16))
    fig, axes = plt.subplots(4, 4, figsize=(8,8))

    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            ax.quiver(xWin, yWin, campos[idx+24*(4*i+j), ::4, ::4, 0], -campos[idx+24*(4*i+j), ::4, ::4, 1], color="black")
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_aspect('equal', 'box')
    
    fig.tight_layout()
    
    return fig


# =========================================================================================== #
# MAIN

flechas = obtener_dataset(PATH_MUESTRAS+NOMBRE_ARCHIVO, "velocidad")
flechas = flechas.reshape(-1, flechas.shape[-3], flechas.shape[-2], flechas.shape[-1])
flechas = flechas[..., 2:] # Nos quedamos solo con las componentes U y V
print("Flechas shape: ", flechas.shape)

proyeccion = obtener_dataset(PATH_PROYECCION, "proyeccion")
print("Proyeccion shape: ", proyeccion.shape)

centroP = np.asarray((COORDENADA_X, COORDENADA_Y)).reshape(1,2)
distancias = cdist(centroP, proyeccion).squeeze()
idx = np.argsort(distancias)[0] # Cogemos el punto más cercano a las coordenadas

f = muestra_quiver_temporal(flechas, idx)

f.savefig(PATH_FIGURES + f"Quiver_{COORDENADA_X}_{COORDENADA_Y}.png")

plt.show()