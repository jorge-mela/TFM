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

import numpy as np
import matplotlib.pyplot as plt

from libs.generales import obtener_dataset


# =========================================================================================== #
# PARÁMETROS

PATH_MUESTRAS = "./Experimento_02/data/"

NOMBRE_ARCHIVO = "imagenes_muestras.hdf5"


# =========================================================================================== #
# FUNCIONES

def muestra_quiver_temporal(imgs:np.ndarray, campos:np.ndarray, idx:int=0):
    D = 150; xWin, yWin = np.meshgrid(np.arange(0, D*2, 4), np.arange(0, D*2, 4))
    fig = plt.figure()
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow(imgs[idx,i], cmap="Greys")
        plt.quiver(xWin, yWin, campos[idx,i,:,:,0], -campos[idx,i,:,:,1], color="yellow")
    
    return fig


def muestra_quiver(idx:int=0):
    D = 75; xWin, yWin = np.meshgrid(np.arange(0, D*2, 4), np.arange(0, D*2, 4))
    plt.figure()
    plt.imshow(imagenes[idx], cmap="Greys")
    plt.quiver(xWin, yWin, flechas[idx,:,:,0], -flechas[idx,:,:,1], color="yellow")


# =========================================================================================== #
# MAIN

imagenes = obtener_dataset(PATH_MUESTRAS+NOMBRE_ARCHIVO, "imagenes")
flechas = obtener_dataset(PATH_MUESTRAS+NOMBRE_ARCHIVO, "flechas")
# features = obtener_dataset(PATH_MUESTRAS+NOMBRE_ARCHIVO, "muestras")

idx = int(input("Índice a buscar: "))

f = muestra_quiver_temporal(imagenes, flechas, idx)

f.savefig()

plt.show()