"""
# =========================================================================================== #
# Script que se dedica exclusivamente al limpiado y escalado de los datos.                    #
# En este caso, se hace un clipping a los percentiles 1 y 99 de cada canal y un escalado      #
# MinMax.                                                                                     #
#                                                                                             #
# Autor: Jorge Menéndez Lagunilla                                                             #
# Fecha: 11/2023                                                                              #
#                                                                                             #
# =========================================================================================== #
"""


# =========================================================================================== #
# LIBRERIAS

import h5py
import matplotlib.pyplot as plt
import numpy as np
from skimage.util.shape import view_as_windows
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from libs.generales import obtener_dataset

# =========================================================================================== #
# PARÁMETROS

PATH_DATOS = "/home/jorgemel/motilidad2/atemporal/fft/data/"
PATH_RESULT = "/home/jorgemel/motilidad2/atemporal/fft/data/"


# =========================================================================================== #
# FUNCIONES




# =========================================================================================== #
# MAIN

print("Extraemos los datos del archivo")
muestras = obtener_dataset(PATH_DATOS + "01_muestras.hdf5", "muestras")

print("Comenzamos la ejecución del programa")

# Eliminamos los outliers
percentilSUP = np.percentile(muestras.reshape(-1,muestras.shape[-1]), 99, axis=0)

for i in range(muestras.shape[-1]):
    slicing = [slice(None)] * len(muestras.shape)
    slicing[-1] = i
    aux = muestras[tuple(slicing)]
    aux[aux> percentilSUP[i]] = percentilSUP[i]
    muestras[tuple(slicing)] = aux

# Escalamos los datos
scaler1 = StandardScaler()

muestras_s = scaler1.fit_transform(muestras.reshape(-1,1)).reshape(muestras.shape)


print("Introducimos todas las muestras en un dataset")
f = h5py.File(PATH_RESULT + "02_muestras.hdf5" , "w")
f.create_dataset("muestras", data=muestras_s)
f.close()
print("Terminamos de introducir las muestras")