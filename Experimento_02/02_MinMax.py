"""
+-------------------------------------------------------------------------------------------+
| Script que se dedica exclusivamente al limpiado y escalado de los datos.                  |
| En este caso, se hace un clipping a los percentiles 1 y 99 de cada canal y un escalado    |
| MinMax.                                                                                   |
|                                                                                           |
| Autor: Jorge Menéndez Lagunilla                                                           |
| Fecha: 11/2023                                                                            |
|                                                                                           |
+-------------------------------------------------------------------------------------------+
"""


# =========================================================================================== #
# LIBRERIAS
import sys

import h5py
import numpy as np
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, "./")
from libs.generales import obtener_dataset

# =========================================================================================== #
# PARÁMETROS

PATH_DATOS = "./Experimento_02/data/"
PATH_RESULT = PATH_DATOS

NOMBRE_DATOS = "01_muestras.hdf5" # "01_muestras.hdf5"
NOMBRE_RESULT = "02_muestras_THR.hdf5" if ("THR" in NOMBRE_DATOS) else "02_muestras.hdf5"


# =========================================================================================== #
# FUNCIONES




# =========================================================================================== #
# MAIN

print("Extraemos los datos del archivo")
muestras = obtener_dataset(PATH_DATOS + NOMBRE_DATOS, "muestras")

print("Comenzamos la ejecución del programa")

# Eliminamos los outliers
percentilINF = np.percentile(muestras.reshape(-1,5), 1, axis=0)
percentilSUP = np.percentile(muestras.reshape(-1,5), 99, axis=0)

for i in range(muestras.shape[-1]):
    slicing = [slice(None)] * len(muestras.shape)
    slicing[-1] = i
    # Esas dos líneas son equivalentes a hacer muestras[...,i]
    aux = muestras[tuple(slicing)]
    aux[aux< percentilINF[i]] = percentilINF[i]
    aux[aux> percentilSUP[i]] = percentilSUP[i]
    muestras[tuple(slicing)] = aux

# Escalamos los datos
scaler1 = MinMaxScaler()

muestras_s = scaler1.fit_transform(muestras.reshape(-1,5)).reshape(muestras.shape)

# Lo preparamos para que sea nmuestras x nfilas x ncols x ncanales
N_CANALES = muestras.shape[-1]
N_FILAS = muestras.shape[-2]
N_COLS = muestras.shape[-3]
# muestras_s = muestras_s.reshape((-1, N_FILAS, N_COLS, N_CANALES))
print("Shape muestra_s: ", muestras_s.shape)


# Generamos el dataset
print("Introducimos todas las muestras en un dataset")
f = h5py.File(PATH_RESULT + NOMBRE_RESULT , "w")
f.create_dataset("muestras", data=muestras_s)
f.close()
print("Terminamos de introducir las muestras")