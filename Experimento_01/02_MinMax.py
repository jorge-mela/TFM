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

import h5py
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from libs.generales import obtener_dataset


# =========================================================================================== #
# PARÁMETROS

PATH_DATOS = "./Experimento_01/data/"
PATH_RESULT = PATH_DATOS

NOMBRE_DATOS = "01_muestras.hdf5"
NOMBRE_RESULT = "02_muestras.hdf5"


# =========================================================================================== #
# FUNCIONES




# =========================================================================================== #
# MAIN

print("Extraemos los datos del archivo")
muestras = obtener_dataset(PATH_DATOS + NOMBRE_DATOS, "muestras")

print("Comenzamos la ejecución del programa")

# Eliminamos los outliers
percentilINF = np.percentile(muestras.reshape(-1,7), 5, axis=0)
percentilSUP = np.percentile(muestras.reshape(-1,7), 95, axis=0)

for i in range(muestras.shape[-1]):
    slicing = [slice(None)] * len(muestras.shape)
    slicing[-1] = i
    aux = muestras[tuple(slicing)]
    aux[aux< percentilINF[i]] = percentilINF[i]
    aux[aux> percentilSUP[i]] = percentilSUP[i]
    muestras[tuple(slicing)] = aux

# Escalamos los datos
scaler1 = MinMaxScaler()

muestras_s = scaler1.fit_transform(muestras.reshape(-1,7)).reshape(muestras.shape)


print("Introducimos todas las muestras en un dataset")
f = h5py.File(PATH_RESULT + NOMBRE_RESULT , "w")
f.create_dataset("muestras", data=muestras_s)
f.close()
print("Terminamos de introducir las muestras")