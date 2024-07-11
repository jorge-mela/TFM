"""
# =========================================================================================== #
# A partir de los campos de velocidad extraídos por el método de Gunnar-Farbenback usado en   #
# el código de Ana se procederá a extraer por técnicas de regresión las características       #
# (features) de la forma más parecida posible al procesamiento de Ana.                        #
#                                                                                             #
# Las características que se van a extraer son:                                               #
#           · Rotacional                                                                      #
#           · Divergencia                                                                     #
#           · Estiramiento en X                                                               #
#           · Estiramiento en Y                                                               #
#           · Cortadura en XY                                                                 #
#                                                                                             #
# Se para una vez se han obtenido las características. En este caso que la espacialidad es    #
# importante, se guardarán con el formato especial.                                           #
#                                                                                             #
# Autor: Jorge Menéndez Lagunilla                                                             #
# Fecha: 11/2023                                                                              #
#                                                                                             #
# =========================================================================================== #
"""


# =========================================================================================== #
# LIBRERIAS

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from skimage.util.shape import view_as_windows
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from libs.generales import obtener_dataset

# =========================================================================================== #
# PARÁMETROS

PATH_VELOCIDADES = "./00_comun/velocidades/"
NOMBRE_DSET = "video_flow"
PATH_RESULT = "./Experimento_01/data/"

WIN_SIZE = 75  # tamaño de las ventanas
N_FRAMES = 360 # Nos quedamos con los primeros N_FRAMES

# Para el recorte de los bordes que no son parte de la imagen
LIMITE_SUPERIOR = 73
LIMITE_INFERIOR = 428

NCH = 16 # Número de frames para tener en cuenta la temporalidad
S = 1 # Saltos de frames entre una muestra y la siguiente


# =========================================================================================== #
# FUNCIONES

def calcula_features(
    u_data: np.ndarray,
    v_data: np.ndarray,
    x_data: np.ndarray,
    y_data: np.ndarray, 
):
    """
    Cálculo de las características provenientes del estudio del campo vectorial.
    ## Parámetros
    - u_data: Componente U del campo de velocidad como np.array
    - v_data: Componente V del campo de velocidad como np.array
    - x_data: Coordenada X del punto como np.array
    - y_data: Coordenada Y del punto como np.array
    """

    # Copiamos los datos para no modificar el dataset
    vx = u_data.copy()
    vy = v_data.copy()
    xx = x_data.copy()
    xy = y_data.copy()

    # Cambiamos los NaN por 0s
    vx[np.isnan(vx)] = 0
    vy[np.isnan(vy)] = 0

    lr = Ridge(alpha=0.2)

    # rot=np.zeros(shape=(vx.shape[0]))
    # div=np.zeros(shape=(vx.shape[0]))
    # bias=np.zeros(shape=(vx.shape[0], 2))

    x = np.asarray([xx.ravel(), xy.ravel()])
    v = np.asarray([vx.ravel(), vy.ravel()])

    lr.fit(x.T, v.T)  # Entrenamos el modelo
    A = lr.coef_  # Obtenemos la matriz A
    B = lr.intercept_  # Obtenemos el bias

    rot = A[1, 0] - A[0, 1]
    div = A[0, 0] + A[1, 1]
    bias = B
    Ex = A[0, 0]
    Ey = A[1, 1]
    Txy = (A[0, 1] + A[1, 0])/2

    return (rot, div, bias, Ex, Ey, Txy)


# =========================================================================================== #
# MAIN

allVideos = [f for f in os.listdir(PATH_VELOCIDADES) if not f.startswith(".")]
allVideos.sort()

N_VIDEOS = len(allVideos) # Número de vídeos que se procesarán

allRot_ = []
allDiv_ = []
allEx_ = []
allEy_ = []
allTxy_ = []
allBias_ = []
allCampos_ = []

# Definimos la meshgrid para el cálculo de las regresiones
ventanas_X, ventanas_Y = np.meshgrid(np.linspace(0, WIN_SIZE*4, WIN_SIZE),
                                     np.linspace(0, WIN_SIZE*4, WIN_SIZE))

# Calculamos las características para cada video
for video in allVideos:
    
    print("Extraemos la información de los campos de velocidad", video)
    campos = obtener_dataset(PATH_VELOCIDADES + video, "video_flow")
    print("Terminamos de extraer la información", video)
    print("Shape campos: ", campos.shape)

    # Recortamos los bordes superior e inferior que no contienen información
    campos = np.transpose(campos[:,:,LIMITE_SUPERIOR:LIMITE_INFERIOR,:], [0,2,1,3])

    # Número de ventanas en que se divide cada fotograma
    N_WIN = int((campos.shape[1] // WIN_SIZE) * (campos.shape[2] // WIN_SIZE))

    # Reservamos memoria para los arrays
    result_rot = np.zeros((N_FRAMES, N_WIN))
    result_div = np.zeros((N_FRAMES, N_WIN))
    result_Ex = np.zeros((N_FRAMES, N_WIN))
    result_Ey = np.zeros((N_FRAMES, N_WIN))
    result_Txy = np.zeros((N_FRAMES, N_WIN))
    result_bias = np.zeros((N_FRAMES, N_WIN, 2))
    result_campos = np.zeros((N_FRAMES, N_WIN, WIN_SIZE, WIN_SIZE, 4))
    # print(result_rot.shape) # TESTING

    print("Comenzamos la extracción de características")
    for i in range(N_FRAMES):
        # ventanas_X = np.vstack(view_as_windows(campos[i, :, :, 0], WIN_SIZE, WIN_SIZE))
        # ventanas_Y = np.vstack(view_as_windows(campos[i, :, :, 1], WIN_SIZE, WIN_SIZE))
        ventanas_U = np.vstack(view_as_windows(campos[i, :, :, 2], WIN_SIZE, WIN_SIZE))
        ventanas_V = np.vstack(view_as_windows(campos[i, :, :, 3], WIN_SIZE, WIN_SIZE))

        # print(ventanas_U.shape) # TESTING

        for j in range(N_WIN):
            rotacional, divergencia, bias, Ex, Ey, Txy = calcula_features(
                ventanas_U[j], ventanas_V[j], ventanas_X, ventanas_Y
            )
            result_rot[i,j] = rotacional
            result_div[i,j] = divergencia
            result_Ex [i,j] = Ex
            result_Ey [i,j] = Ey
            result_Txy [i,j] = Txy
            result_bias[i,j] = bias

            result_campos[i,j] = np.stack((ventanas_X, ventanas_Y, ventanas_U[j], ventanas_V[j]), axis=-1)

        # print("iter", i)

    # Almacenamos los datos idndividuales de cada video por si acaso
    # print("Almacenamos los datos...", video)

    # f = h5py.File(PATH_RESULT + str(WIN_SIZE) + "_" + video[:-5] + ".hdf5", "w")

    # dset1 = f.create_dataset("rotacional", data=result_rot, compression="gzip")
    # dset2 = f.create_dataset("divergencia", data=result_div, compression="gzip")
    # dset3 = f.create_dataset("Ex", data=result_Ex, compression="gzip")
    # dset4 = f.create_dataset("Ey", data=result_Ey, compression="gzip")
    # dset5 = f.create_dataset("Txy", data=result_Txy, compression="gzip")
    # dset6 = f.create_dataset("bias", data=result_bias, compression="gzip")
    # dset7 = f.create_dataset(NOMBRE_DSET, data=result_campos, compression="gzip")

    # f.close()

    print("Terminado...", video)
    print("\n")

    allRot_.append(result_rot)
    allDiv_.append(result_div)
    allEx_.append(result_Ex)
    allEy_.append(result_Ey)
    allTxy_.append(result_Txy)
    allBias_.append(result_bias)

    allCampos_.append(result_campos)

# Seguimos procesando los datos de todos los vídeos en conjunto
    
allRot = np.asarray(allRot_)
allDiv = np.asarray(allDiv_)
allEx = np.asarray(allEx_)
allEy = np.asarray(allEy_)
allTxy = np.asarray(allTxy_)
allBias = np.asarray(allBias_)

# Juntamos todas las características bajo una misma variable
muestras = np.stack(
    (allRot, allDiv, allEx, allEy, allTxy, allBias[:,:,:,0], allBias[:,:,:,1]),
    axis=-1
)

N_CANALES = muestras.shape[-1]

muestras_w_ = []

for i in range(N_VIDEOS):  # vídeo
    for j in range(0, N_FRAMES - NCH, S):  # fotograma
        for k in range(N_WIN):  # ventana
            muestras_w_.append((muestras[i, j : j+NCH, k, :]))

muestras_w = np.asarray(muestras_w_)
muestras_w = muestras_w.reshape(N_VIDEOS, N_FRAMES-NCH, N_WIN, NCH, N_CANALES)

print("Introducimos todas las muestras en un dataset")
f = h5py.File(PATH_RESULT + "01_muestras.hdf5" , "w")
f.create_dataset("muestras", data=muestras_w)
f.close()
print("Terminamos de introducir las muestras")