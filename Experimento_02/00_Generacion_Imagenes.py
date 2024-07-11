"""
+-------------------------------------------------------------------------------------------+
| Script para generar las imágenes y quivers correspondientes a las muestras usadas.        |
|                                                                                           |
| Autor: Jorge Menéndez Lagunilla                                                           |
| Fecha: 11/2023                                                                            |
|                                                                                           |
+-------------------------------------------------------------------------------------------+
"""


# =========================================================================================== #
# LIBRERIAS

import os

import h5py
import numpy as np
from skimage.util.shape import view_as_windows
from sklearn.linear_model import Ridge

import sys
sys.path.insert(0,"./")
from libs.generales import obtener_dataset, cargar_video


# =========================================================================================== #
# PARÁMETROS

PATH_VIDEOS = "./00_comun/videos/SDHB/"
PATH_VELOCIDADES = "./00_comun/velocidades/"
NOMBRE_DSET = "video_flow"
PATH_RESULT = "./Experimento_02/data/"

WIN_SIZE = 80  # tamaño de las ventanas
N_FRAMES = 360 # Nos quedamos con los primeros N_FRAMES

# Para el recorte de los bordes que no son parte de la imagen
LIMITE_SUPERIOR = 73
LIMITE_INFERIOR = 428


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
for i, v in enumerate(allVideos):
    allVideos[i] = v.split('.')[0]
allVideos.sort()

N_VIDEOS = len(allVideos) # Número de vídeos que se procesarán

# Definimos la meshgrid para el cálculo de las regresiones
ventanas_X, ventanas_Y = np.meshgrid(np.linspace(0, WIN_SIZE*4, WIN_SIZE),
                                     np.linspace(0, WIN_SIZE*4, WIN_SIZE))


allCampos_ = []
allImagenes_ = []

# Calculamos las características para cada video
for video in allVideos:
    
    print("Extraemos la información de los campos de velocidad", video)
    campos = obtener_dataset(PATH_VELOCIDADES + video+ ".hdf5", "video_flow")
    imagenes, _ = cargar_video(PATH_VIDEOS + video + ".avi", N_FRAMES)
    # imagenes = imagenes #/ 255 # Normalizamos las imágenes
    print("Terminamos de extraer", video)
    print("Shape campos: ", campos.shape)

    # Recortamos los bordes superior e inferior que no contienen información
    campos = np.transpose(campos[:,:,LIMITE_SUPERIOR:LIMITE_INFERIOR,:], [0,2,1,3])
    imagenes = imagenes[:,LIMITE_SUPERIOR*4:LIMITE_INFERIOR*4,:]
    
    # Número de ventanas en que se divide cada fotograma
    N_WIN = int((campos.shape[1] // WIN_SIZE) * (campos.shape[2] // WIN_SIZE))

    # Reservamos memoria para los arrays
    result_campos = np.zeros((N_FRAMES, N_WIN, WIN_SIZE, WIN_SIZE, 2))
    result_imagenes = np.zeros((N_FRAMES, N_WIN, WIN_SIZE*4, WIN_SIZE*4), dtype=np.int16)

    print("Comenzamos la partición de los fotogramas")
    for i in range(N_FRAMES):
        ventanas_U = np.vstack(view_as_windows(campos[i, :, :, 2], WIN_SIZE, WIN_SIZE))
        ventanas_V = np.vstack(view_as_windows(campos[i, :, :, 3], WIN_SIZE, WIN_SIZE))
        ventanas_frame = np.vstack(view_as_windows(imagenes[i], WIN_SIZE*4, WIN_SIZE*4))

        for j in range(N_WIN):
            result_campos[i,j] = np.stack((ventanas_U[j], ventanas_V[j]), axis=-1)
            result_imagenes[i,j] = ventanas_frame[j]

    print("Terminado...", video)
    print("\n")

    allCampos_.append(result_campos)
    allImagenes_.append(result_imagenes)


# Seguimos procesando los datos de todos los vídeos en conjunto
camposRef = np.stack((ventanas_X, ventanas_Y), axis=-1)
allCampos = np.asarray(allCampos_).reshape((-1, WIN_SIZE, WIN_SIZE, 2))
allImagenes = np.asarray(allImagenes_).reshape(-1, 4*WIN_SIZE, 4*WIN_SIZE)

print("Introducimos todas las muestras en un dataset")
f = h5py.File(PATH_RESULT + "imagenes_muestras.hdf5" , "w")
f.create_dataset("campoXY", data=camposRef, compression="gzip")
f.create_dataset("campoUV", data=allCampos, compression="gzip")
f.create_dataset("imagenes", data=allImagenes, dtype=np.int16, compression="gzip")
f.close()
print("Terminamos de introducir las muestras")