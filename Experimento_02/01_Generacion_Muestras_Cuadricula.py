"""
# =========================================================================================== #
# A partir de los campos de velocidad extraídos por el método de Gunnar-Farbenback usado en   #
# el código de Ana se procederá a extraer por técnicas de regresión el valor del rotacional   #
# intentando obtener unos datos con formato similar a los de Ana.                             #
# En este caso se dividirá el campo de velocidad en ventanas de 80x80 (a diferencia de las    #
# ventanas de 75x75) y estas se dividirán a su vez en 64 subventanas de 10x10 (distribuidas   #
# dentro de la primera ventana en una matriz de 8x8 para mantener la información espacial)    #
# de las que se obtendrá un valor de rotacional. De esta manera tendremos 64 valores de       #
# rotacional por ventana, teniendo una menor agregación de los datos.                         #
#                                                                                             #
# Autor: Jorge Menéndez Lagunilla                                                             #
# Fecha: 12/2023                                                                              #
#                                                                                             #
# =========================================================================================== #
"""

# =========================================================================================== #
# LIBRERIAS

import os
import sys

import cv2
import h5py
import numpy as np
from skimage.util.shape import view_as_windows
from sklearn.linear_model import Ridge

sys.path.insert(0, "./")
from libs.generales import obtener_dataset

# =========================================================================================== #
# PARÁMETROS

PATH_VELOCIDADES = "./00_comun/velocidades/"
PATH_RESULT = "./Experimento_02/data/"

N_FRAMES = 360

LIMITE_INFERIOR = 428
LIMITE_SUPERIOR = 73

THRESHOLDING = False
UMBRAL_FLECHAS = 0.5
REGWIN_SIZE = 10
GROUPWIN_SIZE = 8


# =========================================================================================== #
# FUNCIONES

def calcula_features(
    u_data: np.ndarray,
    v_data: np.ndarray,
    x_data: np.ndarray,
    y_data: np.ndarray,
    winsize: int,
    vertsize: int,
    horzsize: int,
):
    """Calcula las características deseadas por regresión lineal en ventanas.

    Args:
        u_data (np.ndarray): Componente U del campo de velocidad como np.array
        v_data (np.ndarray): Componente V del campo de velocidad como np.array
        x_data (np.ndarray): Coordenada X del punto como np.array
        y_data (np.ndarray): Coordenada Y del punto como np.array
        winsize (int): Tamaño de la ventana a utilizar para el cálculo de la regresión
        vertsize (int): Número de subventanas en la vertical
        horzsize (int): Número de subventanas en la horizontal

    Returns:
        Tuple: Valores de las distintas características
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

    uwin = view_as_windows(vx, winsize, winsize)
    vwin = view_as_windows(vy, winsize, winsize)
    xwin = xx
    ywin = xy

    rot_matrix = np.zeros((vertsize, horzsize))  # Reservamos memoria
    div_matrix = np.zeros((vertsize, horzsize))  # Reservamos memoria
    exx_matrix = np.zeros((vertsize, horzsize))  # Reservamos memoria
    eyy_matrix = np.zeros((vertsize, horzsize))  # Reservamos memoria
    exy_matrix = np.zeros((vertsize, horzsize))  # Reservamos memoria

    for i in range(vertsize):
        for j in range(horzsize):
            # x = np.asarray([xwin[i, j].ravel(), ywin[i, j].ravel()])
            # v = np.asarray([uwin[i, j].ravel(), vwin[i, j].ravel()])
            x = np.asarray([xwin.ravel(), ywin.ravel()])
            v = np.asarray([uwin[i, j].ravel(), vwin[i, j].ravel()])

            lr.fit(x.T, v.T)  # Calculamos la regresión
            A = lr.coef_  # Obtenemos la matriz A
            rot = A[1, 0] - A[0, 1]
            div = A[0, 0] + A[1, 1]
            exy = (A[1, 0] + A[0, 1]) / 2

            rot_matrix[i, j] = rot
            div_matrix[i, j] = div
            exx_matrix[i, j] = A[0, 0]
            eyy_matrix[i, j] = A[1, 1]
            exy_matrix[i, j] = exy

    # for k in range(vertsize * horzsize):
    #     x = np.asarray([xwin[k].ravel(), ywin[k].ravel()])
    #     v = np.asarray([uwin[k].ravel(), vwin[k].ravel()])

    #     lr.fit(x.T, v.T)  # Entrenamos el modelo
    #     A = lr.coef_  # Obtenemos la matriz A

    #     rot = A[1, 0] - A[1, 0]
    #     div = A[0, 0] + A[1, 1]
    #     exy = (A[1, 0] + A[1, 0]) / 2

    #     rot_matrix[k // vertsize, k % horzsize] = rot
    #     div_matrix[k // vertsize, k % horzsize] = div
    #     exx_matrix[k // vertsize, k % horzsize] = A[0, 0]
    #     eyy_matrix[k // vertsize, k % horzsize] = A[1, 1]
    #     exy_matrix[k // vertsize, k % horzsize] = exy

    return (rot_matrix, div_matrix, exx_matrix, eyy_matrix, exy_matrix)


# =========================================================================================== #
# MAIN

allVideos = [f for f in os.listdir(PATH_VELOCIDADES) if not f.startswith(".")]
allVideos.sort()

campoX, campoY = np.meshgrid(
    np.arange(0, REGWIN_SIZE * 4, 4), np.arange(0, REGWIN_SIZE * 4, 4)
)


rotacional_ = []
divergencia_ = []
Exx_ = []
Eyy_ = []
Exy_ = []

for video in allVideos:
    print("Extraemos la información de los campos de velocidad", video)
    campos = obtener_dataset(PATH_VELOCIDADES + video, "video_flow")
    print("Shape campos: ", campos.shape)
    print("Terminamos de extraer la información", video)

    # Recortamos y trasponemos los campos para ponerlos como queremos
    # campoX = campos[:N_FRAMES, :, LIMITE_SUPERIOR:LIMITE_INFERIOR, 0]
    # campoY = campos[:N_FRAMES, :, LIMITE_SUPERIOR:LIMITE_INFERIOR, 1]
    campoU = campos[:N_FRAMES, :, LIMITE_SUPERIOR:LIMITE_INFERIOR, 2]
    campoV = campos[:N_FRAMES, :, LIMITE_SUPERIOR:LIMITE_INFERIOR, 3]

    print("Shape componente U: ", campoU.shape)

    # Cambiamos el eje vertical por el horizontal
    # campoX = np.transpose(campoX, [0, 2, 1])
    # campoY = np.transpose(campoY, [0, 2, 1])
    campoU = np.transpose(campoU, [0, 2, 1])
    campoV = np.transpose(campoV, [0, 2, 1])

    # ------------------------------------------------------------------------------------------- #
    # UMBRALIZADO
    if THRESHOLDING:
        # Hacemos un umbralizado en función de la magnitud de las flechas para eliminar el ruido
        mag, _ = cv2.cartToPolar(campoU, campoV)
        mag[mag < UMBRAL_FLECHAS] = 0
        campoU[mag == 0] = 0
        campoV[mag == 0] = 0
        mag[mag >= UMBRAL_FLECHAS] = 1
    # ------------------------------------------------------------------------------------------- #

    print("Shape componente U: ", campoU.shape)

    # Calculamos los gradientes mediante regresión lineal
    # Calculamos el número de ventanas en las que se divide el frame para reservar memoria
    N_WIN_V = campoU[0].shape[0] // REGWIN_SIZE
    N_WIN_H = campoU[0].shape[1] // REGWIN_SIZE

    # Reservamos memoria para los arrays
    rotacional = np.zeros((N_FRAMES, N_WIN_V, N_WIN_H))
    divergencia = np.zeros((N_FRAMES, N_WIN_V, N_WIN_H))
    Exx = np.zeros((N_FRAMES, N_WIN_V, N_WIN_H))
    Eyy = np.zeros((N_FRAMES, N_WIN_V, N_WIN_H))
    Exy = np.zeros((N_FRAMES, N_WIN_V, N_WIN_H))

    for frame in range(N_FRAMES):
        (
            rotacional[frame],
            divergencia[frame],
            Exx[frame],
            Eyy[frame],
            Exy[frame],
        ) = calcula_features(
            campoU[frame],
            campoV[frame],
            campoX,  # [frame]
            campoY,  # [frame]
            REGWIN_SIZE,
            N_WIN_V,
            N_WIN_H,
        )
        print(frame)

    rotacional_.append(rotacional)
    divergencia_.append(divergencia)
    Exx_.append(Exx)
    Eyy_.append(Eyy)
    Exy_.append(Exy)

    # nombreArchivo = video.split(".")[0] + "_" + str(REGWIN_SIZE) + ".hdf5"
    # if THRESHOLDING:
    #     nombreArchivo = nombreArchivo.split(".")[0] + "_umbralizado" + ".hdf5"

    # print("Generamos el dataset", nombreArchivo)
    # f = h5py.File(PATH_RESULT + nombreArchivo, "w")
    # dset1 = f.create_dataset("rotacional", data=rotacional, compression="gzip")
    # dset2 = f.create_dataset("divergencia", data=divergencia, compression="gzip")
    # dset3 = f.create_dataset("Xestiramiento", data=Exx, compression="gzip")
    # dset4 = f.create_dataset("Yestiramiento", data=Eyy, compression="gzip")
    # dset5 = f.create_dataset("Testriamiento", data=Exy, compression="gzip")
    # f.close()

    # print("Terminado...", video)
    # print("\n")

# Juntamos las características de todos los vídeos en una sola variable
allRotacional = np.asarray(rotacional_).reshape(-1, N_WIN_V, N_WIN_H)
allDivergencia = np.asarray(divergencia_).reshape(-1, N_WIN_V, N_WIN_H)
allExx = np.asarray(Exx_).reshape(-1, N_WIN_V, N_WIN_H)
allEyy = np.asarray(Eyy_).reshape(-1, N_WIN_V, N_WIN_H)
allExy = np.asarray(Exy_).reshape(-1, N_WIN_V, N_WIN_H)


auxRot_ = []
auxDiv_ = []
auxExx_ = []
auxEyy_ = []
auxExy_ = []

for i in range(allRotacional.shape[0]):
    auxRot_.append(view_as_windows(allRotacional[i], GROUPWIN_SIZE, GROUPWIN_SIZE))
    auxDiv_.append(view_as_windows(allDivergencia[i], GROUPWIN_SIZE, GROUPWIN_SIZE))
    auxExx_.append(view_as_windows(allExx[i], GROUPWIN_SIZE, GROUPWIN_SIZE))
    auxEyy_.append(view_as_windows(allEyy[i], GROUPWIN_SIZE, GROUPWIN_SIZE))
    auxExy_.append(view_as_windows(allExy[i], GROUPWIN_SIZE, GROUPWIN_SIZE))

muestras = np.stack((auxRot_, auxDiv_, auxExx_, auxEyy_, auxExy_), axis=-1)
muestras = muestras.reshape(-1, GROUPWIN_SIZE, GROUPWIN_SIZE, 5)


nombreArchivo = "01_muestras.hdf5"

if THRESHOLDING:
    nombreArchivo = "01_muestras_THR.hdf5"

print("Generamos el dataset", nombreArchivo)
f = h5py.File(PATH_RESULT + nombreArchivo, "w")
dset1 = f.create_dataset("muestras", data=muestras, compression="gzip")

f.close()

print("Terminado...", nombreArchivo)
print("\n")


# print("Generamos el dataset completo")
# f = h5py.File(
#     PATH_RESULT
#     + str(WIN_SIZE)
#     + "_"
#     + str(SUBWIN_SIZE)
#     + "_"
#     + "Divergencia2D_SDHB.hdf5",
#     "w",
# )

# dset1 = f.create_dataset("divergencia", data=result_div, compression="gzip")
# dset2 = f.create_dataset("binarias", data=bin, compression="gzip")
# dset3 = f.create_dataset("campoU", data=campoU, compression="gzip")
# dset4 = f.create_dataset("campoV", data=campoV, compression="gzip")
# dset5 = f.create_dataset("campoX", data=campoX, compression="gzip")
# dset6 = f.create_dataset("campoY", data=campoY, compression="gzip")
# dset7 = f.create_dataset("magnitud", data=mag, compression="gzip")

# f.close()

print("Terminamos el dataset completo")
