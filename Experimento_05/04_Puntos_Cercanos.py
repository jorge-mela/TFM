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

# ESTO NO ACABA DE ESTAR BIEN, LOS COLORES NO ESTÁN BIEN ESCALADOS

# =========================================================================================== #
# LIBRERIAS

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler

from libs.generales import obtener_dataset


# =========================================================================================== #
# PARÁMETROS

# PATH_IMAGENES = 
# PATH_FLECHAS = 

MODELO = "Conv2relu"
LS_DIMS = "20"

PATH_DATOS = "/home/jorgemel/motilidad2/atemporal/fft/proyecciones/"
ARCHIVO = MODELO + "_proyecciones_" + LS_DIMS +".hdf5"

PATH_MODELOS = "/home/jorgemel/motilidad2/atemporal/fft/models/"

ENCODER = MODELO + "_encoder_" + LS_DIMS + ".keras" 
DECODER = MODELO + "_decoder_" + LS_DIMS + ".keras"


m = input("c/i: ")
if m=="c":
    modo = "coordenadas"
elif m=="i":
    modo = "aleatorio"

# =========================================================================================== #
# FUNCIONES

def get_image(path, zoom=1):
    return OffsetImage(plt.imread(path), zoom=zoom)


def generar_plots(data: np.ndarray, indice:int, vmin:int, vmax:int):
    
    plt.imshow(data[indice], cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
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
    
    plt.imshow(data[indice].reshape(1,ndims), cmap="viridis", vmin=data.min(), vmax=data.max())
    plt.xticks([])
    plt.yticks([])

    return None    


def generar_grafica_ls(data: np.ndarray, index:tuple):

    nombres = ["Original"]
    plt.plot(np.arange(data[index[0]].size), data[index[0]], color="orange")
    for i, indx in enumerate(index[1:6]):
        plt.plot(np.arange(data[indx].size), data[indx], color="blue")
        nombres.append("Vecino "+ str(i))
    plt.legend(nombres)

    return None


# =========================================================================================== #
# DATOS

# Cargamos la proyeccion y sus datos
proyeccion = obtener_dataset(PATH_DATOS + ARCHIVO, "proyeccion")
print("Shape proyeccion: ", proyeccion.shape)

datos = obtener_dataset(PATH_DATOS + ARCHIVO, "muestras")
datos = datos.squeeze()
print("Shape muestras: ", datos.shape)

# # Cargamos las imágenes y las flechas
# imagenes = obtener_dataset(PATH_DATOS, "imagenes")
# imagenes = imagenes / 255  # Normalizamos
# imagenes = imagenes.squeeze()
# print("Shape imagenes: ", imagenes.shape)

# flechas =  obtener_dataset(PATH_DATOS, "flechas")
# flechas = flechas.squeeze()
# print("Shape flechas: ", flechas.shape)


# =========================================================================================== #
# MAIN

# Generamos un índice aleatorios para plotear
# N_VIDEOS = proyeccion.shape[0]
# N_FRAMES = proyeccion.shape[1]
# N_WIN = proyeccion.shape[2]
# N_MUESTRAS_VIDEO = N_FRAMES * N_WIN
N_MUESTRAS = proyeccion.shape[0]
N_CANALES = datos.shape[-1]

encoder = load_model(PATH_MODELOS + ENCODER)
decoder = load_model(PATH_MODELOS + DECODER)

# datosEspacioLatente = encoder.predict(datos.reshape(N_MUESTRAS, 10, 10, 1))
inputShape = list(encoder.input_shape)
inputShape[0] = -1
datosEspacioLatente = encoder.predict(datos.reshape(inputShape))

if len(datosEspacioLatente.shape) > 2:
    datosEspacioLatente2 = datosEspacioLatente.reshape((N_MUESTRAS, -1))
else:
    datosEspacioLatente2 = datosEspacioLatente

latentDims = datosEspacioLatente2.shape[1]
print("Shape espacio latente: ", datosEspacioLatente.shape)
datosDecodificados = decoder.predict(datosEspacioLatente) #.reshape(datos.shape)
print("Shape de datos decodificados: ", datosDecodificados.shape)

datosEspacioLatente2 = MinMaxScaler((0,1)).fit_transform(datosEspacioLatente2)

if modo == "aleatorio":
    # indiceCentro = np.random.randint(low=0, high=N_MUESTRAS)
    indiceCentro = int(input(f"Introducir valor hasta {N_MUESTRAS-1}: "))
    proyeccionPlana = proyeccion.reshape(-1, 2)
    centroP = proyeccionPlana[indiceCentro].reshape(1,2)
    distancias = cdist(centroP, proyeccionPlana).squeeze()
    indicesVecinos = np.argsort(distancias)[:12] # Cogemos los 11 vecinos más cercanos

if modo == "coordenadas":
    x = float(input("Coordenada X:"))
    y = float(input("Coordenada Y:"))
    indiceCentro = np.random.randint(low=0, high=N_MUESTRAS)
    proyeccionPlana = proyeccion.reshape(-1,2)
    centroP = np.asarray((x,y)).reshape(1,2)
    distancias = cdist(centroP, proyeccionPlana).squeeze()
    indicesVecinos = np.argsort(distancias)[:12] # Cogemos los 11 vecinos más cercanos
    
# # Figuras de los datos
f = plt.figure()
plt.suptitle("Datos Brutos")
for i, indx in enumerate(indicesVecinos):

    ax = plt.subplot(3, 4, i+1)
    if i==0:
        ax.title.set_text("Muestra centrada")
    else:
        titulo = f"Indice {indx}"
        ax.title.set_text(titulo)

    generar_plots(datos, indx, 0, 1)

f.savefig("/home/jorgemel/capturasCodigo/PtsCercanosDatosBrutos.png")


f = plt.figure()
plt.suptitle("Datos Decodificados")
for i, indx in enumerate(indicesVecinos):

    ax = plt.subplot(3, 4, i+1)

    if i==0:
        ax.title.set_text("Muestra centrada")
    else:
        titulo = f"Indice {indx}"
        ax.title.set_text(titulo)

    generar_plots(datosDecodificados, indx, 0, 1)

f.savefig("/home/jorgemel/capturasCodigo/PtsCercanosDatosDecodificados.png")

# vmin = int(datos[indicesVecinos].min()) - 1
# vmax = int(datos[indicesVecinos].max()) + 1

# f = plt.figure(figsize=(6,6))
# plt.suptitle("Datos vs Reconstrucción")
# for i, indx in enumerate(indicesVecinos):

#     ax = plt.subplot(3, 4, i+1)
#     if i==0:
#         ax.title.set_text("Muestra centrada")
#     else:
#         titulo = f"Indice {indx}"
#         ax.title.set_text(titulo)

#     generar_plots(datos, indx, vmin, vmax)
#     generar_plots(datosDecodificados, indx, vmin, vmax)

# f.savefig("/home/jorgemel/capturasCodigo/PtsCercanosDatosBrutos.png")
    

if len(datosEspacioLatente.shape) > 2:
    vmin = datosEspacioLatente[indicesVecinos].min()-1
    vmax = datosEspacioLatente[indicesVecinos].max()+1

    f = plt.figure()
    plt.suptitle("Espacio Latente")
    for i, indx in enumerate(indicesVecinos):

        ax = plt.subplot(3, 4, i+1)
        if i==0:
            ax.title.set_text(f"Indice {indx}")
        else:
            titulo = f"Indice {indx}"
            ax.title.set_text(titulo)

        generar_plots(datosEspacioLatente, indx, datosEspacioLatente.min(), datosEspacioLatente.max())
    f.savefig("/home/jorgemel/capturasCodigo/PtsCercanosLS.png")


f, axes = plt.subplots()
nombresMuestras = ["muestra" + str(i) for i in range(len(indicesVecinos)) ]
nombresMuestras[0] = "Muestra centrada"
axes.set_yticks(np.arange(1, len(nombresMuestras)+1, 2))# , labels=nombresMuestras)
axes.set_xticks(np.arange(0, latentDims))
axes.set_ylabel("Muestras")
axes.set_xlabel("Dimensiones latentes")
axes.set_title("Espacio Latente (Flatten)")
vecinos = []
for i, indx in enumerate(indicesVecinos):
    vecinos.append(datosEspacioLatente2[indx])

plt.imshow(vecinos, cmap="seismic", vmin=0, vmax=1)
plt.colorbar()

f.savefig("/home/jorgemel/capturasCodigo/PtsCercanosLSFlatten.png")

f = plt.figure()
generar_grafica_ls(datosEspacioLatente2, indicesVecinos)
f.savefig("/home/jorgemel/capturasCodigo/PtsCercanosLSGRF.png")

# Figura de scatter
f = plt.figure()
plt.scatter(proyeccionPlana[:,0], proyeccionPlana[:,1], alpha=0.3)
plt.scatter(proyeccionPlana[indicesVecinos,0], proyeccionPlana[indicesVecinos,1])
f.savefig("/home/jorgemel/capturasCodigo/PtsCercanosProyeccion.png")


# # Figura de quiver
# f = plt.figure()
# plt.tight_layout()
# plt.suptitle("Celulas + quiver")
# for i, indx in enumerate(indicesVecinos):

#     ax = plt.subplot(3, 4, i+1)
#     if i==0:
#         ax.title.set_text("Muestra marcada")
#     else:
#         titulo = f"Indice {indx}"
#         ax.title.set_text(titulo)

#     generar_quivers(imagenes, flechas, indx)

# f.savefig("./capturasCodigo/PtsCercanosQuiver.png")


plt.show()
