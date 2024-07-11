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

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler


from libs.generales import obtener_dataset


# =========================================================================================== #
# PARÁMETROS

# PATH_IMAGENES = 
# PATH_FLECHAS = 

MODELO = "CAE1D"

PATH_DATOS = "/home/jorgemel/motilidad2/temporal/1Drotdiv/proyecciones/"
PATH_FLECHAS = "/home/jorgemel/motilidad2/temporal/1Drotdiv/data/imagenes_muestras.hdf5"
ARCHIVO = MODELO + "_proyecciones_TEMPORAL.hdf5"

PATH_MODELOS = "/home/jorgemel/motilidad2/temporal/1Drotdiv/models/"

ENCODER = MODELO + "_encoder_TEMPORAL.keras" 
DECODER = MODELO + "_decoder_TEMPORAL.keras"


# m = input("c/i: ")
# if m=="c":
#     modo = "coordenadas"
# elif m=="i":
#     modo = "aleatorio"
modo = "coordenadas"

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
# DATOS

# Cargamos la proyeccion y sus datos
proyeccion = obtener_dataset(PATH_DATOS + ARCHIVO, "proyeccion")
print("Shape proyeccion: ", proyeccion.shape)

datos = obtener_dataset(PATH_DATOS + ARCHIVO, "muestras")
datos = datos.squeeze()
print("Shape muestras: ", datos.shape)

# Cargamos las imágenes y las flechas
imagenes = obtener_dataset(PATH_FLECHAS, "imagenes")
# imagenes = imagenes / 255  # Normalizamos
# imagenes = imagenes.squeeze() 
print("Shape imagenes: ", imagenes.shape)

flechas =  obtener_dataset(PATH_FLECHAS, "velocidad")
# flechas = flechas.squeeze()
print("Shape flechas: ", flechas.shape)


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

datosEspacioLatente2 = MinMaxScaler((-1,1)).fit_transform(datosEspacioLatente2)

scalerDatos = MinMaxScaler((-1,1))
scalerDatos.fit(datos.reshape((-1, 7)))
datos = scalerDatos.transform(datos.reshape((-1,7))).reshape((-1, 16, 7))
datosDecodificados = scalerDatos.transform(datosDecodificados.reshape((-1,7))).reshape((-1, 16, 7))

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
    indicesVecinos = np.argsort(distancias)[:12] # Cogemos los 12 puntos más cercanos
    
# Figuras de los datos
f = plt.figure(figsize=(8,8))
plt.suptitle("Datos Brutos")
for i, indx in enumerate(indicesVecinos):

    ax = plt.subplot(3, 4, i+1)
    # if i==0:
    #     ax.title.set_text("Muestra centrada")
    # else:
    titulo = f"Muestra {indx}"
    ax.title.set_text(titulo)
    ax.set_yticks((0,5,10,15))
    ax.set_xticks((0,2,4,6))

    generar_plots(datos, indx, -1, 1)

f.tight_layout()
f.savefig("/home/jorgemel/capturasCodigo/PtsCercanosDatosBrutos.png")


# f = plt.figure(figsize=(8,8))
# plt.suptitle("Datos Decodificados")
# for i, indx in enumerate(indicesVecinos):

#     ax = plt.subplot(3, 4, i+1)

#     # if i==0:
#     #     ax.title.set_text("Muestra centrada")
#     # else:
#     titulo = f"Muestra {indx}"
#     ax.title.set_text(titulo)

#     generar_plots(datosDecodificados, indx, -1, 1)

# f.tight_layout()
# f.savefig("/home/jorgemel/capturasCodigo/PtsCercanosDatosDecodificados.png")

# Cuando tenemos el espacio latente de dos dimensiones mostramos las imágenes
# de los tensores de cada muestra. 
# if len(datosEspacioLatente.shape) > 2:
#     vmin = datosEspacioLatente[indicesVecinos].min()-1
#     vmax = datosEspacioLatente[indicesVecinos].max()+1

#     f = plt.figure()
#     plt.suptitle("Espacio Latente")
#     for i, indx in enumerate(indicesVecinos):

#         ax = plt.subplot(3, 4, i+1)
#         if i==0:
#             ax.title.set_text(f"Indice {indx}")
#         else:
#             titulo = f"Indice {indx}"
#             ax.title.set_text(titulo)

#         generar_plots(datosEspacioLatente, indx)
#     f.savefig("/home/jorgemel/capturasCodigo/PtsCercanosLS.png")


f, axes = plt.subplots(figsize=(8,8))
nombresMuestras = ["muestra" + str(i) for i in range(len(indicesVecinos)) ]
# nombresMuestras[0] = "Muestra centrada"
axes.set_yticks(np.arange(1, len(nombresMuestras)+1, 2))# , labels=nombresMuestras)
axes.set_xticks(np.arange(0, latentDims))
axes.set_ylabel("Muestras")
axes.set_xlabel("Dimensiones latentes")
axes.set_title("Espacio Latente (Flatten)")
vecinos = []
for i, indx in enumerate(indicesVecinos):
    vecinos.append(datosEspacioLatente2[indx])

plt.imshow(vecinos, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()

f.tight_layout()
f.savefig("/home/jorgemel/capturasCodigo/PtsCercanosLSFlatten.png")


# Gráfico de espacio latente
f = plt.figure(figsize=(8,8))
generar_grafica_ls(datosEspacioLatente2, indicesVecinos, -1, 1)
f.tight_layout()
f.savefig("/home/jorgemel/capturasCodigo/PtsCercanosLSGRF.png")

# Figura de scatter
f = plt.figure()
plt.scatter(proyeccionPlana[:,0], proyeccionPlana[:,1], alpha=0.3)
plt.scatter(proyeccionPlana[indicesVecinos,0], proyeccionPlana[indicesVecinos,1])
f.tight_layout()
f.savefig("/home/jorgemel/capturasCodigo/PtsCercanosProyeccion.png")


# # Figura de quiver
# f = plt.figure()
# plt.suptitle("Celulas + quiver")
# for i, indx in enumerate(indicesVecinos):

#     ax = plt.subplot(3, 4, i+1)
#     titulo = f"Indice {indx}"
#     ax.title.set_text(titulo)

#     generar_quivers(imagenes, flechas, indx)

# f.tight_layout()
# f.savefig("./capturasCodigo/PtsCercanosQuiver.png")


plt.show()
