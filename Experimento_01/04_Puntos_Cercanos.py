"""
+-------------------------------------------------------------------------------------------+
| Visualización no interactiva que genera el scatter plot en 2 dimensiones de los datos y   |
| las muestras que dieron lugar a los 11 puntos más cercanos al punto seleccionado en la    |
| proyección, saleccionado por índice o por coordenadas (baja dimensión).                   |
|                                                                                           |
| Autor: Jorge Menéndez Lagunilla                                                           |
| Fecha: 12/2023                                                                            |
|                                                                                           |
+-------------------------------------------------------------------------------------------+
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

import sys
sys.path.insert(0, "/home/jorgemel/motitlidad2/")
from libs.generales import obtener_dataset


# =========================================================================================== #
# PARÁMETROS

MODELO = "CAE1D_01"

PATH_DATOS = "./proyecciones/"
PATH_FLECHAS = "./data/imagenes_muestras.hdf5"
NOMBRE_ARCHIVO = "CAE1D_01_proyecciones_15_K3.hdf5"

PATH_MODELOS = "./models/"

MODELO_ENCODER = "encoder_CAE1D_01_15_K3.keras"
MODELO_DECODER = "decoder_CAE1D_01_15_K3.keras"


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


def generar_plots(data: np.ndarray, indice:int, vmin:float=None, vmax:float=None):
    """Genera las imágenes para representar los datos del espacio ambiente. Supone
    que los datos están escalados a intervalo [-1,1].

    Args:
        data (np.ndarray): Datos a mostrar.
        indice (int): Índices de los datos a visualizar.
        vmin (float, optional): Valor mínimo para tomar en el escalado del color. Defaults to None.
        vmax (float, optional): Valor máximo para tomar en el escalado del color. Defaults to None.
    """

    norm = Normalize() # Para hacer la colorbar bien

    if (vmin is not None) and (vmax is not None):
        norm = Normalize(vmin=vmin, vmax=vmax)
        # plt.imshow(data[indice], cmap="coolwarm", origin="lower", norm=norm) # vmin=vmin, vmax=vmax
    # else:
        # norm = Normalize(vmin=np.percentile(data, 5), vmax=np.percentile(data, 95))
    plt.imshow(data[indice], cmap="coolwarm", origin="lower", norm=norm)
    
    plt.xticks(())
    plt.colorbar(ScalarMappable(norm=norm, cmap="coolwarm" , ticks=(vmin, (vmin+vmax)/2 , vmax))) 

    # plt.plot(data[indice])
    # plt.xticks(np.arange(data.shape[-1]))
    # plt.yticks(np.arange(vmin, vmax))

    return None


def generar_quivers(img: np.ndarray, quiver: np.ndarray, index:tuple):

    xWin, yWin = np.meshgrid(np.arange(0, img[index].shape[-2], 4), np.arange(0, img[index].shape[-1], 4))
    plt.imshow(img[index], cmap="Greys", vmin=0, vmax=1)
    plt.quiver(xWin, yWin, quiver[index, :, :, 2], -quiver[index, :, :, 3], color="yellow")
    plt.xticks([])
    plt.yticks([])

    return None


def generar_plots_ls(data: np.ndarray, indice:int, vmin:float=None, vmax:float=None):
    
    
    norm = Normalize() # Para hacer la colorbar bien

    if (vmin is not None) and (vmax is not None):
        norm = Normalize(vmin=vmin, vmax=vmax)

    plt.imshow(data[indice].reshape(1,data.shape[-1]), cmap="coolwarm", norm=norm)
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
    plt.xticks(np.arange(0,data.shape[-1]))

    return None


# =========================================================================================== #
# MAIN

# Cargamos la proyeccion y sus datos
proyeccion = obtener_dataset(PATH_DATOS + NOMBRE_ARCHIVO, "proyeccion")
print("Shape proyeccion: ", proyeccion.shape)

datos = obtener_dataset(PATH_DATOS + NOMBRE_ARCHIVO, "muestras")
# datos = datos.squeeze()
print("Shape muestras: ", datos.shape)

# Cargamos las imágenes y las flechas
imagenes = obtener_dataset(PATH_FLECHAS, "imagenes")    
imagenes = imagenes / 255  # Normalizamos
imagenes = imagenes.reshape((-1,imagenes.shape[-2],imagenes.shape[-1]))
# imagenes = imagenes.squeeze() 
print("Shape imagenes: ", imagenes.shape)

flechas =  obtener_dataset(PATH_FLECHAS, "velocidad")
flechas = flechas.reshape((-1,flechas.shape[-3],flechas.shape[-2],flechas.shape[-1]))
# flechas = flechas.squeeze()
print("Shape flechas: ", flechas.shape)

# Generamos un índice aleatorios para plotear
# N_VIDEOS = proyeccion.shape[0]
# N_FRAMES = proyeccion.shape[1]
# N_WIN = proyeccion.shape[2]
# N_MUESTRAS_VIDEO = N_FRAMES * N_WIN
N_MUESTRAS = proyeccion.shape[0]
N_CANALES = datos.shape[-1]

encoder = load_model(PATH_MODELOS + MODELO_ENCODER)
decoder = load_model(PATH_MODELOS + MODELO_DECODER)

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

# ------------------------------------------------------------------------------------------- #
# Escalamos el espacio latente y los datos del espacio ambiente, cada uno por su lado,
# para facilitar las visualizaciones. Pasamos a un intervalo [-1,1] porque los datos oscilan
# entorno a 0.

datosEspacioLatente2 = MinMaxScaler((-1,1)).fit_transform(datosEspacioLatente2)

scalerDatos = MinMaxScaler((-1,1))
scalerDatos.fit(datos.reshape((-1, 7)))
datos = scalerDatos.transform(datos.reshape((-1,7))).reshape((-1, 16, 7))
datosDecodificados = scalerDatos.transform(datosDecodificados.reshape((-1,7))).reshape((-1, 16, 7))

# ------------------------------------------------------------------------------------------- #

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
# f.savefig(f"/home/jorgemel/motilidad2/temporal/1Drotdiv/figures/PtsCercanosDatosBrutos_{x}_{y}.png")


f = plt.figure(figsize=(8,8))
plt.suptitle("Datos Decodificados")
for i, indx in enumerate(indicesVecinos):

    ax = plt.subplot(3, 4, i+1)

    titulo = f"Muestra {indx}"
    ax.title.set_text(titulo)

    generar_plots(datosDecodificados, indx, -1, 1)

f.tight_layout()
# f.savefig(f"/home/jorgemel/motilidad2/temporal/1Drotdiv/figures/PtsCercanosDatosDecodificados_{x}_{y}.png")

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
# f.savefig(f"/home/jorgemel/motilidad2/temporal/1Drotdiv/figures/PtsCercanosLSFlatten_{x}_{y}.png")


# Gráfico de espacio latente
f = plt.figure(figsize=(8,8))
generar_grafica_ls(datosEspacioLatente2, indicesVecinos, -1, 1)
f.tight_layout()
# f.savefig(f"/home/jorgemel/motilidad2/temporal/1Drotdiv/figures/PtsCercanosLSGRF_{x}_{y}.png")

# Figura de scatter
f = plt.figure()
plt.scatter(proyeccionPlana[:,0], proyeccionPlana[:,1], alpha=0.3)
plt.scatter(proyeccionPlana[indicesVecinos,0], proyeccionPlana[indicesVecinos,1])
f.tight_layout()
# f.savefig(f"/home/jorgemel/motilidad2/temporal/1Drotdiv/figures/PtsCercanosProyeccion_{x}_{y}.png")


# Figura de quiver
f = plt.figure()
plt.suptitle("Celulas + quiver")
for i, indx in enumerate(indicesVecinos):

    ax = plt.subplot(3, 4, i+1)
    titulo = f"Indice {indx}"
    ax.title.set_text(titulo)

    generar_quivers(imagenes, flechas, indx)

f.tight_layout()
# f.savefig(f"/home/jorgemel/motilidad2/temporal/1Drotdiv/figures/PtsCercanosQuiver_{x}_{y}.png")


plt.show()
