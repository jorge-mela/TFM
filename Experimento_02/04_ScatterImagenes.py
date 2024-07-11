"""
+-------------------------------------------------------------------------------------------+ 
| Visualización de las sparklines dentro de la proyección 2D para observar la ordenación    |
| general conseguida por el embedding.                                                      |
|                                                                                           |
| Autor: Jorge Menéndez Lagunilla                                                           |
| Fecha: 02/2024                                                                            |
|                                                                                           |
+-------------------------------------------------------------------------------------------+
"""


# =========================================================================================== #
# LIBRERIAS

import matplotlib.pyplot as plt
import matplotlib.colors as cm
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.preprocessing import MinMaxScaler

from scipy.spatial.distance import cdist

import sys
sys.path.insert(0, "/home/jorgemel/motilidad2/")
from libs.generales import obtener_dataset


# =========================================================================================== #
# PARÁMETROS

NOMBRE_ARCHIVO = "ROT_CAE2D_04_proyecciones_15_K2.hdf5"
PATH_DATOS = "./Experimento_02/proyecciones/"

PATH_IMAGENES = "./.borrame/" 

N_SAMPLES = 200

SIZE_FIG = 8
ZOOM_IMGS = SIZE_FIG / 3.6

# =========================================================================================== #
# FUNCIONES

def getImage(path, zoom=1):
    return OffsetImage(plt.imread(path), zoom=zoom)


def calcula_histogramas_canales(datos:np.ndarray, nbins:int):
    """Devuelve el histograma de la distribución de cada uno de los canales
    de los datos introducidos. Espera que la última dimensión sea el
    número de canales.

    Args:
        datos (np.ndarray): Muestras sobre las que se desea calcular su distribución
        nbins (int): Número de bins que se desea tener en el histograma

    Returns:
        (tuple): Devuelve h como un np.ndarray con los valores de los histogramas
         para cada canal y x con los bins para cada canal
    """

    nCanales = datos.shape[-1]
    h = np.zeros(shape=(nbins, nCanales))
    x = np.zeros(shape=(nbins+1, nCanales))

    for i in range(nCanales):
        h[:,i], x[:,i] = np.histogram(datos[:,i], nbins)


    return (h,x)

# Hay que cambiar el contenido de esta función cada vez que se cambien los datos
def generarFigura(data: np.ndarray, idx:int):
    fig = plt.figure(figsize=(0.2, 0.2), facecolor="#ffffff00")

    # Solo cambiar esta parte
    # ****************************************************************************#

    plt.imshow(data[idx], cmap="coolwarm", vmin=-1, vmax=1)
    # plt.plot(data[idx], color="orange", marker=".")

    # etiquetas = ["rotacional", "divergencia", "Ex", "Ey", "Txy", "bx", "by"]
    # colores = ["tab:red", "tab:blue", "tab:orange", "tab:yellow", "tab:green", "tab:purple", "tab:black"]

    # etiquetas = ["a11", "a12", "a21", "a22", "bx", "by"]
    # colores = ["red", "blue", "orange", "yellow", "green", "purple"]

    # plt.bar(etiquetas, data[idx], color=colores )

    # D = 75; xWin, yWin = np.meshgrid(np.arange(0, D*2, 4), np.arange(0, D*2, 4))
    # plt.quiver(xWin, yWin, data[idx,:,:,0], -data[idx,:,:,1], color="black", scale=50)


    # ****************************************************************************#

    plt.tight_layout()
    plt.axis("off")
    nombreImagen = str(idx) + ".png"
    fig.savefig(PATH_IMAGENES + nombreImagen, bbox_inches="tight")
    plt.close(fig)


def poisson_disc_centres(r:float, coords:np.ndarray):

    available = coords.copy()
    centros_ = []

    # Para controlar que no se quede pinzado
    k = 0

    while available.shape[0] != 0 and k < 30:
        shapeAnterior = available.shape[0]
        pt = np.expand_dims(available[np.random.randint(0, available.shape[0])], 0)
        centros_.append(pt)
        dist = cdist(pt, available)

        # Nos quedamos solo con los elementos que están suficientemente alejados
        elim = dist > (2*r)
        available = available[elim[0]]

        # Si no se reduce la longitud de available, terminamos
        if available.shape[0] == shapeAnterior:
            k = k+1
        else:
            k = 0
            shapeAnterior = available.shape[0]

    centros = np.asarray(centros_)
    centros = np.squeeze(centros)

    return centros


def poisson_disc_indices(r:float, coords:np.ndarray):

    available = coords.copy()
    indx = np.arange(len(available))
    indices_ = []

    # Para controlar que no se quede pinzado
    k = 0

    while available.shape[0] != 0 and k < 30:
        shapeAnterior = available.shape[0]
        i = np.random.randint(0, available.shape[0])
        indices_.append(indx[i])
        pt = np.expand_dims(available[i], 0)
        dist = cdist(pt, available)

        # Nos quedamos solo con los elementos que están suficientemente alejados
        elim = dist > (2*r)
        available = available[elim[0]]
        indx = indx[elim[0]]

        # Si no se reduce la longitud de available, terminamos
        if available.shape[0] == shapeAnterior:
            k = k+1
        else:
            k = 0
            shapeAnterior = available.shape[0]


    return indices_


# =========================================================================================== #
# MAIN

# Cargamos la proyeccion
proyeccion = obtener_dataset(PATH_DATOS + NOMBRE_ARCHIVO, "proyeccion")
print(proyeccion.shape)
proyeccion = proyeccion.reshape(-1,2)

MAX_INDEX = proyeccion.shape[0]

datos = obtener_dataset(PATH_DATOS + NOMBRE_ARCHIVO, "muestras")

datos_s = MinMaxScaler((-1, 1)).fit_transform(datos.reshape((-1,7)))
datos_s = datos_s.reshape(datos.shape)

f, ax = plt.subplots(1, 1, figsize=(SIZE_FIG, SIZE_FIG))
f.tight_layout()
ax.scatter(proyeccion[:,0], proyeccion[:,1], alpha=0.3)
ax.set_title("Espacio ambiente")


# Generamos unos indices para plotear
# indices = np.random.randint(low=0, high=(MAX_INDEX), size=N_SAMPLES)
indices = poisson_disc_indices(0.4, proyeccion)


colorMap = cm.Colormap("coolwarm", 256)

for i in indices:
    im = OffsetImage(datos_s[i], zoom=ZOOM_IMGS, cmap=colorMap)
    img = im.get_children()[-1]
    img.set_cmap("coolwarm")
    img.set_clim(vmin=-1, vmax=1)

    artist = []
    ab = AnnotationBbox(im, (proyeccion[i,0], proyeccion[i,1]), xycoords='data', frameon=False)
    artist.append(ax.add_artist(ab))



f.savefig("Experimento_02/figures/scatterFlechas.png")

plt.show()
