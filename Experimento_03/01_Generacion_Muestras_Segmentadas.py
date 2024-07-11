"""
+-------------------------------------------------------------------------------------------+
| Segmentación de los vídeos para la extracción de ventanas centradas en las células, para  |
| aplicar la extracción de características fft2.                                            |
|                                                                                           |
| Autor: Jorge Menéndez Lagunilla                                                           |
| Fecha: 12/2023                                                                            |
|                                                                                           |
+-------------------------------------------------------------------------------------------+
"""

# =========================================================================================== #
# LIBRERIAS
import sys 

import cv2 as cv
import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge

from skimage.util.shape import view_as_windows

sys.path.insert(0, '/home/jorgemel/motilidad2/')
from libs.generales import cargar_video, obtener_dataset


# =========================================================================================== #
# PARÁMETROS

LIMITE_INFERIOR = 428 * 4
LIMITE_SUPERIOR = 73 * 4

allVideos = ["vid_08_SDHB", "vid_09_SDHB"]
PATH_RESULT = "./Experimento_03/data/"

D = 80  # Doble del tamaño de la ventana

REGWIN_SIZE = 10
N_WIN_V =  (D//2) // REGWIN_SIZE
N_WIN_H = (D//2) // REGWIN_SIZE

# Coordenadas para las muestras
xWin, yWin = np.meshgrid(np.arange(0, D * 2, 4), np.arange(0, D * 2, 4))


# =========================================================================================== #
# FUNCIONES

def calcula_histogramas_canales(datos: np.ndarray, nbins: int):
    """
    Devuelve el histograma de la distribución de cada uno de los canales
    de los datos introducidos. Espera que la última dimensión sea el
    número de canales.

    ## Parámetros
    - datos: Muestras sobre las que se desea calcular su distribución
    - nbins: Número de bins que se desea tener en el histograma
    """

    nCanales = datos.shape[-1]
    h = np.zeros(shape=(nbins, nCanales))
    x = np.zeros(shape=(nbins + 1, nCanales))

    for i in range(nCanales):
        h[:, i], x[:, i] = np.histogram(datos[:, i], nbins)

    return (h, x)


def plot_histogramas_canales(
    h: np.ndarray, x: np.ndarray, titulos: list = None, log: bool = False
):
    """
    Genera una figura de matplotlib con el histograma para cada uno de
    los canales. Espera que la última dimensión sea el número de canales.

    ## Parámetros
    - h: Altura de cada uno de los bins del histograma, por canales
    - x: Anchura de los bins del histograma, por canales
    - log: Booleano para indicar si se quiere escala logarítmica en el eje y
    """
    ncanales = h.shape[-1]

    if titulos is not None:
        if len(titulos) != ncanales:
            errStr = f"Expected titles length to be the same as number of channels. Got {len(titulos)} for {ncanales} channels."
            raise ValueError(errStr)

    if log:
        for channel in range(ncanales):
            plt.figure(titulos[channel])
            plt.bar(x[:-1, channel], h[:, channel], log=True)
    else:
        for channel in range(ncanales):
            plt.figure(titulos[channel])
            plt.bar(x[:-1, channel], h[:, channel], width=0.01)

    return None


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

    # Partimos los datos sumninistrados en las ventanas de regresión
    uwin = view_as_windows(vx, winsize, winsize)
    vwin = view_as_windows(vy, winsize, winsize)
    # xwin = view_as_windows(xx, winsize, winsize)
    # ywin = view_as_windows(xy, winsize, winsize)

    dxx = xx[0,1] - xx[0,0] # Miramos el desplazamiento de las muestras en X
    dxy = xy[1,0] - xy[0,0] # Miramos el desplazamiento de las muestras en Y

    xwin, ywin = np.meshgrid(np.arange(0, winsize*dxx, dxx), np.arange(0, winsize*dxy, dxy))

    # Reservamos memoria
    rot_matrix = np.zeros((vertsize, horzsize))
    div_matrix = np.zeros((vertsize, horzsize))
    exx_matrix = np.zeros((vertsize, horzsize))
    eyy_matrix = np.zeros((vertsize, horzsize))
    exy_matrix = np.zeros((vertsize, horzsize))

    # Para cada ventana generada, hacemos una regresión
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

    return (rot_matrix, div_matrix, exx_matrix, eyy_matrix, exy_matrix)


# =========================================================================================== #

for NOMBRE_VIDEO in allVideos:
    PATH_VIDEO = (
        "./00_comun/videos/SDHB/" + NOMBRE_VIDEO + ".avi"
    )
    PATH_FLECHAS = (
        "./00_comun/velocidades/" + NOMBRE_VIDEO + ".hdf5"
    )
    # Cargamos los fotogramas
    frames, info = cargar_video(PATH_VIDEO, 500)  # Cargamos 500 frames de un video

    # Cargamos las flechas
    flechas = obtener_dataset(PATH_FLECHAS, "video_flow")
    # Las flechas están guardadas traspuestas, así que las corregimos
    flechas = flechas.transpose([0, 2, 1, 3])

    # Quitamos las zonas blancas
    frames = frames[:, LIMITE_SUPERIOR:LIMITE_INFERIOR, :]
    flechas = flechas[:, LIMITE_SUPERIOR // 4 : LIMITE_INFERIOR // 4, :, :]

    # Para almacenar las células que encontremos
    camposFlechas_ = []
    imagenCelula_ = []
    coordsCelula_ = []
    muestraRot_ = []
    muestraDiv_ = []
    muestraExx_ = []
    muestraEyy_ = []
    muestraExy_ = []

    for i, fr in enumerate(frames):

        # ------------------------------------------------------------------------------------ #
        
        # SEGMENTACIÓN
        print("Procesando frame...", i)
        # Hacemos un filtro de mediana para eliminar el ruido S&P
        medFilt = cv.medianBlur(fr, 11)
        # Hacemos un filtro gaussiano para eliminar el ruido blanco
        medGaussFilt = cv.GaussianBlur(medFilt, (11, 11), 0)

        # Binarizamos
        binaria = cv.bitwise_not(
            cv.adaptiveThreshold(
                medGaussFilt,
                255,
                cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv.THRESH_BINARY,
                11,
                2,
            )
        )

        # Hacemos un cierre 3 veces para eliminianar las regiones más pequeñas
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
        closing = cv.morphologyEx(binaria, cv.MORPH_CLOSE, kernel, iterations=3)

        # Rellenamos el fondo de manera que nos quedamos con los agujeros de dentro de las
        # regiones en negro
        agujeros = closing.copy()
        mask = np.zeros((closing.shape[0] + 2, closing.shape[1] + 2), np.uint8)
        cv.floodFill(agujeros, mask, (0, 0), 255)
        # sobrescribe lo que le metas

        # Invertimos el resultado para quedarnos con los agujeros en blanco
        agujerosInv = cv.bitwise_not(agujeros)

        # Sumamos los agujeros en blanco con la imagen inicial para rellenar las regiones
        rellenada = closing + agujerosInv

        # Etiquetamos las regiones y filtramos para quedarnos con las que tienen más probabilidad
        # de ser células
        _, regiones, stats, centroides = cv.connectedComponentsWithStats(
            rellenada, 8, cv.CV_32S
        )
        areas = stats[:, -1]

        # Nos quedamos solo con los centroides de las regiones cuya área cumpla una condición:
        # la ventana entera que encierra la célula ha de estar completamente dentro del FOV
        # de la cámara
        centroidesValidos = []
        areasValidas = []
        for j in range(len(areas)):
            if areas[j] > 850 and areas[j] < 20000:
                centroidesValidos.append(centroides[j])
                areasValidas.append(areas[j])

        centroidesValidos = np.asarray(centroidesValidos)

        # Generamos las ventanas alrededor de las células que nos interesan
        # Hay que darle la vuelta a las coordenadas de los centroides para usarlos en la imagen
        # (i=y, j=x)

        # ------------------------------------------------------------------------------------ #

        # ENVENTANADO
        windows_ = []
        coordsWindows_ = []
        for centro in centroidesValidos:
            x0 = int(centro[1] - D)
            x1 = int(centro[1] + D)
            y0 = int(centro[0] - D)
            y1 = int(centro[0] + D)

            if x0 > 0 and x1 < regiones.shape[0]:
                if y0 > 0 and y1 < regiones.shape[1]:
                    windows_.append(fr[x0:x1, y0:y1])
                    coordsWindows_.append([x0, x1, y0, y1])

        windows = np.asarray(windows_)
        coordsWindows = np.asarray(coordsWindows_)

        print("Encontradas..", len(windows), "células válidas\n")

        # Extraemos las flechas correspondientes a cada muestra
        celulas_ = []
        for j, w in enumerate(windows):
            campoU = flechas[
                i,
                coordsWindows[j, 0] // 4 : coordsWindows[j, 0] // 4 + (D // 2), # + 1,
                coordsWindows[j, 2] // 4 : coordsWindows[j, 2] // 4 + (D // 2), # + 1,
                2,
            ]  # SALE 40x40
            campoV = flechas[
                i,
                coordsWindows[j, 0] // 4 : coordsWindows[j, 0] // 4 + (D // 2), # + 1,
                coordsWindows[j, 2] // 4 : coordsWindows[j, 2] // 4 + (D // 2), # + 1,
                3,
            ]  # SALE 40x40

            celulas_.append([campoU, campoV])

        celulas = np.asarray(celulas_)

        # ------------------------------------------------------------------------------------ #

        # CARACTERÍSTICAS
        # Calculamos la característica
        for j in range(len(celulas)):
            (    
                rot,
                div,
                exx,
                eyy,
                exy,
            ) = calcula_features(
                celulas[j,0],
                celulas[j,1],
                xWin,  
                yWin,  
                REGWIN_SIZE,
                N_WIN_V,
                N_WIN_H,
            )

            # Almacenamos la información importante
            # camposFlechas_.append(celula_envent)
            imagenCelula_.append(windows[j])
            coordsCelula_.append(coordsWindows[j])
            muestraRot_.append(rot)
            muestraDiv_.append(div)
            muestraExx_.append(exx)
            muestraEyy_.append(eyy)
            muestraExy_.append(exy)
        # ------------------------------------------------------------------------------------ #


        camposFlechas_.append(celulas)

    camposFlechas = np.asarray(camposFlechas_.pop(0))
    for o in camposFlechas_:
        if len(o.shape) != 1:
            camposFlechas = np.concatenate((camposFlechas, o), axis=0)
    camposFlechas = camposFlechas.transpose([0, 2, 3, 1])

    print("CamposFlechas.shape: ", camposFlechas.shape)
    imagenCelula = np.asarray(imagenCelula_)
    print("imagenCelula.shape: ", imagenCelula.shape)
    coordsCelula = np.asarray(coordsCelula_)
    print("coordsCelula.shape: ", coordsCelula.shape)
    muestras = np.stack(
        (muestraRot_, muestraDiv_, muestraExx_, muestraEyy_, muestraExy_),
        axis=-1,
    )
    print("muestras.shape: ", muestras.shape)

    print("\n\n\nComenzamos la generación del dataset...", NOMBRE_VIDEO)

    # Una vez procesado todo el vídeo, generamos el dataset con las muestras
    f = h5py.File(PATH_RESULT + NOMBRE_VIDEO + ".hdf5", "w")
    dset1 = f.create_dataset("flechas", data=camposFlechas, compression="gzip")
    dset2 = f.create_dataset("imagenes", data=imagenCelula, compression="gzip")
    dset3 = f.create_dataset("coordenadas", data=coordsCelula, compression="gzip")
    dset4 = f.create_dataset("muestras", data=muestras, compression="gzip")
    f.close()

    print("Terminamos la generación del dataset...", NOMBRE_VIDEO)


# Juntamos los resultados en un único archivo
video = allVideos.pop(0)  # Extraemos uno de los vídeos utilizados
flechas = obtener_dataset(PATH_RESULT + video + ".hdf5", "flechas")
imagenes = obtener_dataset(PATH_RESULT + video + ".hdf5", "imagenes")
coordenadas = obtener_dataset(PATH_RESULT + video + ".hdf5", "coordenadas")
rotdiv = obtener_dataset(PATH_RESULT + video + ".hdf5", "muestras")

# Iteramos sobre el resto de vídeos para concatenar los resultados
for video in allVideos:
    flechas = np.concatenate(
        (flechas, obtener_dataset(PATH_RESULT + video + ".hdf5", "flechas")),
        axis=0,
    )
    imagenes = np.concatenate(
        (imagenes, obtener_dataset(PATH_RESULT + video + ".hdf5", "imagenes")),
        axis=0,
    )
    coordenadas = np.concatenate(
        (
            coordenadas,
            obtener_dataset(PATH_RESULT + video + ".hdf5", "coordenadas"),
        ),
        axis=0,
    )
    rotdiv = np.concatenate(
        (rotdiv, obtener_dataset(PATH_RESULT + video + ".hdf5", "muestras")),
        axis=0,
    )

print("\n\n\nComenzamos la generación del dataset...")

# Una vez procesado todo el vídeo, generamos el dataset con las muestras
f = h5py.File(PATH_RESULT + "01_muestras" + ".hdf5", "w")
dset1 = f.create_dataset("flechas", data=flechas, compression="gzip")
dset2 = f.create_dataset("imagenes", data=imagenes, compression="gzip")
dset3 = f.create_dataset("coordenadas", data=coordenadas, compression="gzip")
dset4 = f.create_dataset("muestras", data=rotdiv, compression="gzip")
f.close()

print("Terminamos la generación del dataset...")
