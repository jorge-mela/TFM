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

sys.path.insert(0, '/home/jorgemel/motilidad2/')  
from libs.generales import cargar_video, obtener_dataset


# =========================================================================================== #
# PARÁMETROS

LIMITE_INFERIOR = 428 * 4
LIMITE_SUPERIOR = 73 * 4

allVideos = ["vid_08_SDHB", "vid_09_SDHB"]
PATH_RESULT = "./Experimento_04/data/"

D = 75  # Tamaño usado para cada lado de la ventana
# 75 parece ir bastante bien. Al final nos quedarán ventanas de 150x150 en el vídeo.

RESOLUCION = 128

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
    muestraFFT_ = []

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
                coordsWindows[j, 0] // 4 : coordsWindows[j, 0] // 4 + (D // 2) + 1,
                coordsWindows[j, 2] // 4 : coordsWindows[j, 2] // 4 + (D // 2) + 1,
                2,
            ]  # SALE 38x38
            campoV = flechas[
                i,
                coordsWindows[j, 0] // 4 : coordsWindows[j, 0] // 4 + (D // 2) + 1,
                coordsWindows[j, 2] // 4 : coordsWindows[j, 2] // 4 + (D // 2) + 1,
                3,
            ]  # SALE 38x38

            celulas_.append([campoU, campoV])

        celulas = np.asarray(celulas_)

        # ------------------------------------------------------------------------------------ #

        # CARACTERÍSTICAS
        # Calculamos la característica
        for j in range(len(celulas)):
            x = np.empty(celulas.shape[2:], dtype=np.complex128)
            x.real = celulas[j,0]
            x.imag = celulas[j,1]

            # Generamos una vetana de hanning 2D que aplicamos a las flechas
            hann2D = np.asarray([np.hanning(x.shape[0])] * x.shape[1])
            hann2D = hann2D.T * np.hanning(x.shape[1])

            # Reservamos memoria para guardar el resultado en el formato que queremos
            celula_envent = np.empty(x.shape, dtype=np.complex128)
            celula_envent = hann2D * x

            #*****************************************************************************#
            # Hacemos un padding para aumentar la resolución de la fft
            originalSize = celula_envent.shape[0] # Las ventanas son cuadradas
            celula_envent = np.pad(celula_envent,((RESOLUCION-celula_envent.shape[0])//2))

            #*****************************************************************************#

            Z = np.fft.fft2(celula_envent)
            Zshift = np.fft.fftshift(Z)

            #************************************************************#
            # Recortamos la imagen para quedarnos con el tamaño inicial
            # nuevaSize = Zshift.shape[0]
            # Zshift = Zshift[nuevaSize//2 - originalSize//2 : nuevaSize//2 + originalSize//2,
            #                 nuevaSize//2 - originalSize//2 : nuevaSize//2 + originalSize//2]
            #
            #************************************************************#

            Zmag = abs(Zshift)

            # Almacenamos la información importante
            # camposFlechas_.append(celula_envent)
            imagenCelula_.append(windows[j])
            coordsCelula_.append(coordsWindows[j])
            muestraFFT_.append(Zmag) # Solo cogemos la magnitud

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
    muestraFFT = np.asarray(muestraFFT_)
    print("muestraA.shape: ", muestraFFT.shape)

    print("\n\n\nComenzamos la generación del dataset...", NOMBRE_VIDEO)

    # Una vez procesado todo el vídeo, generamos el dataset con las muestras
    f = h5py.File(PATH_RESULT + NOMBRE_VIDEO + ".hdf5", "w")
    dset1 = f.create_dataset("flechas", data=camposFlechas, compression="gzip")
    dset2 = f.create_dataset("imagenes", data=imagenCelula, compression="gzip")
    dset3 = f.create_dataset("coordenadas", data=coordsCelula, compression="gzip")
    dset4 = f.create_dataset("muestras", data=muestraFFT, compression="gzip")
    f.close()

    print("Terminamos la generación del dataset...", NOMBRE_VIDEO)


# Juntamos los resultados en un único archivo
video = allVideos.pop(0)  # Extraemos uno de los vídeos utilizados
flechas = obtener_dataset(PATH_RESULT + video + ".hdf5", "flechas")
imagenes = obtener_dataset(PATH_RESULT + video + ".hdf5", "imagenes")
coordenadas = obtener_dataset(PATH_RESULT + video + ".hdf5", "coordenadas")
fft = obtener_dataset(PATH_RESULT + video + ".hdf5", "muestras")

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
    fft = np.concatenate(
        (fft, obtener_dataset(PATH_RESULT + video + ".hdf5", "muestras")),
        axis=0,
    )

    # Recortamos las muestras para que tengan el tamaño original (PROVISIONAL)
    fft = fft[:,
              RESOLUCION//2 - originalSize//2 : RESOLUCION//2 + originalSize//2,
              RESOLUCION//2 - originalSize//2 : RESOLUCION//2 + originalSize//2]
    
    # Añadimos el axis para el canal a las muestras
    fft = np.expand_dims(fft, -1)

print("\n\n\nComenzamos la generación del dataset...")

# Una vez procesado todo el vídeo, generamos el dataset con las muestras
f = h5py.File(PATH_RESULT + "01_muestras" + ".hdf5", "w")
dset1 = f.create_dataset("flechas", data=flechas, compression="gzip")
dset2 = f.create_dataset("imagenes", data=imagenes, compression="gzip")
dset3 = f.create_dataset("coordenadas", data=coordenadas, compression="gzip")
dset4 = f.create_dataset("muestras", data=fft, compression="gzip")
f.close()

print("Terminamos la generación del dataset...")
