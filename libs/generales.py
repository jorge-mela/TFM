'''
Funciones de ámbito general que pueden tener cabida en la mayoría de scripts

Autor: Jorge Menéndez Lagunilla

'''

# =========================================================================================== #
# LIBRERIAS

import os
import h5py 
import numpy as np
import cv2 as cv
import random as python_random
import tensorflow as tf
from sklearn.model_selection import train_test_split as __tts


# =========================================================================================== #
# FUNCIONES

def cargar_video(
    path: str, nframes: int = None
):  # --> tuple (np.ndarray, tuple(int, int, int))
    """
    Retorna una tupla con el vídeo en formato ndarray y una tupla de enteros
    indicando el tamaño y el número de fotogramas del vídeo.

    ## Parámetros
    - path: String con el path hasta el archivo .hd5f
    - nframes: (opcional) entero con el número de frames a coger del vídeo
    """

    if not os.path.isfile(path):
        exStr = f"No such file or directory {path}"
        raise ValueError(exStr)

    # Decimos qué vídeo estamos cargando
    print("Cargando video...", path.split(sep="/")[-1])

    cap = cv.VideoCapture(path)
    # Extraemos la información del vídeo para reservar la memoria
    n_ancho = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))  # 2000
    n_alto = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))  # 2000
    n_fotogramas = int(cap.get(cv.CAP_PROP_FRAME_COUNT))  # 539

    if nframes != None:
        if nframes <= n_fotogramas:
            n_fotogramas = nframes
        else:
            ex_str = f"Expected type int with value less than or equal to n_frames ({n_fotogramas})"
            raise TypeError(ex_str)

    video = np.empty((n_fotogramas, n_alto, n_ancho), np.dtype("uint8"))
    # Se pasa a escala de grises por la función a utilizar para obtener el
    # campo de velocidad.

    # Apuntamos al primer fotograma y nos quedamos con el flag para ver si se hizo correctamente
    ret = cap.grab()

    # Inicializamos el contador para recorrer los fotogramas del video
    i = 0
    while ret and i < n_fotogramas:
        # Extraemos el fotograma (el primer argumento es un booleano)
        frame = cap.retrieve()[1]
        # Lo pasamos a escala de grises y lo almacenamos
        video[i] = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Apuntamos al siguiente frame
        ret = cap.grab()
        i += 1

    # Liberamos la memoria tras haber terminado de recorrer el vídeo
    cap.release()

    info_video = (
        f"\nNumero de fotogramas: {n_fotogramas}\nAncho: {n_ancho}\nAlto: {n_alto}\n"
    )
    print(info_video)

    info = (n_fotogramas, n_alto, n_ancho)

    return video, info


def obtener_dataset(path: str, nombreData: str):  # --> data: np.ndarray
    """
    Retorna un ndarray con los valores del dataset seleccionado.

    ## Parámetros
    - path: String con el path hasta el archivo .hd5f
    - nombreData: String con el nombre del dataset dentro del archivo
    """

    if not os.path.isfile(path):
        exStr = f"No such file or directory {path}"
        raise ValueError(exStr)

    f = h5py.File(path, "r")

    if nombreData not in f.keys():
        keys = f.keys()
        exStr = f"No such key ({nombreData}) in {path} dataset. Available keys: {keys}"
        f.close()
        raise ValueError(exStr)

    data = np.asarray(f[nombreData]).copy()
    f.close()

    return data


def calcula_error(original:np.ndarray, estimada:np.ndarray, metrica:str): # Funciona para datos unidimensionales

        if metrica == "MSE":
            error = np.sum((estimada - original)**2)/len(estimada)

        if metrica == "RMSE":
            error = np.sqrt(np.sum((estimada - original)**2)/len(estimada))

        return error


def obtener_vector_unitario(a: np.ndarray):

    # Idealmente es un vector fila, pero por si acaso aplanamos
    n = a.flatten()

    norma = np.sqrt(np.sum(n**2))
    n = n / norma

    return n


def reset_seeds(seed_value=39):
    # ref: https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    # necessary for starting Numpy generated random numbers in a well-defined initial state.
    np.random.seed(seed_value)
    # necessary for starting core Python generated random numbers in a well-defined state.
    python_random.seed(seed_value)
    # set_seed() will make random number generation
    tf.random.set_seed(seed_value)


def train_val_test_split(data, train_size:float=0.7, val_size:float=0.2, test_size:float=0.1, semilla:int=42):
    """
    Genera las particiones de entrenamiento, validación y test para aplicarlos a un modelo de Machine Learning.
 
    ## Parámetros
    - data: Muestras a las que hacer la partición, con el número de muestras en el primer eje
    - train_size: Flotante entre 0 y 1 que indica el porcentaje de muestras dedicadas a entrenamiento
    - val_size: Flotante entre 0 y 1 que indica el porcentaje de muestras dedicadas a validacion
    - test_size: Flotante entre 0 y 1 que indica el porcentaje de muestras dedicadas a test
    """

    if not np.isclose((train_size+val_size+test_size), 1, atol=1e-9):
        suma = train_size+val_size+test_size
        exStr = f'Los porcentajes de las particiones han de sumar 1 ({train_size}+{val_size}+{test_size}={suma})'
        raise ValueError(exStr)

    trainSamples, valSamples = __tts(data, test_size=val_size, random_state=semilla)
    result = [trainSamples, valSamples]

    if test_size != 0:
        # Para obtener una partición de validación
        long_test = int(test_size/train_size * trainSamples.shape[0])
        indice_test= trainSamples.shape[0] - long_test

        testSamples = trainSamples[indice_test:] # La parte del final del training para test
        trainSamples= trainSamples[0:indice_test] # Actualizamos al verdadero training

        result = [trainSamples, valSamples, testSamples]

    return result


# =========================================================================================== #

# _l = []
# copy_dict = dict(locals())
# for key, value in copy_dict.items():
#     if "function" in str(value):
#         _l.append(key)
# print(_l)

# class __Main():
#     def _main():
#         texto = (
#             '''
#     En este fichero se definen funciones para la visualizacion de los datos.
#             '''
#         )
#         print(texto)
#         print("Funciones:\n")
#         for func in _l:
#             print("- " + func)

#         print("\n")

#         return None


# if "__main__" == __name__:

#     __Main()._main()