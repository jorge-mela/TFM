"""
+-------------------------------------------------------------------------------------------+ 
| Script para el entrenamiento con validación cruzada para el denominado "Experimento 2"    | 
|                                                                                           | 
| Autor: Jorge Menéndez Lagunilla                                                           | 
| Fecha: 11/2023                                                                            | 
|                                                                                           |
+-------------------------------------------------------------------------------------------+
"""


# =========================================================================================== #
# LIBRERIAS

import argparse
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import pickle
import sys
from enum import Enum

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import umap
from keras import backend as K
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import (Add, Concatenate, Conv2D, Conv2DTranspose,
                          Cropping2D, Dense, Flatten, Input, MaxPooling2D,
                          Reshape, UpSampling2D)
from keras.models import Model
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split

sys.path.insert(0, '/home/jorgemel/motilidad2/')    
from libs.generales import calcula_error, obtener_dataset, reset_seeds


# =========================================================================================== #
# PARÁMETROS

parser = argparse.ArgumentParser(
    description="Script para el entrenamiento con 2D rotdiv atemporal"
)

parser.add_argument("--model", required=True, type=str)
# parser.add_argument("--seed", required=True, type=int)
parser.add_argument("--dims", required=True, type=int)

args = parser.parse_args()

# Variables de entorno
PATH_MUESTRAS = "./Experimento_02/data/"
PATH_LOGS = "./Experimento_02/logs/"
PATH_FIGURES = "./Experimento_02/figures/"
PATH_PROYECCIONES = "./Experimento_02/proyecciones/"
PATH_MODELOS = "./Experimento_02/models/"

FILE_MUESTRAS = "02_muestras.hdf5"

MODELO = "".join(args.model)

LS_DIMS = int(args.dims)
N_EPOCH = 300
BATCH_SIZE = 16

UMAP_NEIGHBORS = 150
UMAP_MINDIST = 0.4

TSNE_NITER = 3000
TSNE_PERPLEXITY = 50

SEED = 42 # Fijamos la semilla del random (reproducibilidad)

# En caso de que se quisieran hacer experimentos con otras características
CARACTERISTICA = "ROT" 
class Caracteristicas(Enum):
    ROT = 0
    DIV = 1
    EXX = 2
    EYY = 3
    EXY = 4
    ALL = slice(None)

ES = EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=60,
    verbose=1,
    mode="auto",
    baseline=0.7,
    restore_best_weights=True,
)

ROPL = ReduceLROnPlateau()


# =========================================================================================== #
# FUNCIONES

def entrenamiento_modelo(modelo:Model, input_train:list, output_train:list, input_validation:list, output_validation:list, nEpoch:int = 250, batchSize: int = 64, cbacks: tuple = None):

    modelo.fit(
        x=input_train,
        y=output_train,
        shuffle=True,
        epochs=nEpoch,
        batch_size=batchSize,
        validation_data=(input_validation, output_validation),
        verbose=1,
        callbacks=cbacks,
    )

    lastTrainError = modelo.history.history["loss"][-1]
    lastValError = modelo.history.history["val_loss"][-1]

    info = "Error de train: " + str(lastTrainError) + "\nError de val: " + str(lastValError)

    return info


def plot_loss_history(historico, path:str = None, size:tuple=(6,4)):

    if len(size) > 2:
        size = size[:2]

    fig = plt.figure(figsize=size)

    plt.plot(historico.history["loss"], "g")
    plt.plot(historico.history["val_loss"], "b")
    plt.title("model loss", fontsize=16)
    plt.ylabel("loss", fontsize=14)
    plt.xlabel("epoch", fontsize=14)
    plt.legend(["train loss", "test loss"])

    plt.tight_layout()

    if path is not None:
        fig.savefig(path) # Guardamos la figura del entrenamiento

    return fig


def plot_imagen_datos(
    original: np.ndarray,
    decoded: np.ndarray,
    titulo: str, 
    img1titulo: str,
    img2titulo: str,
    path:str = None,
    **kwargs
):
    
    Xlabel1 = ""
    Xlabel2 = ""
    Ylabel1 = ""
    Ylabel2 = ""

    if "X1" in kwargs.keys():
        Xlabel1 = kwargs["Xlabel1"]
    if "X2" in kwargs.keys():
        Xlabel2 = kwargs["Xlabel2"]
    if "Y1" in kwargs.keys():
        Ylabel1 = kwargs["Ylabel1"]
    if "Y2" in kwargs.keys():
        Ylabel2 = kwargs["Ylabel2"]
    
    i = np.random.randint(0, len(original))

    f = plt.figure(figsize=(4, 4))
    plt.suptitle(titulo)
    plt.subplot(121)
    plt.imshow(original[i,:,:], cmap="viridis", vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])
    plt.title(img1titulo)
    plt.xlabel(Xlabel1)
    plt.ylabel(Xlabel2)

    plt.subplot(122)
    plt.imshow(decoded[i], cmap="viridis", vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])
    plt.title(img2titulo)
    plt.xlabel(Ylabel1)
    plt.ylabel(Ylabel2)
    plt.tight_layout()

    if path is not None:
        plt.savefig(path)

    return f


def plot_grafica_datos(
    original: np.ndarray,
    decoded: np.ndarray,
    titulo: str,
    axis1titulo: str,
    axis2titulo: str,
    path:str = None,
    **kwargs
):

    i = np.random.randint(0, len(original))


    if "canal" in kwargs.keys():
        channel = kwargs["canal"]
        slicing=[slice(None)]*len(original.shape); slicing[0]=i; slicing[-1]=channel
        error = "Error: " + str(calcula_error(original[slicing], decoded[slicing], "RMSE"))

        f = plt.figure(figsize=(6, 4))
        plt.plot(np.arange(original[slicing].size), original[slicing].ravel(), "b")
        plt.plot(np.arange(decoded[slicing].size), decoded[slicing].ravel(), "g")
        plt.title(titulo+error)
        plt.legend([axis1titulo, axis2titulo])

    else:
        error = "Error: " + str(calcula_error(original[i], decoded[i], "RMSE"))
        f = plt.figure(figsize=(6, 4))
        plt.plot(np.arange(original[i].size), original[i].ravel(), "b")
        plt.plot(np.arange(decoded[i].size), decoded[i].ravel(), "g")
        plt.title(titulo+error)
        plt.legend([axis1titulo, axis2titulo])

    if path is not None:
        plt.savefig(path)

    return f


def train_val_test_split(data, train_size:float=0.7, val_size:float=0.2, test_size:float=0.1, semilla:int=0):
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

    trainSamples, valSamples = train_test_split(data, test_size=val_size, random_state=semilla)
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
# MODELOS

def generar_modelo_01(shapeEntrada:tuple):

    ncanales = shapeEntrada[-1]

    #encoder
    entrada = Input(shape=(shapeEntrada), name="Entrada")
    encoder = Conv2D(64, (3, 3), activation="relu", padding="same")(entrada)
    encoder = MaxPooling2D((2, 2), strides=2)(encoder)
    encoder = Conv2D(32, (3, 3), activation="relu", padding="same", name="EntradaFlatten")(encoder)
    encoder = Flatten(name="Aplanar")(encoder)
    encoder = Dense(64, activation="relu")(encoder)
    encoder = Dense(32, activation="relu")(encoder) # sigmoid?
    cuelloBotellaEN = Dense(LS_DIMS, name="En_LatentSpace")(encoder)

    encoder = Model(entrada, cuelloBotellaEN)

    # ------------------------------------------------------------------------------------------- #

    # decoder
    cuelloBotellaDE = Input(shape=(cuelloBotellaEN.shape[1:]), name="De_LatentSpace")
    decoder = Dense(32, activation="relu")(cuelloBotellaDE) # sigmoid?
    decoder = Dense(64, activation="relu")(decoder)
    decoder = Dense(encoder.get_layer(name="Aplanar").output_shape[1], activation="relu", name="De_Dense_Reshape")(decoder)
    decoder = Reshape(encoder.get_layer(name="EntradaFlatten").output_shape[1:], name="De_Reshape")(decoder)
    decoder = Conv2DTranspose(1, (3,3), strides=2, activation="relu", padding="same", name="De_ConvTr_01")(decoder)
    decoder = Conv2DTranspose(64, (3,3), strides=1, activation="relu", padding="same")(decoder)
    salida = Conv2D(ncanales, (1, 1), padding="same", name="Salida")(decoder)
    
    decoder = Model(cuelloBotellaDE, salida)


    autoencoder = Model(entrada, decoder(encoder(entrada)))


    return encoder, decoder, autoencoder


def generar_modelo_02(shapeEntrada:tuple):

    ncanales = shapeEntrada[-1]

    # encoder
    entrada = Input(shape=(shapeEntrada), name="Entrada")
    encoder = Conv2D(64, (3, 3), activation="relu", padding="same")(entrada)
    encoder = MaxPooling2D((2, 2), strides=2, name="En_MP_01")(encoder)
    encoder = Conv2D(32, (3, 3), activation="relu", padding="same")(encoder)
    encoder = MaxPooling2D((2,2), strides=2, name="En_MP_02")(encoder)
    encoder = Conv2D(16, (3, 3), activation="relu", padding="same")(encoder)
    encoder = Conv2D(16, (3, 3), activation="relu", padding="same", name="EntradaFlatten")(encoder)
    encoder = Flatten(name="Aplanar")(encoder)
    encoder = Dense(1024, activation="relu")(encoder)
    encoder = Dense(512, activation="relu")(encoder)
    encoder = Dense(512, activation="relu")(encoder)
    encoder = Dense(256, activation="relu")(encoder)
    encoder = Dense(64, activation="relu")(encoder)
    encoder = Dense(32, activation="relu")(encoder)
    cuelloBotellaEN = Dense(LS_DIMS, name="En_LatentSpace")(encoder)

    encoder = Model(entrada, cuelloBotellaEN)

    # ------------------------------------------------------------------------------------------- #

    # decoder
    cuelloBotellaDE = Input(shape=(cuelloBotellaEN.shape[1:]), name="De_LatentSpace")
    decoder = Dense(32, activation="relu", name="De_Dense_32" )(cuelloBotellaDE)
    decoder = Dense(64, activation="relu", name="De_Dense_64")(decoder)
    decoder = Dense(256, activation="relu", name="De_Dense_256")(decoder)
    decoder = Dense(512, activation="relu", name="De_Dense_512_01")(decoder)
    decoder = Dense(512, activation="relu", name="De_Dense_512_02")(decoder)
    decoder = Dense(1024, activation="relu", name="De_Dense_1024")(decoder)
    decoder = Dense(encoder.get_layer(name="Aplanar").output_shape[1], activation="relu", name="De_Dense_Reshape")(decoder)
    decoder = Reshape(encoder.get_layer(name="EntradaFlatten").output_shape[1:], name="De_Reshape")(decoder)
    decoder = Conv2DTranspose(16, (3,3), strides=1, activation="relu", padding="same")(decoder)
    decoder = Conv2DTranspose(1, (3,3), strides=2, activation="relu", padding="same", name="De_ConvTr_01")(decoder)
    decoder = Conv2DTranspose(32, (3,3), strides=1, activation="relu", padding="same")(decoder)
    decoder = Conv2DTranspose(1, (3,3), strides=2, activation="relu", padding="same", name="De_ConvTr_02")(decoder)
    decoder = Conv2DTranspose(64, (3,3), strides=1, activation="relu",padding="same", name="De_ConvTr3")(decoder)
    salida = Conv2D(ncanales, (1, 1), padding="same", name="Salida")(decoder)

    decoder = Model(cuelloBotellaDE, salida)
    

    autoencoder = Model(entrada, decoder(encoder(entrada)))


    return encoder, decoder, autoencoder


def generar_modelo_03(shapeEntrada:tuple):

    ncanales = shapeEntrada[-1]

    # encoder
    entrada = Input(shape=(shapeEntrada), name="Entrada")
    encoder = Conv2D(64, (3, 3), activation="relu", padding="same")(entrada)
    encoder = MaxPooling2D((2, 2), strides=2, name="En_MP_01")(encoder)
    encoder = Conv2D(32, (3, 3), activation="relu", padding="same")(encoder)
    encoder = MaxPooling2D((2,2), strides=2, name="En_MP_02")(encoder)
    encoder = Conv2D(16, (3, 3), activation="relu", padding="same")(encoder)
    encoder = Conv2D(16, (3, 3), activation="relu", padding="same", name="EntradaFlatten")(encoder)
    encoder = Flatten(name="Aplanar")(encoder)
    encoder = Dense(1024, activation="relu")(encoder)
    encoder = Dense(256, activation="relu")(encoder)
    encoder = Dense(32, activation="relu")(encoder)
    cuelloBotellaEN = Dense(LS_DIMS, name="En_LatentSpace")(encoder)

    encoder = Model(entrada, cuelloBotellaEN)

    # ------------------------------------------------------------------------------------------- #

    # decoder
    cuelloBotellaDE = Input(shape=(cuelloBotellaEN.shape[1:]), name="De_LatentSpace")
    decoder = Dense(32, activation="relu", name="De_Dense_32" )(cuelloBotellaDE)
    decoder = Dense(256, activation="relu", name="De_Dense_256")(decoder)
    decoder = Dense(1024, activation="relu", name="De_Dense_1024")(decoder)
    decoder = Dense(encoder.get_layer(name="Aplanar").output_shape[1], activation="relu", name="De_Dense_Reshape")(decoder)
    decoder = Reshape(encoder.get_layer(name="EntradaFlatten").output_shape[1:], name="De_Reshape")(decoder)
    decoder = Conv2DTranspose(16, (3,3), strides=1, activation="relu", padding="same")(decoder)
    decoder = Conv2DTranspose(1, (3,3), strides=2, activation="relu", padding="same", name="De_ConvTr_01")(decoder)
    decoder = Conv2DTranspose(32, (3,3), strides=1, activation="relu", padding="same")(decoder)
    decoder = Conv2DTranspose(1, (3,3), strides=2, activation="relu", padding="same", name="De_ConvTr_02")(decoder)
    decoder = Conv2DTranspose(64, (3,3), strides=1, activation="relu",padding="same", name="De_ConvTr3")(decoder)
    salida = Conv2D(ncanales, (1, 1), padding="same", name="Salida")(decoder)

    decoder = Model(cuelloBotellaDE, salida)
    

    autoencoder = Model(entrada, decoder(encoder(entrada)))


    return encoder, decoder, autoencoder


def generar_modelo_04(shapeEntrada:tuple):

    ncanales = shapeEntrada[-1]

    # encoder
    entrada = Input(shape=(shapeEntrada), name="Entrada")
    encoder = Conv2D(64, (3, 3), strides=1, activation="relu", padding="same")(entrada)
    encoder = Conv2D(64, (3, 3), strides=1, activation="relu", padding="same")(encoder)
    encoder = MaxPooling2D((2, 2), strides=2, name="En_MP_01")(encoder)
    encoder = Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(encoder)
    encoder = Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(encoder)
    encoder = MaxPooling2D((2,2), strides=2, name="En_MP_02")(encoder)
    encoder = Conv2D(20, (3, 3), strides=1, activation="relu", padding="same")(encoder)
    encoder = Conv2D(20, (3, 3), strides=2, activation="relu", padding="same", name="EntradaFlatten")(encoder)
    encoder = Flatten(name="Aplanar")(encoder)
    cuelloBotellaEN = Dense(LS_DIMS, name="En_LatentSpace")(encoder)

    encoder = Model(entrada, cuelloBotellaEN)

    # ------------------------------------------------------------------------------------------- #

    # decoder
    cuelloBotellaDE = Input(shape=(cuelloBotellaEN.shape[1:]), name="De_LatentSpace")
    decoder = Dense(encoder.get_layer(name="Aplanar").output_shape[1], activation="relu", name="De_Dense_Reshape")(cuelloBotellaDE)
    decoder = Reshape(encoder.get_layer(name="EntradaFlatten").output_shape[1:], name="De_Reshape")(decoder)
    decoder = Conv2DTranspose(20, (3,3), strides=2, activation="relu", padding="same")(decoder)
    decoder = Conv2DTranspose(32, (3,3), strides=2, activation="relu", padding="same")(decoder)
    decoder = Conv2DTranspose(32, (3,3), strides=1, activation="relu", padding="same")(decoder)
    decoder = Conv2DTranspose(64, (3,3), strides=2, activation="relu", padding="same")(decoder)
    decoder = Conv2DTranspose(64, (3,3), strides=1, activation="relu", padding="same")(decoder)
    salida = Conv2D(ncanales, (1, 1), padding="same", name="Salida")(decoder)

    decoder = Model(cuelloBotellaDE, salida)
    

    autoencoder = Model(entrada, decoder(encoder(entrada)))


    return encoder, decoder, autoencoder


# def generar_modelo_05(shapeEntrada:tuple): # RECONSTRUYE BIEN

    entrada = Input(shape=(shapeEntrada), name="Entrada")
    encoder = Conv2D(64, (3, 3), strides=1, activation="relu", padding="same")(entrada)
    encoder = MaxPooling2D((2, 2), strides=2, name="En_MP_01")(encoder)
    skipConnection01_e = encoder
    encoder = Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(encoder)
    encoder = MaxPooling2D((2,2), strides=2, name="En_MP_02")(encoder)
    skipConnection02_e = encoder
    encoder = Conv2D(16, (3, 3), activation="sigmoid", padding="same", name="EntradaFlatten")(encoder)
    encoder = Flatten(name="Aplanar")(encoder)
    cuelloBotellaEN = Dense(20, name="En_LatentSpace")(encoder)

    encoder = Model(entrada, cuelloBotellaEN)
    encoder_u = Model(entrada,[cuelloBotellaEN, skipConnection01_e, skipConnection02_e])

    # ------------------------------------------------------------------------------------------- #

    # decoder
    cuelloBotellaDE = Input(shape=(cuelloBotellaEN.shape[1:]), name="De_LatentSpace")
    skipConnection01_d = Input(shape=(skipConnection01_e.shape[1:]), name="De_sk01")
    skipConnection02_d = Input(shape=(skipConnection02_e.shape[1:]), name="De_sk02")
    decoder = Dense(encoder.get_layer(name="Aplanar").output_shape[1], activation="sigmoid", name="De_Dense_Reshape")(cuelloBotellaDE)
    decoder = Reshape(encoder.get_layer(name="EntradaFlatten").output_shape[1:], name="De_Reshape")(decoder)
    decoder = Conv2DTranspose(32, (3,3), strides=1, activation="relu", padding="same")(decoder)
    decoder = Add()([decoder, skipConnection02_d])
    decoder = Conv2DTranspose(1, (3,3), strides=2, activation="relu", padding="same", name="De_ConvTr_01")(decoder)
    decoder = Conv2DTranspose(32, (3,3), strides=1, activation="relu", padding="same")(decoder)
    decoder = Conv2DTranspose(64, (3,3), strides=1, activation="relu", padding="same")(decoder)
    decoder = Add()([decoder, skipConnection01_d])
    decoder = Conv2DTranspose(1, (3,3), strides=2, activation="relu", padding="same", name="De_ConvTr_02")(decoder)
    decoder = Conv2DTranspose(64, (3,3), strides=1, activation="relu", padding="same")(decoder)
    salida = Conv2D(N_CHANNELS, (1, 1), padding="same", name="Salida")(decoder)

    decoder = Model([cuelloBotellaDE, skipConnection01_d,skipConnection02_d], salida)
    
    autoencoder = Model(entrada, decoder(encoder_u(entrada)))


    return encoder, decoder, autoencoder


# def generar_modelo_06(shapeEntrada:tuple):

    entrada = Input(shape=(shapeEntrada), name="Entrada")
    encoder = Conv2D(64, (3, 3), strides=1, activation="relu", padding="same")(entrada)
    encoder = MaxPooling2D((2, 2), strides=2, name="En_MP_01")(encoder)
    skipConnection01 = encoder
    encoder = Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(encoder)
    encoder = MaxPooling2D((2,2), strides=2, name="En_MP_02")(encoder)
    skipConnection02 = encoder
    encoder = Conv2D(16, (3, 3), activation="sigmoid", padding="same", name="EntradaFlatten")(encoder)
    encoder = Flatten(name="Aplanar")(encoder)
    cuelloBotellaEN = Dense(20, name="En_LatentSpace")(encoder)

    encoder = Model(entrada, cuelloBotellaEN)

    # ------------------------------------------------------------------------------------------- #

    # decoder
    cuelloBotellaDE = Input(shape=(cuelloBotellaEN.shape[1:]), name="De_LatentSpace")
    decoder = Dense(encoder.get_layer(name="Aplanar").output_shape[1], activation="sigmoid", name="De_Dense_Reshape")(cuelloBotellaDE)
    decoder = Reshape(encoder.get_layer(name="EntradaFlatten").output_shape[1:], name="De_Reshape")(decoder)
    decoder = Conv2DTranspose(32, (3,3), strides=1, activation="relu", padding="same")(decoder)
    decoder = Conv2DTranspose(1, (3,3), strides=2, activation="relu", padding="same", name="De_ConvTr_01")(decoder)
    decoder = Concatenate()([decoder, skipConnection02])
    decoder = Conv2DTranspose(32, (3,3), strides=1, activation="relu", padding="same")(decoder)
    decoder = Conv2DTranspose(64, (3,3), strides=1, activation="relu", padding="same")(decoder)
    decoder = Conv2DTranspose(1, (3,3), strides=2, activation="relu", padding="same", name="De_ConvTr_02")(decoder)
    decoder = Concatenate()([decoder, skipConnection01])
    decoder = Conv2DTranspose(64, (3,3), strides=1, activation="relu", padding="same")(decoder)
    salida = Conv2D(N_CHANNELS, (1, 1), padding="same", name="Salida")(decoder)

    decoder = Model(cuelloBotellaDE, salida)
    

    autoencoder = Model(entrada, decoder(encoder(entrada)))


    return encoder, decoder, autoencoder


# def generar_modelo_07(shapeEntrada:tuple): # RECONSTRUYE BIEN

    entrada = Input(shape=(shapeEntrada), name="Entrada")
    encoder = Conv2D(64, (3, 3), strides=1, activation="relu", padding="same")(entrada)
    encoder = MaxPooling2D((2, 2), strides=2, name="En_MP_01")(encoder)
    skipConnection01_e = encoder
    encoder = Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(encoder)
    encoder = MaxPooling2D((2,2), strides=2, name="En_MP_02")(encoder)
    skipConnection02_e = encoder
    encoder = Conv2D(32, (3, 3), activation="relu", padding="same")(encoder)
    encoder = MaxPooling2D((2,2), strides=2, name="En_MP_03")(encoder)
    encoder = Conv2D(32, (3, 3), activation="relu", padding="same")(encoder)
    encoder = MaxPooling2D((2,2), strides=2, name="En_MP_04")(encoder)
    encoder = Conv2D(16, (3, 3), activation="sigmoid", padding="valid")(encoder)
    encoder = Conv2D(16, (3, 3), activation="sigmoid", padding="valid", name="EntradaFlatten")(encoder)
    encoder = Flatten(name="Aplanar")(encoder)
    cuelloBotellaEN = Dense(10, name="En_LatentSpace")(encoder)

    encoder = Model(entrada, cuelloBotellaEN)
    encoder_u = Model(entrada,[cuelloBotellaEN, skipConnection01_e, skipConnection02_e])

    # ------------------------------------------------------------------------------------------- #

    # decoder
    cuelloBotellaDE = Input(shape=(cuelloBotellaEN.shape[1:]), name="De_LatentSpace")
    skipConnection01_d = Input(shape=(skipConnection01_e.shape[1:]), name="De_sk01")
    skipConnection02_d = Input(shape=(skipConnection02_e.shape[1:]), name="De_sk02")
    decoder = Dense(encoder.get_layer(name="Aplanar").output_shape[1], activation="sigmoid", name="De_Dense_Reshape")(cuelloBotellaDE)
    decoder = Reshape(encoder.get_layer(name="EntradaFlatten").output_shape[1:], name="De_Reshape")(decoder)
    decoder = Conv2DTranspose(16, (3,3), strides=1, activation="sigmoid", padding="valid")(decoder)
    decoder = Conv2DTranspose(32, (3,3), strides=1, activation="sigmoid", padding="valid")(decoder)
    decoder = Conv2DTranspose(32, (3,3), strides=2, activation="relu", padding="same")(decoder)
    decoder = Conv2DTranspose(32, (3,3), strides=1, activation="relu", padding="same")(decoder)
    decoder = Conv2DTranspose(32, (3,3), strides=2, activation="relu", padding="same")(decoder)
    # decoder = Conv2DTranspose(32, (3,3), strides=1, activation="relu", padding="same", name="De_ConvTr_01")(decoder)
    decoder = Add()([decoder, skipConnection02_d])
    decoder = Conv2DTranspose(64, (3,3), strides=2, activation="relu", padding="same")(decoder)
    # decoder = Conv2DTranspose(64, (3,3), strides=1, activation="relu", padding="same")(decoder)
    decoder = Add()([decoder, skipConnection01_d])
    decoder = Conv2DTranspose(1, (3,3), strides=2, activation="relu", padding="same", name="De_ConvTr_02")(decoder)
    decoder = Conv2DTranspose(64, (3,3), strides=1, activation="relu", padding="same")(decoder)
    salida = Conv2D(N_CHANNELS, (1, 1), padding="same", name="Salida")(decoder)

    decoder = Model([cuelloBotellaDE, skipConnection01_d,skipConnection02_d], salida)
    

    autoencoder = Model(entrada, decoder(encoder_u(entrada)))


    return encoder, decoder, autoencoder


modelos = {
    "CAE2D_01": generar_modelo_01,
    "CAE2D_02": generar_modelo_02,
    "CAE2D_03": generar_modelo_03,
    "CAE2D_04": generar_modelo_04,
}


# =========================================================================================== #
# DATOS

# Cargamos las características
muestras = obtener_dataset(PATH_MUESTRAS + FILE_MUESTRAS, "muestras")

muestras = muestras[...,Caracteristicas[CARACTERISTICA].value] # Solo cogemos la caracteristica deseada
if (CARACTERISTICA != "ALL"):
    muestras = np.expand_dims(muestras, axis=-1)
shape = muestras.shape
print("muestras.shape:", shape)

# Ponemos las muestras en formato (N_MUESTRAS x N_HORZ x N_VERT x N_CHANNELS)
muestras = muestras.reshape((-1, shape[-3], shape[-2], shape[-1]))
N_MUESTRAS = muestras.shape[0]
N_CHANNELS = muestras.shape[-1]


# =========================================================================================== #
# DATASETS DE TRABAJO

# Barajamos los datos
# Dividimos los datos según las particiones proporcionadas
reset_seeds(SEED)

# Xtrain, Xval, Xtest = train_val_test_split(muestras, PROP_TRAIN, PROP_VAL, PROP_TEST)

kfold = KFold(5, shuffle=True, random_state=SEED)

for k, (Xtrain, Xval) in enumerate(kfold.split(muestras)):

# print("datos Train: ", Xtrain.shape)
# print("datos Val: ", Xval.shape)
# print("datos Test:  ", Xtest.shape)


# =========================================================================================== #
# AUTOENCODER

    K.clear_session()

    encoder, decoder, ae = modelos[MODELO](muestras.shape[1:]) # Hacemos las llamadas desde un diccionario

# ------------------------------------------------------------------------------------------- #

    encoder.summary()
    print("\n\n\n")
    decoder.summary()
    print("\n\n\n")
    ae.summary()

    # Compilamos el modelo
    ae.compile(loss="mean_squared_error", optimizer="adam")

    # Entrenamos el modelo
    entrenamiento_modelo(ae, muestras[Xtrain], muestras[Xtrain], muestras[Xval], muestras[Xval], N_EPOCH, BATCH_SIZE, (ES))

    # Para poder utilizar el mismo script y distinguir ambos experimentos
    if "THR" in FILE_MUESTRAS:
        MODELO = MODELO + "_THR"

    # Guardamos un log de los errores para poder compararlo luego
    nombreLogErrores = PATH_LOGS + f"Loss_{CARACTERISTICA}_{MODELO}_{LS_DIMS}_K{k+1}.csv"
    history = ae.history
    np.savetxt(nombreLogErrores, np.c_[history.history["loss"], history.history["val_loss"]])


# =========================================================================================== #
# PROYECCIÓN DE LOS DATOS

    # Proyección de los datos en el espacio latente
    muestras_ls = encoder.predict(muestras)
    
    print("MUESTRAS LS: ", muestras_ls.shape)
    
    # Reducción de la dimensión (usando UMAP): 20D -> 2D
    if LS_DIMS != 2:
        muestras_ls = muestras_ls.reshape((N_MUESTRAS, -1))
        proyeccionInicial = PCA(n_components=2, random_state=SEED).fit_transform(muestras_ls)
        modelo_umap = umap.UMAP(n_neighbors=UMAP_NEIGHBORS, min_dist=UMAP_MINDIST, random_state=SEED, init=proyeccionInicial)
        puntos_2d = modelo_umap.fit_transform(muestras_ls)
        print("MUESTRAS LS + UMAP: ", puntos_2d.shape)
    else:
        puntos_2d = muestras_ls


    plot_loss_history(history, PATH_FIGURES + f"Loss_{CARACTERISTICA}_{MODELO}_{LS_DIMS}_K{k+1}.png")

    # RECONSTRUCCION DE LOS DATOS
    Xtrain_r = ae.predict(muestras[Xtrain])
    Xval_r = ae.predict(muestras[Xval])
    # Xtest_r = ae.predict(muestras[Xtest])

    # Para poder visualizar las imágenes fácilmente cogemos solo la primera característica
    # No es lo óptimo pero visualizar todas las características de varias muestras es inviable
    if N_CHANNELS > 1: 
        Xtrain = Xtrain[:,:,:,0]
        Xtrain_r = Xtrain_r[:,:,:,0]
        Xval = Xval[:,:,:,0]
        Xval_r = Xval_r[:,:,:,0]
        # Xtest = Xtest[:,:,:,0]
        # Xtest_r = Xtest_r[:,:,:,0]

    # Gráficas mostrando reconstrucciones
    plot_imagen_datos(
        muestras[Xtrain], 
        Xtrain_r, 
        "Entrenamiento", 
        "Originales", 
        "Decodificadas", 
        PATH_FIGURES + f"Imagen_Entrenamiento_{CARACTERISTICA}_{MODELO}_{LS_DIMS}_K{k+1}.png",
    )

    plot_imagen_datos(
        muestras[Xval], 
        Xval_r, 
        "Validación", 
        "Originales", 
        "Decodificadas",
        PATH_FIGURES + f"Imagen_Validacion_{CARACTERISTICA}_{MODELO}_{LS_DIMS}_K{k+1}.png",
    )

    # plot_imagen_datos(
    #     muestras[Xtest], 
    #     Xtest_r, 
    #     "Test", 
    #     "Originales", 
    #     "Decodificadas",
    #     PATH_FIGURES + "Imagen_Test_{CARACTERISTICA}_{MODELO}_{LS_DIMS}_K{k+1}.png",
    # )

    plot_grafica_datos(
        muestras[Xtrain], 
        Xtrain_r, 
        "Entrenamiento", 
        "Original", 
        "Decodificada",
        PATH_FIGURES + f"Grafica_Entrenamiento_{CARACTERISTICA}_{MODELO}_{LS_DIMS}_K{k+1}.png",
    )

    plot_grafica_datos(
        muestras[Xval], 
        Xval_r, 
        "Validación", 
        "Original", 
        "Decodificada",
        PATH_FIGURES + f"Grafica_Validacion_{CARACTERISTICA}_{MODELO}_{LS_DIMS}_K{k+1}.png",
    )

    # plot_grafica_datos(
    #     Xtest, 
    #     Xtest_r, 
    #     "Test", 
    #     "Original", 
    #     "Decodificada",
    #     PATH_FIGURES + "Grafica_Test_{CARACTERISTICA}_{MODELO}_{LS_DIMS}_K{k+1}.png",
    # )

    f = plt.figure()
    plt.title("Proyección del espacio latente")
    plt.scatter(puntos_2d[:,0], puntos_2d[:,1])
    f.savefig(PATH_FIGURES + f"ProyeccionLS_{CARACTERISTICA}_{MODELO}_{LS_DIMS}_K{k+1}.png")

    # plt.show()
    plt.close('all') # Para liberar la memoria


    # Guardamos los datos de las proyecciones
    f1 = h5py.File(PATH_PROYECCIONES + f"{CARACTERISTICA}_{MODELO}_proyecciones_{LS_DIMS}_K{k+1}.hdf5", "w")
    print("Introducimos las proyecciones...")
    dset1 = f1.create_dataset("proyeccion", data=puntos_2d, compression="gzip")
    dset2 = f1.create_dataset("espacioLatente", data=muestras_ls, compression="gzip")
    dset3 = f1.create_dataset("muestras", data=muestras, compression="gzip")
    f1.close()


    # Guardamos los modelos del autoencoder y el UMAP si existe
    print("Guardamos los modelos")
    encoder.save(PATH_MODELOS + f"encoder_{CARACTERISTICA}_{MODELO}_{LS_DIMS}_K{k+1}.keras", overwrite=True)
    decoder.save(PATH_MODELOS + f"decoder_{CARACTERISTICA}_{MODELO}_{LS_DIMS}_K{k+1}.keras", overwrite=True)
    if LS_DIMS > 2:
        pickle.dump(modelo_umap, open(PATH_MODELOS + f"UMAP_{CARACTERISTICA}_{MODELO}_{LS_DIMS}_K{k+1}.sav", "wb"))
    print("Terminamos de guardar los modelos")

# =========================================================================================== #