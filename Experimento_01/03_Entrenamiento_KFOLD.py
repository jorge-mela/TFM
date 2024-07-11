"""
+-------------------------------------------------------------------------------------------+
| Entrenamiento de los modelos de ML.                                                       |
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

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import umap
from keras import backend as K
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import (BatchNormalization, Conv1D, Conv1DTranspose, Conv2D,
                          Conv2DTranspose, Conv3D, Conv3DTranspose, Cropping2D,
                          Dense, Flatten, Input, LeakyReLU, MaxPooling2D,
                          Reshape, UpSampling2D)
from keras.losses import MeanSquaredError
from keras.models import Model
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.initializers import he_normal

import sys
sys.path.insert(0, '/home/jorgemel/motilidad2/')
from libs.generales import obtener_dataset, reset_seeds, calcula_error


# =========================================================================================== #
# PARÁMETROS

parser = argparse.ArgumentParser(
    description="Script para ejecutar el entrenamiento con 1Drotdiv temporal"
)

parser.add_argument("--model", required=True, type=str)
# parser.add_argument("--seed", required=True, type=int)
parser.add_argument("--dims", required=True, type=int)

args = parser.parse_args()

PATH_MUESTRAS = "./Experimento_01/data/02_muestras.hdf5"
PATH_LOGS = "./Experimento_01/logs/"
PATH_FIGURES = "./Experimento_01/figures/"
PATH_PROYECCIONES = "./Experimento_01/proyecciones/"
PATH_MODELOS = "./Experimento_01/models/"


MODELO = "".join(args.model) # "CAE1D_01" # input("MODELO: ") # "CAE2D"

LS_DIMS = int(args.dims)# 15
N_EPOCH = 300 
BATCH_SIZE = 16

UMAP_NEIGHBORS = 150
UMAP_MINDIST = 0.4

TSNE_NITER = 3000
TSNE_PERPLEXITY = 50

SEED = 42 # Fijamos la semilla del random (reproducibilidad)


# Callbacks del entrenamiento
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
):  #:np.array, testDecoded:np.array, index:int
    
    Xlabel1 = ""
    Xlabel2 = ""
    Ylabel1 = ""
    Ylabel2 = ""
    vmin = 0
    vmax = 1
    channels = 1

    if "X1" in kwargs.keys():
        Xlabel1 = kwargs["Xlabel1"]
    if "X2" in kwargs.keys():
        Xlabel2 = kwargs["Xlabel2"]
    if "Y1" in kwargs.keys():
        Ylabel1 = kwargs["Ylabel1"]
    if "Y2" in kwargs.keys():
        Ylabel2 = kwargs["Ylabel2"]
    if "vmin" in kwargs.keys():
        vmin = kwargs["vmin"]
    if "vmax" in kwargs.keys():
        vmax = kwargs["vmax"]
    if channels in kwargs.keys():
        channels = kwargs["canales"]
    
    i = np.random.randint(0, len(original))

    f = plt.figure(figsize=(4, 4))
    plt.suptitle(titulo)
    plt.subplot(121)
    plt.imshow(original[i,:,:], cmap="viridis", vmin=vmin, vmax=vmax)
    plt.xticks([])
    plt.yticks([])
    plt.title(img1titulo)
    plt.xlabel(Xlabel1)
    plt.ylabel(Xlabel2)

    plt.subplot(122)
    plt.imshow(decoded[i], cmap="viridis", vmin=vmin, vmax=vmax)
    plt.xticks([])
    plt.yticks([])
    plt.title(img2titulo)
    plt.xlabel(Ylabel1)
    plt.ylabel(Ylabel2)
    plt.tight_layout()

    if path is not None:
        plt.savefig(path)

    return f


def plot_imagen_datos_multiple(
    original: np.ndarray,
    decoded: np.ndarray,
    titulo: str, 
    img1titulo: str,
    img2titulo: str,
    **kwargs
):  #:np.array, testDecoded:np.array, index:int
    
    Xlabel1 = ""
    Xlabel2 = ""
    Ylabel1 = ""
    Ylabel2 = ""
    vmin = 0
    vmax = 1
    channels = 1

    if "X1" in kwargs.keys():
        Xlabel1 = kwargs["Xlabel1"]
    if "X2" in kwargs.keys():
        Xlabel2 = kwargs["Xlabel2"]
    if "Y1" in kwargs.keys():
        Ylabel1 = kwargs["Ylabel1"]
    if "Y2" in kwargs.keys():
        Ylabel2 = kwargs["Ylabel2"]
    if "vmin" in kwargs.keys():
        vmin = kwargs["vmin"]
    if "vmax" in kwargs.keys():
        vmax = kwargs["vmax"]
    if channels in kwargs.keys():
        channels = kwargs["canales"]
    
    i = np.random.randint(0, len(original))

    f, axes = plt.subplots(2, channels)
    plt.suptitle(titulo)
    for j, axis_row in enumerate(axes):
        for k, ax in enumerate(axis_row):
            if j == 0:
                ax.imshow(original[i,:,:,k], cmap="viridis", vmin=vmin, vmax=vmax)
            if j == 1:
                ax.imshow(decoded[i,:,:,k], cmap="viridis", vmin=vmin, vmax=vmax)

        if "path" in kwargs.keys():
            plt.savefig(kwargs["path"])

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
        error = " RMSE: " + str(np.round(calcula_error(original[slicing], decoded[slicing], "RMSE")),2)

        f = plt.figure(figsize=(6, 4))
        plt.plot(np.arange(original[slicing].size), original[slicing].ravel(), "b")
        plt.plot(np.arange(decoded[slicing].size), decoded[slicing].ravel(), "g")
        plt.title(titulo+error)
        plt.legend([axis1titulo, axis2titulo])

    else:
        error = " RMSE: " + str(np.round(calcula_error(original[i], decoded[i], "RMSE"),2))
        f = plt.figure(figsize=(6, 4))
        plt.plot(np.arange(original[i].size), original[i].ravel(), "b")
        plt.plot(np.arange(decoded[i].size), decoded[i].ravel(), "g")
        plt.title(titulo+error)
        plt.legend([axis1titulo, axis2titulo])

    if path is not None:
        plt.savefig(path)

    return f


def plot_grafica_datos_multiple(
    original: np.ndarray,
    decoded: np.ndarray,
    titulo: str,
    axis1titulo: str,
    axis2titulo: str,
    **kwargs
):

    i = np.random.randint(0, len(original))

    channels = 1
    if "channels" in kwargs.keys():
        channels = kwargs["channels"]

    f,axes = plt.subplots(channels,1)
    plt.suptitle(titulo)
    for j, axis in enumerate(axes):
        slicing=[slice(None)]*len(Xtrain.shape); slicing[0]=i; slicing[-1]=j
        # error = " RMSE: " + str(np.round(calcula_error(original[slicing], decoded[slicing], "RMSE"), 2))
        axis.plot(np.arange(original[slicing].size), original[slicing].ravel(), "b")
        axis.plot(np.arange(decoded[slicing].size), decoded[slicing].ravel(), "g")
        # axis.set_title((titulo + error))
        # axis.legend([axis1titulo, axis2titulo])
    lines = axes[0].get_children()[:2] # Porque todos son lo mismo y lo que se pinta son las dos primeras líneas
    labels = [axis1titulo, axis2titulo]
    f.legend(lines, labels, loc='upper right', ncol=2)

    if "path" in kwargs.keys():
        plt.savefig(kwargs["path"])

    return f


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


def calcula_histogramas_canales(datos:np.ndarray, nbins:int):
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
    x = np.zeros(shape=(nbins+1, nCanales))

    for i in range(nCanales):
        h[:,i], x[:,i] = np.histogram(datos[:,i], nbins)

    return (h, x)


def plot_histogramas_canales(h:np.ndarray, x:np.ndarray, titulos:list=None, log:bool=False):
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
            plt.figure()
            plt.bar(x[:-1,channel], h[:,channel], log=True)
    else:
        for channel in range(ncanales):
            plt.figure()
            plt.bar(x[:-1,channel], h[:,channel])

    return None


# =========================================================================================== #
# MODELOS

def cae1D_01(shapeEntrada:tuple):

    nChannels = shapeEntrada[-1]

    # encoder
    entrada = Input(shape=shapeEntrada, name="En_Entrada")
    encoder = Conv1D(256, 3, activation='relu', strides=1, padding='valid')(entrada)
    encoder = Conv1D(128, 3, activation='relu', strides=1, padding='valid')(encoder)
    encoder = Conv1D(64, 3, activation='relu', strides=1, padding='valid', name='EntradaFlatten')(encoder)
    encoder = Flatten(name="Aplanar")(encoder)
    encoder = Dense(256, activation="relu")(encoder)
    encoder = Dense(128, activation="relu")(encoder)
    encoder = Dense(64, activation='relu')(encoder)
    encoder = Dense(32, activation='relu')(encoder)
    cuelloBotellaEN = Dense(LS_DIMS, activation=None, name='En_LatentSpace')(encoder)

    encoder = Model(entrada, cuelloBotellaEN)

    # ------------------------------------------------------------------------------------------- #

    # decoder
    cuelloBotellaDE = Input(shape=(cuelloBotellaEN.shape[1:]), name="De_LatentSpace")
    decoder = Dense(32, activation='relu', name='De_Dense_32')(cuelloBotellaDE)
    decoder = Dense(64,  activation="relu", name="De_Dense_64")(decoder)
    decoder = Dense(128, activation="relu", name="De_Dense_128")(decoder)
    decoder = Dense(256, activation="relu", name="De_Dense_256")(decoder)
    decoder = Dense(encoder.get_layer(name="Aplanar").output_shape[1], activation="relu", name="De_Dense_Reshape")(decoder)
    decoder = Reshape(encoder.get_layer(name="EntradaFlatten").output_shape[1:], name="De_Reshape")(decoder)
    decoder = Conv1DTranspose(128, 3, activation='relu', strides=1, padding='valid')(decoder)
    decoder = Conv1DTranspose(256, 3, activation='relu', strides=1, padding='valid')(decoder)

    salida = Conv1DTranspose(nChannels, 3, activation='relu', strides=1, padding='valid', name='De_Salida')(decoder)

    decoder = Model(cuelloBotellaDE, salida)
    autoencoder = Model(entrada, decoder(encoder(entrada)))

    return encoder, decoder, autoencoder


def cae1D_02(shapeEntrada:tuple):

    nChannels = shapeEntrada[-1]
    
    # encoder
    entrada = Input(shape=shapeEntrada, name="En_Entrada")
    encoder = Conv1D(256, 3, activation='relu', strides=1, padding='valid')(entrada)
    encoder = Conv1D(128, 3, activation='relu', strides=1, padding='valid')(encoder)
    encoder = Conv1D(64, 3, activation='relu', strides=1, padding='valid')(encoder)
    encoder = Conv1D(32, 3, activation='relu', strides=1, padding='valid')(encoder)
    encoder = Conv1D(16, 3, activation='relu', strides=1, padding='valid', name='EntradaFlatten')(encoder)
    encoder = Flatten(name="Aplanar")(encoder)
    encoder = Dense(256, activation="relu")(encoder)
    encoder = Dense(128, activation="relu")(encoder)
    encoder = Dense(64, activation='relu')(encoder)
    encoder = Dense(32, activation='relu')(encoder)
    cuelloBotellaEN = Dense(LS_DIMS, activation=None, name='En_LatentSpace')(encoder)

    encoder = Model(entrada, cuelloBotellaEN)

    # ------------------------------------------------------------------------------------------- #

    # decoder
    cuelloBotellaDE = Input(shape=(cuelloBotellaEN.shape[1:]), name="De_LatentSpace")
    decoder = Dense(32, activation='relu')(cuelloBotellaDE)
    decoder = Dense(64,  activation="relu")(decoder)
    decoder = Dense(128, activation="relu")(decoder)
    decoder = Dense(256, activation="relu")(decoder)
    decoder = Dense(encoder.get_layer(name="Aplanar").output_shape[1], activation="relu", name="De_Dense_Reshape")(decoder)
    decoder = Reshape(encoder.get_layer(name="EntradaFlatten").output_shape[1:], name="De_Reshape")(decoder)
    decoder = Conv1DTranspose(32, 3, activation='relu', strides=1, padding='valid', name='De_Conv1DTr32x3')(decoder)
    decoder = Conv1DTranspose(64, 3, activation='relu', strides=1, padding='valid', name='De_Conv1DTr64x3')(decoder)
    decoder = Conv1DTranspose(128, 3, activation='relu', strides=1, padding='valid', name='De_Conv1DTr128x3')(decoder)
    decoder = Conv1DTranspose(256, 3, activation='relu', strides=1, padding='valid', name='De_Conv1DTr256x3')(decoder)

    salida = Conv1DTranspose(nChannels, 3, activation='relu', strides=1, padding='valid', name='De_Salida')(decoder)

    decoder = Model(cuelloBotellaDE, salida)
    autoencoder = Model(entrada, decoder(encoder(entrada)))

    return encoder, decoder, autoencoder


modelos = {

    "CAE1D_01" : cae1D_01,
    "CAE1D_02" : cae1D_02,

}


# =========================================================================================== #
# DATOS

# Cargamos las características
muestras = obtener_dataset(PATH_MUESTRAS, "muestras")
shape = muestras.shape
print("muestras.shape: ", shape)

# Ponemos las muestras en formato (N_MUESTRAS x NCH x N_CHANNELS)
muestras = muestras.reshape((-1, shape[-2], shape[-1]))
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

    encoder, decoder, ae = modelos[MODELO](muestras.shape[1:])

# ------------------------------------------------------------------------------------------- # 

    encoder.summary()
    print("\n\n\n")
    decoder.summary()
    print("\n\n\n")
    ae.summary()
    print("\n\n\n")

    # Compilamos el modelo
    ae.compile(loss="mean_squared_error", optimizer="adam")

    entrenamiento_modelo(ae, muestras[Xtrain], muestras[Xtrain], muestras[Xval], muestras[Xval], N_EPOCH, BATCH_SIZE, (ES))

    # Guardamos un log de los errores para poder compararlo luego
    nombreLogErrores = PATH_LOGS + f"Loss_{MODELO}_{LS_DIMS}_K{k+1}.csv"
    history = ae.history
    np.savetxt(nombreLogErrores, np.c_[history.history["loss"], history.history["val_loss"]]) # Lo guardamos como columnas


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
    else:
        puntos_2d = muestras_ls


    print("MUESTRAS LS + UMAP: ", puntos_2d.shape)

    plot_loss_history(history, PATH_FIGURES + f"Loss_{MODELO}_{LS_DIMS}_K{k+1}.png")

    # RECONSTRUCCION DE LOS DATOS
    Xtrain_r = ae.predict((muestras[Xtrain]))
    Xval_r = ae.predict((muestras[Xval]))
    # Xtest_r = ae.predict((muestras[Xtest]))


    # Gráficas mostrando las reconstrucciones
    plot_grafica_datos(
        muestras[Xtrain], 
        Xtrain_r, 
        "Entrenamiento", 
        "Original", 
        "Decodificada",
        path= PATH_FIGURES + f"EntrenamientoTrainGRF_{MODELO}_{LS_DIMS}_K{k+1}.png",
        channels=N_CHANNELS,
    )

    plot_grafica_datos(
        muestras[Xval], 
        Xval_r, 
        "Validación", 
        "Original", 
        "Decodificada",
        path= PATH_FIGURES + f"EntrenamientoValidacionGRF_{MODELO}_{LS_DIMS}_K{k+1}.png",
        channels=N_CHANNELS,
    )

    # plot_grafica_datos(
    #     muestras[Xtest], 
    #     Xtest_r, 
    #     "Test", 
    #     "Original", 
    #     "Decodificada",
    #     path= PATH_FIGURES + f"EntrenamientoTestGRF_{MODELO}_{LS_DIMS}_K{k}.png",
    #     channels=N_CHANNELS,
    # )

    # Figura del embedding 2D
    f = plt.figure()
    plt.title("Proyección del espacio latente")
    plt.scatter(puntos_2d[:,0], puntos_2d[:,1])
    f.savefig(PATH_FIGURES + f"ProyeccionLS_{MODELO}_{LS_DIMS}_K{k+1}.png")

    # plt.show()
    plt.close('all') # Para liberar la memoria


    # Guardamos los datos de las proyecciones
    f1 = h5py.File(PATH_PROYECCIONES + f"{MODELO}_proyecciones_{LS_DIMS}_K{k+1}.hdf5", "w")
    print("Introducimos las proyecciones...")
    dset1 = f1.create_dataset("proyeccion", data=puntos_2d, compression="gzip")
    dset2 = f1.create_dataset("espacioLatente", data=muestras_ls, compression="gzip")
    dset3 = f1.create_dataset("muestras", data=muestras, compression="gzip")
    f1.close()


    # Guardamos los modelos del autoencoder y el UMAP si existe
    print("Guardamos los modelos")
    encoder.save(PATH_MODELOS + f"encoder_{MODELO}_{LS_DIMS}_K{k+1}.keras", overwrite=True)
    decoder.save(PATH_MODELOS + f"decoder_{MODELO}_{LS_DIMS}_K{k+1}.keras", overwrite=True)
    if LS_DIMS > 2:
        pickle.dump(modelo_umap, open(PATH_MODELOS + f"UMAP_{MODELO}_{LS_DIMS}_K{k+1}.sav", "wb"))
    print("Terminamos de guardar los modelos")

# =========================================================================================== #