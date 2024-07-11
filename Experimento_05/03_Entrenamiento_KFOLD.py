"""
+-------------------------------------------------------------------------------------------+
| Entrenamiento de los modelos de ML a partir de las células segmentadas.                   |
|                                                                                           |
| Autor: Jorge Menéndez Lagunilla                                                           |
| Fecha: 02/2024                                                                            |
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

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import umap
from cv2 import cartToPolar
from keras import backend as K
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.initializers import he_normal
from keras.layers import (Conv2D, Conv2DTranspose, Conv3D, Conv3DTranspose,
                          Cropping2D, Dense, Flatten, Input, LeakyReLU,
                          MaxPooling2D, Reshape, UpSampling2D)
from keras.models import Model
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split

sys.path.insert(0, '/home/jorgemel/motilidad2/')    
from libs.generales import calcula_error, obtener_dataset, reset_seeds
from libs.visualizacion import (plot_grafica_datos, plot_imagen_datos,
                                plot_loss_history)


# =========================================================================================== #
# PARÁMETROS

parser = argparse.ArgumentParser(
    description="Script para el entrenamiento con 2D rotdiv atemporal"
)

parser.add_argument("--model", required=True, type=str)
# parser.add_argument("--seed", required=True, type=int)
parser.add_argument("--dims", required=True, type=int)

args = parser.parse_args()

PATH_MUESTRAS = "./Experimento_05/data/02_muestras.hdf5"
PATH_LOGS = "./Experimento_05/logs/"
PATH_FIGURES = "./Experimento_05/figures/"
PATH_PROYECCIONES = "./Experimento_05/proyecciones/"
PATH_MODELOS = "./Experimento_05/models/"

MODELO = "".join(args.model)

LS_DIMS = int(args.dims)
N_EPOCH = 300
BATCH_SIZE = 16

UMAP_NEIGHBORS = 150
UMAP_MINDIST = 0.4

TSNE_NITER = 3000
TSNE_PERPLEXITY = 50

# PROP_TEST = 0.1
# PROP_VAL = 0.2
# PROP_TRAIN = 0.7
SEED = 42  # Fijamos la semilla del random (reproducibilidad)


# Callbacks del entrenamiento
ES = EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=60,
    verbose=1,
    mode="auto",
    baseline=0.5,
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


# =========================================================================================== #
# MODELO

def deep_relu(shapeEntrada:tuple):

    nChannels = shapeEntrada[-1]

    entrada = Input(shape=(shapeEntrada), name="Input")
    encoder = Dense(64, activation="relu", kernel_initializer=he_normal(SEED))(entrada)
    # encoder = Dense(64, activation="relu", kernel_initializer=he_normal(SEED))(encoder)
    encoder = Dense(32, activation="relu", kernel_initializer=he_normal(SEED))(encoder)
    # encoder = Dense(32, activation="relu", kernel_initializer=he_normal(SEED))(encoder)
    encoder = Dense(16, activation="relu", kernel_initializer=he_normal(SEED))(encoder)
    cuelloBotellaEN = Dense(LS_DIMS, activation=None, name="CuelloBotellaEN")(encoder)

    encoder = Model(entrada, cuelloBotellaEN)

    cuelloBotellaDE = Input(shape=cuelloBotellaEN.shape[1:], name="CuelloBotellaDE")
    decoder = Dense(16, activation="relu", kernel_initializer=he_normal(SEED))(cuelloBotellaDE)
    # decoder = Dense(32, activation="relu", kernel_initializer=he_normal(SEED))(decoder)
    decoder = Dense(32, activation="relu", kernel_initializer=he_normal(SEED))(decoder)
    # decoder = Dense(64, activation="relu", kernel_initializer=he_normal(SEED))(decoder)
    decoder = Dense(64, activation="relu", kernel_initializer=he_normal(SEED))(decoder)
    salida = Dense(nChannels, activation=None)(decoder)

    decoder = Model(cuelloBotellaDE, salida)
    autoencoder = Model(entrada, decoder(encoder(entrada)))

    return encoder, decoder, autoencoder


def deep_sigmoid(shapeEntrada:tuple):

    nChannels = shapeEntrada[-1]

    entrada = Input(shape=(shapeEntrada), name="Input")
    encoder = Dense(64, activation="sigmoid")(entrada)
    # encoder = Dense(64, activation="sigmoid")(encoder)
    encoder = Dense(32, activation="sigmoid")(encoder)
    # encoder = Dense(32, activation="sigmoid")(encoder)
    encoder = Dense(16, activation="sigmoid")(encoder)
    cuelloBotellaEN = Dense(LS_DIMS, activation=None, name="CuelloBotellaEN")(encoder)

    encoder = Model(entrada, cuelloBotellaEN)

    cuelloBotellaDE = Input(shape=cuelloBotellaEN.shape[1:], name="CuelloBotellaDE")
    decoder = Dense(16, activation="sigmoid")(cuelloBotellaDE)
    # decoder = Dense(32, activation="sigmoid")(decoder)
    decoder = Dense(32, activation="sigmoid")(decoder)
    # decoder = Dense(64, activation="sigmoid")(decoder)
    decoder = Dense(64, activation="sigmoid")(decoder)
    salida = Dense(nChannels, activation=None)(decoder)

    decoder = Model(cuelloBotellaDE, salida)
    autoencoder = Model(entrada, decoder(encoder(entrada)))

    return encoder, decoder, autoencoder


def deep_leaky_relu(shapeEntrada:tuple):

    nChannels = shapeEntrada[-1]
    slope = 0.3

    entrada = Input(shape=(shapeEntrada), name="Input")
    encoder = Dense(64, activation=None, kernel_initializer=he_normal(SEED))(entrada)
    encoder = LeakyReLU(alpha=slope)(encoder)
    encoder = Dense(32, activation=None, kernel_initializer=he_normal(SEED))(encoder)
    encoder = LeakyReLU(alpha=slope)(encoder)
    encoder = Dense(16, activation=None, kernel_initializer=he_normal(SEED))(encoder)
    encoder = LeakyReLU(alpha=slope)(encoder)
    cuelloBotellaEN = Dense(LS_DIMS, activation=None, name="CuelloBotellaEN")(encoder)

    encoder = Model(entrada, cuelloBotellaEN)

    cuelloBotellaDE = Input(shape=cuelloBotellaEN.shape[1:], name="CuelloBotellaDE")

    decoder = Dense(16, activation=None, kernel_initializer=he_normal(SEED))(cuelloBotellaDE)
    decoder = LeakyReLU(alpha=slope)(decoder)
    decoder = Dense(32, activation=None, kernel_initializer=he_normal(SEED))(decoder)
    decoder = LeakyReLU(alpha=slope)(decoder)
    decoder = Dense(64, activation=None, kernel_initializer=he_normal(SEED))(decoder)
    decoder = LeakyReLU(alpha=slope)(decoder)
    salida = Dense(nChannels, activation=None)(decoder)

    decoder = Model(cuelloBotellaDE, salida)
    autoencoder = Model(entrada, decoder(encoder(entrada)))

    return encoder, decoder, autoencoder



modelos = {
    "Deep_01" : deep_relu,
    "Deep_02" : deep_sigmoid,
    "Deep_03" : deep_leaky_relu,
}


# =========================================================================================== #
# DATOS

# Cargamos las características
# allDatos = [file for file in os.listdir(PATH_MUESTRAS) if not file.startswith('.')]
# allDatos.sort()

# allDatos = ["vid_08_SDHB_RES_128.hdf5", "vid_09_SDHB_RES_128.hdf5"]

muestras = obtener_dataset(PATH_MUESTRAS, "muestras")
shape = muestras.shape
print("muestras.shape: ", shape)

N_MUESTRAS = muestras.shape[0]
N_CHANNELS = muestras.shape[-1]


# =========================================================================================== #
# DATASETS DE TRABAJO

# Barajamos los datos
# Test y Train  Dividimos los datos según las particiones proporcionadas
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


    plot_loss_history(history, PATH_FIGURES + f"Loss_{MODELO}_{LS_DIMS}_K{k+1}.png")

    # RECONSTRUCCION DE LOS DATOS
    Xtrain_r = ae.predict(muestras[Xtrain])
    Xval_r = ae.predict(muestras[Xval])
    # Xtest_r = ae.predict(muestras[Xtest])

    plot_grafica_datos(
        muestras[Xtrain], 
        Xtrain_r, 
        "Entrenamiento", 
        "Original", 
        "Decodificada",
        PATH_FIGURES + f"Grafica_Entrenamiento_{MODELO}_{LS_DIMS}_K{k+1}.png", 
    )

    plot_grafica_datos(
        muestras[Xval], 
        Xval_r, 
        "Validación", 
        "Original", 
        "Decodificada",
        PATH_FIGURES + f"Grafica_Validacion_{MODELO}_{LS_DIMS}_K{k+1}.png",
    )

    # plot_grafica_datos(
    #     Xtest, 
    #     Xtest_r, 
    #     "Test", 
    #     "Original",
    #     "Decodificada",
    #     "../capturasCodigo/EntrenamientoTestGRF.png",
    # )

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
    dset6 = f1.create_dataset("muestras", data=muestras, compression="gzip")
    f1.close()


    # Guardamos los modelos del autoencoder y el UMAP
    print("Guardamos los modelos")
    encoder.save(PATH_MODELOS + f"encoder_{MODELO}_{LS_DIMS}_K{k+1}.keras", overwrite=True)
    decoder.save(PATH_MODELOS + f"decoder_{MODELO}_{LS_DIMS}_K{k+1}.keras", overwrite=True)
    if LS_DIMS > 2:
        pickle.dump(modelo_umap, open(PATH_MODELOS + f"UMAP_{MODELO}_{LS_DIMS}_K{k+1}.sav", "wb"))
    print("Terminamos de guardar los modelos")

# =========================================================================================== #