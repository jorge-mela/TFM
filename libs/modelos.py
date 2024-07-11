'''
Clases para los modelos de machine learning que se han usado.

Autor: Jorge Menéndez Lagunilla

'''

# =========================================================================================== #
# LIBRERIAS
from numpy import ndarray
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import (BatchNormalization, Conv1D, Conv1DTranspose, Conv2D,
                          Conv2DTranspose, Conv3D, Conv3DTranspose, Cropping2D,
                          Dense, Flatten, Input, LeakyReLU, MaxPooling2D,
                          Reshape, UpSampling2D)
from keras.losses import MeanSquaredError
from keras.models import Model

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



class __Main():
    def _main():
        texto = (
            '''
    En este fichero se definen clases para los modelos utilizados.
            '''
        )
        print(texto)

        return None

if "__main__" == __name__:
    __Main()._main()