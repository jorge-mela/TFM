'''
Funciones pensadas para la visualización de los datos.

Autor: Jorge Menéndez Lagunilla

'''

# =========================================================================================== #
# LIBRERIAS

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


# =========================================================================================== #
# FUNCIONES

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
    min = 0
    max = 1

    if "X1" in kwargs.keys():
        Xlabel1 = kwargs["Xlabel1"]
    if "X2" in kwargs.keys():
        Xlabel2 = kwargs["Xlabel2"]
    if "Y1" in kwargs.keys():
        Ylabel1 = kwargs["Ylabel1"]
    if "Y2" in kwargs.keys():
        Ylabel2 = kwargs["Ylabel2"]
    if "vmin" in kwargs.keys():
        min = kwargs["vmin"]
    if "vmax" in kwargs.keys():
        max = kwargs["vmax"]
    
    i = np.random.randint(0, len(original))

    f = plt.figure(figsize=(4, 4))
    plt.suptitle(titulo)
    plt.subplot(121)
    plt.imshow(original[i,:,:], cmap="viridis", vmin=min, vmax=max)
    plt.xticks([])
    plt.yticks([])
    plt.title(img1titulo)
    plt.xlabel(Xlabel1)
    plt.ylabel(Xlabel2)

    plt.subplot(122)
    plt.imshow(decoded[i], cmap="viridis", vmin=min, vmax=max)
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
    
    def calcula_error(original:np.ndarray, estimada:np.ndarray, metrica:str): # Funciona para datos unidimensionales

        if metrica == "MSE":
            error = np.sum((estimada - original)**2)/len(estimada)

        if metrica == "RMSE":
            error = np.sqrt(np.sum((estimada - original)**2)/len(estimada))

        return error

    i = np.random.randint(0, len(original))

    slicing = [slice(None)]*len(original.shape)
    slicing[0]=i

    if "canal" in kwargs.keys():
        channel = kwargs["canal"]
        slicing[-1]=channel
    

    error = " RMSE: " + str(round(calcula_error(original[slicing].ravel(), decoded[slicing].ravel(), "RMSE"),2))
    f = plt.figure(figsize=(6, 4))
    plt.plot(np.arange(original[slicing].size), original[slicing].ravel(), "b")
    plt.plot(np.arange(decoded[slicing].size), decoded[slicing].ravel(), "g")


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
    canales: tuple = (),
    path: str = None,
):

    i = np.random.randint(0, len(original))
    
    if len(canales) == 0:
        canales =(0)

    f,axes = plt.subplots(len(canales), 1)
    plt.suptitle(titulo)
    for j, axis in enumerate(axes):
        slicing=[slice(None)]*len(original.shape); slicing[0]=i; slicing[-1]=j
        # error = " RMSE: " + str(np.round(calcula_error(original[slicing], decoded[slicing], "RMSE"), 2))
        axis.plot(np.arange(original[slicing].size), original[slicing].ravel(), "b")
        axis.plot(np.arange(decoded[slicing].size), decoded[slicing].ravel(), "g")
        # axis.set_title((titulo + error))
        # axis.legend([axis1titulo, axis2titulo])
    lines = axes[0].get_children()[:2] # Porque todos son lo mismo y lo que se pinta son las dos primeras líneas
    labels = [axis1titulo, axis2titulo]
    f.legend(lines, labels, loc='upper right', ncol=2)

    if path != None:
        plt.savefig(path)

    return f


def plot_matriz_confusion(predictions:np.ndarray, original:np.ndarray, etiquetas:tuple):

    f = plt.figure()

    cm = confusion_matrix(original, predictions, labels=etiquetas)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=etiquetas)
    disp.plot()

    return f


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