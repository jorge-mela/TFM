"""
+-------------------------------------------------------------------------------------------+
| Script para lanzar los experimentos mostrados en el apartado resultados de la memoria.    |
| Precisa de tener los scripts para dichos experimentos descargados, manteniendo el nombre  |
| de los directorios por debajo de él.                                                      |
|                                                                                           | 
| Autor: Jorge Menéndez Lagunilla                                                           | 
| Fecha: 11/2023                                                                            | 
|                                                                                           | 
+-------------------------------------------------------------------------------------------+ 
"""


# =========================================================================================== #
# LIBRERÍAS

import argparse  # Para poder ejecutar el script con comandos desde la terminal
import os


# =========================================================================================== #
# PARÁMETROS

# Definimos un diccionario con los parámetros correspondientes a cada experimento
experimentos = {
    "Experimento1": (
        "/home/jorgemel/motilidad2/Experimento_01/03_Entrenamiento_KFOLD.py",  # Path hasta el experimento
        ("CAE1D_01", "CAE1D_02"),  # Nombre de los modelos a usar
        (5, 10, 15, 20),  # Dimensiones del espacio latente
    ),
    "Experimento2": (
        "/home/jorgemel/motilidad2/Experimento_02/03_Entrenamiento_KFOLD.py",
        ("CAE2D_01", "CAE2D_02", "CAE2D_03", "CAE2D_04"),
        (5, 10, 15, 20),
    ),
    "Experimento3": (
        "/home/jorgemel/motilidad2/Experimento_03/03_Entrenamiento_KFOLD.py",
        ("CAE2D_01", "CAE2D_02", "CAE2D_03"),
        (5, 10),
    ),
    "Experimento4": (
        "/home/jorgemel/motilidad2/Experimento_04/03_Entrenamiento_KFOLD.py",
        ("CAE2D_01", "CAE2D_02", "CAE2D_03"),
        (25,),
    ),
        "Experimento5": (
        "/home/jorgemel/motilidad2/Experimento_05/03_Entrenamiento_KFOLD.py",
        ("Deep_01", "Deep_02", "Deep_03"), 
        (2, 3),
    ),
}


# =========================================================================================== #
# MAIN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Script para ejecutar los experimentos, bien uno a uno o todos a la vez. Al ejecutar el script,
        se ha de añadir --numexp=i donde i es el número del experimento a realizar.
        """
    )
    parser.add_argument(
        "-n",
        "--numexp",
        required=True,
        type=int,
        help="0 para realizar todos los experimentos, 1-N para uno en concreto",
    )
    args = parser.parse_args()

    selExp = "Experimento" + str(
        args.numexp
    )  # Ojo, los comandos de más de un caracter los devuelve como lista

    PATH_FILE, MODELOS, LS_DIMS = experimentos[selExp]

    for modelo in MODELOS:
        for dim in LS_DIMS:
            os.system(f"python {PATH_FILE} --model={modelo} --dims={dim}")
