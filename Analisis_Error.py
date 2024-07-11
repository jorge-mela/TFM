"""
+-------------------------------------------------------------------------------------------+
| Script que obtiene las métricas presentadas en las tablas presentadas en el apartado de   |
| "resultados" de la memoria a partir de los logs de entrenamiento de los distintos         |
| expoerimentos.                                                                            |
|                                                                                           | 
| Autor: Jorge Menéndez Lagunilla                                                           | 
| Fecha: 11/2023                                                                            | 
|                                                                                           | 
+-------------------------------------------------------------------------------------------+ 
"""


# =========================================================================================== #
# LIBRERÍAS

import argparse
import os

import numpy as np
import pandas as pd


# =========================================================================================== #
# PARÁMETROS

parser = argparse.ArgumentParser(
        description="""
        Script para mostrar una tabla comparativa de los modelos del experimento selecionado.
        Al ejecutar el script, se ha de añadir --numexp=i donde i es el número del
        experimento a realizar.
        """
    )
parser.add_argument("--numexp", required=True, type=str)
args = parser.parse_args()

ARCHIVOS =  {
    "1": "./Experimento_01/logs/",
    "2": "./Experimento_02/logs/",
    "3": "./Experimento_03/logs/",
    "4": "./Experimento_04/logs/",
    "5": "./Experimento_05/logs/",
}

PATH_ARCHIVOS = ARCHIVOS[args.numexp]

N_KFOLDS = 5

LS_DIMS = ("02", "03", "05", "10", "15", "20", "25") # Valores ordenados de las dimensiones de los espacios latentes



# =========================================================================================== #
#  FUNCIONES


# =========================================================================================== #
#  MAIN

allFiles = [f for f in os.listdir(PATH_ARCHIVOS) if not f.startswith('.')]
allFiles.sort()

nModelos = len(allFiles) // N_KFOLDS

SEPARADOR = " & "

# Imprimimos el título de la tabla
titulo = "Nombre   LS Dims   K1      K2       K3       K4       K5     Media     Sigma"
print("\n")
print(titulo)
print('='*len(titulo))

# Estructura prevista
# datosModelos = {
#     "MODELO_A" : {
#         "LS_DIM_1" : [error_k1, ..., error_kN]
#         ...
# 
#         "LS_DIM_M" : [error_k1, ..., error_kN]
#     }
#     ...
# 
#     "MODELO_Z" : {
#         "LS_DIM_1" : [error_k1, ..., error_kN]
#         ...
#         "LS_DIM_M" : [error_k1, ..., error_kN]
#     }
# }

# Generamos un diccionario para almacenar la información de cada modelo
datosModelos = {}

# Iteramos para aglutinar la información de todos los modelos entrenados por CV
for i in range(0, len(allFiles), N_KFOLDS):
    
    nombre = "_".join(allFiles[i].split(sep='_')[1:-2])
    dims = allFiles[i].split(sep='_')[-2]

    # Por estética
    if int(dims) < 10:
        dims = '0' + dims

    valErrors_ = []

    for j in range(N_KFOLDS):
        df = pd.read_csv(PATH_ARCHIVOS+allFiles[i+j], sep=' ', header=None, names=("Train", "Val")) # Leemos los logs por columnas
        valErrors_.append(df["Val"].min())

    # Si el nombre no está introducido, lo metemos
    if nombre not in datosModelos.keys():
        datosModelos[nombre] = {}
    
    # Asignamos los errores de los distintos Folds de cada dimensión
    # datosModelos[nombre][dims] = valErrors_
    datosModelos[nombre][dims] = valErrors_


# Lo mostramos por pantalla ligeramente formateado para una tabla de LaTEX
for nombreModelo in datosModelos.keys():
    for d in LS_DIMS:
        linea = nombreModelo + " - " + d

        try:
            for e in datosModelos[nombreModelo][d]:
                linea = linea + SEPARADOR + "{:5.4f}".format(e)
            # {:5.4f}
            linea = linea + SEPARADOR + "{:5.4f}".format(np.mean(datosModelos[nombreModelo][d]))
            linea = linea + SEPARADOR + "{:3.2e}".format(np.std(datosModelos[nombreModelo][d]))
            linea = linea + " \\\\"
                
            print()
            print(linea)
            print()
            print('-'*len(linea))
            
        except KeyError:
            pass


print("\n")