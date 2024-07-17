# TFM: Evaluación de la motilidad celular mediante el análisis de vídeos de time-lapse de cultivos
### Autor: Jorge Menéndez Lagunilla
### Tutor: José María Enguita González
### Tutor: Ignacio Díaz Blanco


_Estructura del repositorio_

El repositorio se encuentra ordenado en directorios para cada experimento, correspondientes con los experimentos descritos en la memoria. Para facilitar el manejo de los mismos desde python se ajustaron los nombres de los directorios, para clarificación se muestran las equivalencias entre los nombres:
- Experimento_01 = Experimento DV1-C75-T16
- Experimento_02 = Experimento DV2-C80-T1
- Experimento_03 = Experimento DV2-S40-T1
- Experimento_04 = Experimento FFT2-S38-T1
- Experimento_05 = Experimento DV1-S38-T1

Dentro de cada experimento se encuentran distintos directorios con los que interactuarán los diferentes scripts. Se adjunta una pequeña descripción de la finalidad de cada uno:

- *data:* En este directorio se guardan los archivos que contienen datos, generados tras la ejecución de los scritps con prefijos "01_" y "02_" en el nombre.
- *figures:* Aquí se almacenan las figuras procedentes del entrenamiento de los diferentes modelos así como las figuras generadas por los scripts de visualización (aquellos con prefijo "04_"). Dentro de todo este conjunto de figuras, se encuentran las comparaciones de la muestra original y la reconstrucción para cada modelo entrenado, tanto para el conjunto de entrenamiento como para el de validación; las curvas de error obtenidas durante el entrenamiento y las proyecciones resultantes de aplicar el UMAP para reducir el espacio latente a 2 dimensiones.
- *logs:* Bajo este directorio se encuentran los datos de las curvas de entrenamiento generadas para cada modelo en formato numérico, por columnas, en archivos .csv.
- *models:* Aquí se encuentran los .keras para poder hacer uso del encoder y decoder de cada modelo siempre que sea necesario, así como los .sav que contienen los datos con los parámetros del UMAP para cada modelo. 
- *proyecciones:* Este directorio contiene los archivos .hdf5 en los que se encuentran los resultados principales de cada modelo: las muestras previamente a ser codificadas por el autoencoder (dataset denominado "muestras"), las mismas muestras codificadas en el espacio latente (dataset denominado "espacioLatente") y esas mismas muestras expresadas en las coordenadas en el espacio bidimensional de la proyección (dataset denominado "proyeccion").


_Ejecución de experimentos_

Si se desea reproducir algún experimento en concreto, se han de seguir los pasos disponibles en los archivos README disponibles dentro de cada uno de los directorios dedicados.

Para poder comparar los valores de error obtenidos por cada modelo dentro de un mismo experimento, se ha de ejecutar el script "Analisis_Error.py" (habiendo previamente realizado el entrenamiento de los autoencoders) desde una terminal con el comando `python ./Analisis_Error.py --numexp=0` modificando el valor del parámetro numexp por el número del experimento correspondiente. Tras esto se podrá ver en la consola una tabla que mostrará el mínimo error obtenido en cada fold, la media de esos errores y la desviación típica de los mismos.


