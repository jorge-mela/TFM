
# EXPERIMENTO 1

- Enventanado:
    - Cuadrícula
    - Número de flechas: 75 x 75
    - Número de píxels: 300 x 300


- Extracción de características
    - Descriptores vectoriales formato unidimensional
    - 7 características: Rotacional, Divergencia, Estiramiento en X, Estiramiento en Y, Cortante XY, Bias X, Bias Y


- Temporalidad:
    - Número de Frames: t  ->  (t+15)



## Reproducción del experimento

Para poder reproducir el experimento correctamente se han de seguir los siguientes pasos:

1. Ejecutar con la terminal y desde el directorio principal el archivo `01_Generacion_Muestra_Cuadricula.py`

2. Una vez finalizado ejecutar, nuevamente desde el directorio principal, uno de los dos archivos que comienzan con "02_" dependiendo del escalado que se desee usar

3. Para ejecutar el entrenamiento y obtener los modelos, logs y figuras se ha de ejecutar el archivo `Ejecucion_Experimentos.py` con el siguiente comando en la terminal: `python ./Ejecucion_Experimentos.py --numexp=1`

4. Habiendo terminado el entrenamiento de los modelos, si se desea comprobar los resultados presentados en las tablas de la memoria para el MSE, ejecutar el archivo `Analisis_Error.py` con el siguiente comando en la terminal: `python ./Analisis_Error.py --numexp=1`

5. Si se quieren obtener las figuras presentadas en la memoria se han de ejecutar los archivos `04_VisuReconstruccion.py`, `04_Scatter_AS_LS.py` y `04_VisuQuiver.py`. En el caso del último script, se ha de especificar dentro del mismo, en la zona de parámetros, las coordenadas para las que se desea visualizar la muestra más cercana, debido a la necesidad de mostrar 16 campos de velocidad por muestra.

Todas las figuras generadas se guardarán en el directorio "figures" dentro del directorio del "Experimento_01".

Existen otros archivos para generar visualizaciones adicionales a las mostradas en la memoria del proyecto, se han de ejecutar de la misma forma que las otras, pudiendo encontrarse las figuras resultantes en el directorio "figures" dentro del directorio del "Experimento_01".