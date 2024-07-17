
# EXPERIMENTO DV2-C80-T1

- Enventanado:
    - Cuadrícula
    - Número de flechas: 80 x 80
    - Número de píxels: 320 x 320


- Extracción de características
    - Descriptores vectoriales formato bidimensional
    - Tamaño de ventana de regresión: 10 x 10
    - 1 características: Rotacional


- Temporalidad:
    - Número de Frames: t  ->  t


## Reproducción del experimento

Para poder reproducir el experimento correctamente se han de seguir los siguientes pasos:

1. Ejecutar con la terminal y desde el directorio principal el archivo `01_Generacion_Muestra_Cuadricula.py`

2. Una vez finalizado ejecutar, nuevamente desde el directorio principal, uno de los dos archivos que comienzan con "02_" dependiendo del escalado que se desee usar

3. Para ejecutar el entrenamiento y obtener los modelos, logs y figuras se ha de ejecutar el archivo `Ejecucion_Experimentos.py` con el siguiente comando en la terminal: `python ./Ejecucion_Experimentos.py --numexp=2`

4. Habiendo terminado el entrenamiento de los modelos, si se desea comprobar los resultados presentados en las tablas de la memoria para el MSE, ejecutar el archivo `Analisis_Error.py` con el siguiente comando en la terminal: `python ./Analisis_Error.py --numexp=2`

5. Si se quieren obtener las figuras presentadas en la memoria se han de ejecutar los archivos `04_VisuReconstruccion.py`, `04_Scatter_AS_LS.py`, `04_Recorrido_Proyeccion.py` y `04_Scatter_Flechas.py`.

Todas las figuras generadas se guardarán en el directorio "figures" dentro del directorio del "Experimento_02".

Existen otros archivos para generar visualizaciones adicionales a las mostradas en la memoria del proyecto, se han de ejecutar de la misma forma que las otras, pudiendo encontrarse las figuras resultantes en el directorio "figures" dentro del directorio del "Experimento_02".
