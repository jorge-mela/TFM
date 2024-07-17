
# EXPERIMENTO FFT2-S38-T1

- Enventanado:
    - Segmentación
    - Número de flechas: 38 x 38
    - Número de píxels: 150 x 150


- Extracción de características
    - Transformada de fourier 2D
    - Enventanado de Hanning
    - Zero padding: 128 x 128


- Temporalidad:
    - Número de Frames: t  ->  t


## Reproducción del experimento

Para poder reproducir el experimento correctamente se han de seguir los siguientes pasos:

1. Ejecutar con la terminal y desde el directorio principal el archivo `01_Generacion_Muestras_Segmentadas.py`

2. Una vez finalizado ejecutar, nuevamente desde el directorio principal, uno de los dos archivos que comienzan con "02_" dependiendo del escalado que se desee usar

3. Para ejecutar el entrenamiento y obtener los modelos, logs y figuras se ha de ejecutar el archivo `Ejecucion_Experimentos.py` con el siguiente comando en la terminal: `python ./Ejecucion_Experimentos.py --numexp=4`

4. Habiendo terminado el entrenamiento de los modelos, si se desea comprobar los resultados presentados en las tablas de la memoria para el MSE, ejecutar el archivo `Analisis_Error.py` con el siguiente comando en la terminal: `python ./Analisis_Error.py --numexp=4`

5. Si se quieren obtener las figuras presentadas en la memoria se han de ejecutar los archivos `04_VisuReconstruccion.py`, `04_Scatter_AS_LS.py` y `04_Scatter_Flechas.py`.

Todas las figuras generadas se guardarán en el directorio "figures" dentro del directorio del "Experimento_04".

Existen otros archivos para generar visualizaciones adicionales a las mostradas en la memoria del proyecto, se han de ejecutar de la misma forma que las otras, pudiendo encontrarse las figuras resultantes en el directorio "figures" dentro del directorio del "Experimento_04".
