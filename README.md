# Experimentos de clasificación de actividad normal y posible robo

## Experimento 1: Clasificación de escenas completas con YOLOv8-CLS

En este experimento se realizó una primera prueba para clasificar imágenes como **normal** o **robo**. Para ello, se extrajeron frames de videos previamente organizados por clase y se creó un dataset de imágenes. Luego, se entrenó un modelo **YOLOv8** en su versión de clasificación usando los frames completos de la escena. Finalmente, se probó el modelo con una imagen individual para verificar si podía predecir la clase correspondiente y mostrar su nivel de confianza.

## Experimento 2: Clasificación con recortes de personas y dataset balanceado

En este segundo experimento se mejoró el enfoque anterior recortando primero a las personas detectadas en cada frame mediante **YOLOv8**. Con estos recortes se creó un nuevo dataset centrado en la persona, reduciendo parte del ruido visual del fondo. Después, se balancearon las clases **normal** y **robo** para tener una cantidad similar de imágenes, y se entrenó nuevamente un modelo **YOLOv8** de clasificación. Finalmente, se realizaron pruebas tanto en imágenes como en video, clasificando cada frame como **normal** o **posible robo**.
