# Experimentos de clasificación de actividad normal y posible robo

## Experimento 1: Clasificación de escenas completas con YOLOv8-CLS

En este experimento se realizó una primera prueba para clasificar imágenes como **normal** o **robo**. Para ello, se extrajeron frames de videos previamente organizados por clase y se creó un dataset de imágenes. Luego, se entrenó un modelo **YOLOv8** en su versión de clasificación usando los frames completos de la escena. Finalmente, se probó el modelo con una imagen individual para verificar si podía predecir la clase correspondiente y mostrar su nivel de confianza.

## Experimento 2: Clasificación con recortes de personas y dataset balanceado

En este segundo experimento se mejoró el enfoque anterior recortando primero a las personas detectadas en cada frame mediante **YOLOv8**. Con estos recortes se creó un nuevo dataset centrado en la persona, reduciendo parte del ruido visual del fondo. Después, se balancearon las clases **normal** y **robo** para tener una cantidad similar de imágenes, y se entrenó nuevamente un modelo **YOLOv8** de clasificación. Finalmente, se realizaron pruebas tanto en imágenes como en video, clasificando cada frame como **normal** o **posible robo**.

## Experimento 3: Detección de posible robo con YOLOv8 Detect

En este experimento se entrenó un modelo YOLOv8 en modo detección utilizando un dataset local anotado mediante el archivo `data.yaml`. A diferencia de los experimentos anteriores, donde el modelo clasificaba una imagen completa como normal o robo, en este caso se utilizó YOLOv8 para detectar visualmente la posible acción o zona asociada al robo dentro de una imagen o video.

Para el entrenamiento se utilizó el modelo base `yolov8m.pt`, configurado con 30 épocas, tamaño de imagen de 640, batch de 8 y paciencia de 10. Luego, el modelo entrenado fue probado en imágenes y videos, mostrando las detecciones realizadas con sus respectivas cajas y niveles de confianza. Este experimento permitió evaluar un enfoque basado en detección directa, aunque su rendimiento depende bastante de la calidad de las anotaciones y del dataset usado para entrenar.

## Experimento 4: Detección de robo con YOLOv8s y análisis Grad-CAM

En este experimento se entrenó un modelo YOLOv8s en modo detección utilizando un dataset anotado con dos clases: normal y robo. A diferencia de los primeros experimentos basados solo en clasificación, aquí el modelo aprende a detectar visualmente las regiones asociadas a cada clase mediante cajas delimitadoras. Además, se realizó una validación del entrenamiento revisando métricas como pérdidas, precisión y mAP, y se probó el modelo entrenado sobre imágenes y videos.

Como complemento, se aplicó Grad-CAM al modelo entrenado para visualizar qué zonas de la imagen influyen más en la predicción de la clase robo. Esto permitió agregar una parte de interpretabilidad al experimento, ya que no solo se observa la detección final del modelo, sino también las regiones visuales que activan su decisión.

## Experimento 5: Análisis visual de escenas con YOLOv11 y reglas simples

En este experimento se utilizó un modelo YOLOv11 preentrenado para analizar imágenes del dataset y detectar objetos presentes en escenas normales y de robo. Primero se revisó la distribución del dataset y se visualizaron muestras por clase. Luego, se realizaron predicciones con diferentes umbrales de confianza para observar qué objetos detectaba el modelo en cada imagen.

Además, se implementó una clasificación simple basada en reglas visuales, considerando la presencia de personas y objetos como mochilas, bolsos o maletas para asignar un puntaje de posible actividad sospechosa o actividad normal. Este experimento permitió explorar si los objetos detectados en la escena podían servir como señales iniciales para identificar comportamientos sospechosos, aunque sin entrenar un modelo específico para robo.
