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

## Experimento 6: Clasificación con YOLOv8 Pose y SVM

En este experimento se utilizó YOLOv8 Pose para detectar los puntos corporales de las personas en video. A partir de estos keypoints se generaron características normalizadas, como posiciones relativas del cuerpo y distancias entre articulaciones. Luego, estas características fueron evaluadas con un clasificador SVM para diferenciar entre una actividad normal y una posible situación de robo. Además, se aplicó un pequeño historial de predicciones para evitar que una sola detección aislada cambie directamente el resultado final.

## Mejora del Experimento 6: YOLOv8 Pose con XGBoost y ByteTrack

En esta mejora se reemplazó el clasificador SVM por un modelo XGBoost y se incorporó seguimiento de personas mediante ByteTrack. Esto permitió asignar un identificador a cada persona detectada y mantener un historial de predicciones por individuo. De esta manera, la clasificación se volvió más estable, ya que el sistema no analiza solo un frame aislado, sino el comportamiento reciente de cada persona durante el video.

## Experimento 7: Mejora del modelo YOLOv8s con aumento de datos y entrenamiento optimizado

En este experimento se mejoró el entrenamiento del modelo de detección utilizando YOLOv8s y un dataset con las clases normal y robo. Primero se revisó el balance de clases del dataset y luego se aplicó aumento de datos sobre la clase robo, generando variaciones con cambios de brillo, contraste, ruido, rotación y volteo horizontal. Esto permitió aumentar la cantidad de ejemplos de la clase robo y reducir el desbalance del dataset.

Después, se entrenó una nueva versión del modelo con una configuración más completa, usando 100 épocas, optimizador AdamW, augmentación interna de YOLO, entrenamiento con GPU y guardado de métricas. Finalmente, se realizaron pruebas en imágenes y videos, incluyendo una versión donde se detectan personas, se les asigna un ID mediante tracking y se clasifica cada recorte como normal o robo.

# Experimento 8: YOLOv8 Pose + XGBoost

Este experimento implementa un flujo para detectar actividad normal o sospechosa usando YOLOv8 Pose y XGBoost. Primero, se extraen los keypoints corporales de las personas presentes en el dataset y se guardan en un archivo CSV. Luego, estos puntos se usan como características para entrenar un clasificador XGBoost capaz de diferenciar entre actividad normal y actividad sospechosa.

También se incluye un script de inferencia que permite probar el modelo entrenado en imágenes, videos o cámara en vivo. El sistema detecta personas, obtiene sus keypoints, clasifica cada detección como normal o sospechosa y dibuja el resultado en pantalla con su respectiva etiqueta y nivel de confianza.

## Experimento 9: Limpieza de etiquetas y entrenamiento YOLOv8 Detect

En este experimento se realizó una prueba de entrenamiento con YOLOv8 para detectar las clases normal y robo usando un dataset en formato YOLO. Primero se verificó la estructura del dataset, la disponibilidad de GPU y se ejecutó un entrenamiento inicial corto con YOLOv8n para comprobar que el entorno funcionaba correctamente.

Luego, se entrenó una versión con YOLOv8s y se revisaron los archivos de etiquetas del dataset. Durante esta revisión se identificó que algunos labels no estaban en formato de detección con caja, sino que tenían más coordenadas, similares a segmentación o polígonos. Por ello, se creó una versión limpia del dataset convirtiendo esos polígonos a bounding boxes compatibles con YOLO Detect. Finalmente, se volvió a entrenar YOLOv8s usando el dataset corregido y se validó el modelo sobre el conjunto de prueba.

## Experimento 10: Clasificación temporal de videos con YOLOv8 Pose, SVM y XGBoost

En este experimento se trabajó directamente con videos completos de las clases normal y robo. Primero, se analizaron los videos del dataset para revisar la cantidad de archivos y sus duraciones. Luego, se extrajeron frames distribuidos uniformemente de cada video y se utilizó YOLOv8 Pose para obtener los keypoints corporales de la persona principal.

A partir de la secuencia de keypoints se generaron características temporales, como la postura promedio, variación de movimientos, movimiento entre frames y distancias de las muñecas respecto al torso. Estas características fueron guardadas en un CSV y utilizadas para entrenar modelos SVM y XGBoost. Finalmente, se implementó una prueba de inferencia sobre un video completo, donde el sistema selecciona la persona principal, acumula sus keypoints y realiza una predicción final indicando si el video corresponde a actividad normal o posible robo.
