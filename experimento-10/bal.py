import os
import cv2
from pathlib import Path

DATASET_PATH = Path("dataset_videos")

clases = ["normal", "robo"]

for clase in clases:
    carpeta = DATASET_PATH / clase
    videos = list(carpeta.glob("*.*"))

    print(f"\nClase: {clase}")
    print(f"Cantidad de archivos: {len(videos)}")

    duraciones = []

    for video_path in videos:
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print("No se pudo abrir:", video_path)
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if fps > 0:
            duracion = frames / fps
            duraciones.append(duracion)

        cap.release()

    if duraciones:
        print(f"Duración mínima: {min(duraciones):.2f} segundos")
        print(f"Duración máxima: {max(duraciones):.2f} segundos")
        print(f"Duración promedio: {sum(duraciones)/len(duraciones):.2f} segundos")