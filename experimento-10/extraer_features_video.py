import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO


# ===============================
# CONFIGURACIÓN
# ===============================

DATASET_PATH = Path(r"C:\Users\User\Desktop\EXP11\dataset_videos")
OUTPUT_CSV = Path(r"C:\Users\User\Desktop\EXP11\features_videos.csv")

CLASES = ["normal", "robo"]

# Cantidad de frames que tomaremos de cada video
# No procesamos todos para que no sea tan pesado
NUM_FRAMES = 40

# Modelo pose
model = YOLO("yolov8n-pose.pt")


# ===============================
# FUNCIONES
# ===============================

def seleccionar_frames_uniformes(video_path, num_frames=40):
    """
    Selecciona frames distribuidos a lo largo de todo el video.
    Mantiene el orden temporal.
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"No se pudo abrir: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return []

    posiciones = np.linspace(0, total_frames - 1, num_frames).astype(int)

    frames = []

    for pos in posiciones:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()

        if ret:
            frames.append(frame)

    cap.release()
    return frames


def obtener_persona_principal(result):
    """
    Si YOLOv8-Pose detecta varias personas,
    elegimos la persona con mayor área de caja.
    Esa será considerada la persona principal.
    """
    if result.keypoints is None or result.boxes is None:
        return None

    keypoints = result.keypoints.xy

    if keypoints is None or len(keypoints) == 0:
        return None

    boxes = result.boxes.xyxy

    if boxes is None or len(boxes) == 0:
        return None

    areas = []

    for box in boxes:
        x1, y1, x2, y2 = box.cpu().numpy()
        area = (x2 - x1) * (y2 - y1)
        areas.append(area)

    idx_principal = int(np.argmax(areas))

    kpts = keypoints[idx_principal].cpu().numpy()

    return kpts


def normalizar_keypoints(kpts):
    """
    Normaliza los keypoints para reducir el efecto
    del tamaño y posición de la persona en la imagen.
    """
    kpts = np.array(kpts, dtype=np.float32)

    # kpts tiene forma: 17 puntos x 2 coordenadas
    # Si no hay puntos válidos, devolvemos None
    if kpts.shape != (17, 2):
        return None

    # Puntos con coordenadas mayores a 0
    validos = (kpts[:, 0] > 0) & (kpts[:, 1] > 0)

    if validos.sum() < 5:
        return None

    puntos_validos = kpts[validos]

    centro = puntos_validos.mean(axis=0)
    kpts_centrados = kpts - centro

    escala = np.linalg.norm(puntos_validos.max(axis=0) - puntos_validos.min(axis=0))

    if escala == 0:
        return None

    kpts_normalizados = kpts_centrados / escala

    return kpts_normalizados


def extraer_keypoints_video(video_path):
    """
    Extrae keypoints normalizados de varios frames del video.
    Devuelve una secuencia ordenada:
    frame 1 → keypoints
    frame 2 → keypoints
    frame 3 → keypoints
    ...
    """
    frames = seleccionar_frames_uniformes(video_path, NUM_FRAMES)

    secuencia = []

    for frame in frames:
        results = model(frame, verbose=False)

        if len(results) == 0:
            continue

        persona = obtener_persona_principal(results[0])

        if persona is None:
            continue

        persona_norm = normalizar_keypoints(persona)

        if persona_norm is None:
            continue

        secuencia.append(persona_norm)

    return secuencia


def calcular_features_temporales(secuencia):
    """
    Convierte la secuencia completa de keypoints de un video
    en un vector numérico fijo.

    No mezcla videos. Solo resume la secuencia del mismo video.
    """
    if len(secuencia) < 5:
        return None

    arr = np.array(secuencia)  # forma: frames x 17 x 2

    # Posiciones promedio de keypoints durante el video
    promedio = arr.mean(axis=0).flatten()

    # Variación de posturas durante el video
    desviacion = arr.std(axis=0).flatten()

    # Movimiento entre frames consecutivos
    diferencias = np.diff(arr, axis=0)
    movimiento_promedio = np.mean(np.abs(diferencias), axis=0).flatten()
    movimiento_maximo = np.max(np.abs(diferencias), axis=0).flatten()

    # Features específicas de manos
    # Índices COCO:
    # 5 hombro izq, 6 hombro der
    # 7 codo izq, 8 codo der
    # 9 muñeca izq, 10 muñeca der
    # 11 cadera izq, 12 cadera der

    muneca_izq = arr[:, 9, :]
    muneca_der = arr[:, 10, :]
    hombro_izq = arr[:, 5, :]
    hombro_der = arr[:, 6, :]
    cadera_izq = arr[:, 11, :]
    cadera_der = arr[:, 12, :]

    torso = (hombro_izq + hombro_der + cadera_izq + cadera_der) / 4

    dist_muneca_izq_torso = np.linalg.norm(muneca_izq - torso, axis=1)
    dist_muneca_der_torso = np.linalg.norm(muneca_der - torso, axis=1)

    features_manos = np.array([
        dist_muneca_izq_torso.mean(),
        dist_muneca_izq_torso.std(),
        dist_muneca_izq_torso.max(),
        dist_muneca_der_torso.mean(),
        dist_muneca_der_torso.std(),
        dist_muneca_der_torso.max(),
    ])

    # Unimos todo en un solo vector
    features = np.concatenate([
        promedio,
        desviacion,
        movimiento_promedio,
        movimiento_maximo,
        features_manos
    ])

    return features


# ===============================
# PROCESAMIENTO PRINCIPAL
# ===============================

filas = []

for clase in CLASES:
    carpeta = DATASET_PATH / clase
    videos = list(carpeta.glob("*.*"))

    print(f"\nProcesando clase: {clase}")
    print(f"Cantidad de videos: {len(videos)}")

    for i, video_path in enumerate(videos, start=1):
        print(f"[{i}/{len(videos)}] Procesando: {video_path.name}")

        secuencia = extraer_keypoints_video(video_path)

        features = calcular_features_temporales(secuencia)

        if features is None:
            print(f"Saltado por pocos keypoints válidos: {video_path.name}")
            continue

        fila = {
            "video": video_path.name,
            "clase": clase,
            "frames_validos": len(secuencia)
        }

        for j, valor in enumerate(features):
            fila[f"f{j}"] = valor

        filas.append(fila)


df = pd.DataFrame(filas)
df.to_csv(OUTPUT_CSV, index=False)

print("\nProceso terminado.")
print(f"Archivo generado: {OUTPUT_CSV}")
print(f"Total de videos procesados correctamente: {len(df)}")
print(df["clase"].value_counts())