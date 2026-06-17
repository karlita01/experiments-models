"""
=======================================================
  SHOPLIFTING DETECTION - Inferencia en Imagen/Video
=======================================================

Uso:
  python detect.py --source imagen.jpg
  python detect.py --source video.mp4
  python detect.py --source 0
  python detect.py --source video.mp4 --output resultado.mp4
"""

import cv2
import argparse
import pandas as pd
import xgboost as xgb

from pathlib import Path
from ultralytics import YOLO


# ================================================================
# CONFIGURACIÓN
# ================================================================

YOLO_MODEL_PATH = "yolov8n-pose.pt"
XGBOOST_MODEL_PATH = "models/xgboost_model.json"

CONFIDENCE_THRESH = 0.50
SUSPICIOUS_THRESH = 0.50
NUM_KEYPOINTS = 17

COLOR_SUSPICIOUS = (0, 0, 255)  # Rojo
COLOR_NORMAL = (0, 255, 0)      # Verde


# ================================================================
# CARGA DE MODELOS
# ================================================================

def load_models():
    print("Cargando modelos...")

    pose_model = YOLO(YOLO_MODEL_PATH)

    xgb_model = xgb.Booster()
    xgb_model.load_model(XGBOOST_MODEL_PATH)

    print("Modelos cargados correctamente.\n")

    return pose_model, xgb_model


# ================================================================
# CONVERTIR KEYPOINTS A FEATURES
# ================================================================

def keypoints_to_dataframe(keypoints):
    """
    Convierte los 17 keypoints normalizados de YOLOv8 Pose
    en un DataFrame compatible con el modelo XGBoost.

    Orden usado:
    x0, x1, x2, ..., x16, y0, y1, y2, ..., y16
    """

    row = {}

    for j in range(NUM_KEYPOINTS):
        if j < len(keypoints):
            row[f"x{j}"] = keypoints[j][0]
        else:
            row[f"x{j}"] = 0.0

    for j in range(NUM_KEYPOINTS):
        if j < len(keypoints):
            row[f"y{j}"] = keypoints[j][1]
        else:
            row[f"y{j}"] = 0.0

    return pd.DataFrame([row])


# ================================================================
# PREDICCIÓN POR FRAME
# ================================================================

def predict_frame(frame, pose_model, xgb_model):
    """
    Detecta personas con YOLOv8 Pose, extrae keypoints
    y clasifica cada persona como Normal o Suspicious.
    """

    results = pose_model(frame, verbose=False)

    annotated_frame = frame.copy()
    detections = []

    for result in results:
        if result.boxes is None or result.keypoints is None:
            continue

        boxes = result.boxes.xyxy
        confs = result.boxes.conf.tolist()
        keypoints_list = result.keypoints.xyn.tolist()

        for idx, box in enumerate(boxes):
            if confs[idx] < CONFIDENCE_THRESH:
                continue

            keypoints = keypoints_list[idx]

            df_input = keypoints_to_dataframe(keypoints)
            dmatrix = xgb.DMatrix(df_input)

            prob_suspicious = float(xgb_model.predict(dmatrix)[0])

            if prob_suspicious >= SUSPICIOUS_THRESH:
                label = "Suspicious"
                confidence = prob_suspicious
                color = COLOR_SUSPICIOUS
            else:
                label = "Normal"
                confidence = 1 - prob_suspicious
                color = COLOR_NORMAL

            x1, y1, x2, y2 = map(int, box.tolist())

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            text = f"{label} {confidence:.2f}"

            cv2.putText(
                annotated_frame,
                text,
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            detections.append({
                "class": label,
                "confidence": confidence,
                "box": (x1, y1, x2, y2)
            })

    return annotated_frame, detections


# ================================================================
# DETECCIÓN EN IMAGEN
# ================================================================

def detect_image(image_path, output_path=None):
    pose_model, xgb_model = load_models()

    image = cv2.imread(image_path)

    if image is None:
        print(f"No se pudo leer la imagen: {image_path}")
        return

    annotated_image, detections = predict_frame(image, pose_model, xgb_model)

    for det in detections:
        print(f"{det['class']} - confianza: {det['confidence']:.2f}")

    if output_path is None:
        stem = Path(image_path).stem
        output_path = f"{stem}_detected.jpg"

    cv2.imwrite(output_path, annotated_image)

    print(f"Imagen guardada en: {output_path}")

    cv2.imshow("Resultado", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ================================================================
# DETECCIÓN EN VIDEO O CÁMARA
# ================================================================

def detect_video(source, output_path=None):
    pose_model, xgb_model = load_models()

    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"No se pudo abrir la fuente: {source}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None

    if output_path is None and not str(source).isdigit():
        output_path = f"{Path(source).stem}_detected.mp4"

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_num = 0

    print("Procesando video/cámara...")
    print("Presiona Q para salir.\n")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        annotated_frame, detections = predict_frame(frame, pose_model, xgb_model)

        suspicious_count = sum(1 for d in detections if d["class"] == "Suspicious")
        normal_count = sum(1 for d in detections if d["class"] == "Normal")

        info = f"Frame: {frame_num} | Normal: {normal_count} | Suspicious: {suspicious_count}"

        cv2.putText(
            annotated_frame,
            info,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        if writer:
            writer.write(annotated_frame)

        cv2.imshow("Shoplifting Detection", annotated_frame)

        if total_frames > 0:
            print(f"\rProgreso: {frame_num}/{total_frames}", end="", flush=True)

        frame_num += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

    if writer:
        writer.release()

    cv2.destroyAllWindows()

    print(f"\nFrames procesados: {frame_num}")

    if output_path:
        print(f"Video guardado en: {output_path}")


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Inferencia de actividad sospechosa con YOLOv8 Pose + XGBoost"
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Ruta de imagen, video o 0 para cámara"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Ruta de salida opcional"
    )

    args = parser.parse_args()

    source = args.source

    if source.isdigit():
        detect_video(source, args.output)
        return

    ext = Path(source).suffix.lower()

    if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        detect_image(source, args.output)
    elif ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        detect_video(source, args.output)
    else:
        print("Formato no reconocido. Usa una imagen, video o 0 para cámara.")


if __name__ == "__main__":
    main()