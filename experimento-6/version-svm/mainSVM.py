from ultralytics import YOLO
import cv2
import joblib
import numpy as np
import pandas as pd
from collections import deque

pose_model = YOLO("yolov8n-pose.pt")
clf = joblib.load("clasificador_pose_svm_normalizado.pkl")

CLASS_NAMES = ["normal", "robo"]

video_path = "video.mp4"

feature_cols = clf.named_steps["imputer"].feature_names_in_

KEYPOINT_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]

KP = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

def safe_point(keypoints, name):
    return keypoints[KP[name]].astype(float)

def midpoint(p1, p2):
    return (p1 + p2) / 2.0

def normalize_pose_keypoints(keypoints):
    keypoints = np.asarray(keypoints, dtype=float)

    left_shoulder = safe_point(keypoints, "left_shoulder")
    right_shoulder = safe_point(keypoints, "right_shoulder")
    left_hip = safe_point(keypoints, "left_hip")
    right_hip = safe_point(keypoints, "right_hip")

    shoulder_mid = midpoint(left_shoulder, right_shoulder)
    hip_mid = midpoint(left_hip, right_hip)

    body_center = midpoint(shoulder_mid, hip_mid)

    torso_scale = np.linalg.norm(shoulder_mid - hip_mid)
    shoulder_scale = np.linalg.norm(left_shoulder - right_shoulder)

    scale = torso_scale if torso_scale > 1 else shoulder_scale

    if scale <= 1:
        scale = 1.0

    features = {}

    for i, name in enumerate(KEYPOINT_NAMES):
        x, y = keypoints[i]
        features[f"{name}_x_norm"] = (x - body_center[0]) / scale
        features[f"{name}_y_norm"] = (y - body_center[1]) / scale

    pairs = [
        ("left_wrist", "left_hip"),
        ("right_wrist", "right_hip"),
        ("left_wrist", "right_hip"),
        ("right_wrist", "left_hip"),
        ("left_wrist", "left_shoulder"),
        ("right_wrist", "right_shoulder"),
        ("left_wrist", "right_wrist"),
        ("left_elbow", "left_hip"),
        ("right_elbow", "right_hip"),
    ]

    for a, b in pairs:
        pa = safe_point(keypoints, a)
        pb = safe_point(keypoints, b)
        features[f"dist_{a}_to_{b}"] = np.linalg.norm(pa - pb) / scale

    features["torso_dx"] = (shoulder_mid[0] - hip_mid[0]) / scale
    features["torso_dy"] = (shoulder_mid[1] - hip_mid[1]) / scale

    return features

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("No se pudo abrir el video.")
    exit()

historial_robo = deque(maxlen=10)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    results = pose_model.predict(frame, conf=0.25, verbose=False)
    result = results[0]

    annotated_frame = result.plot()

    label_text = "SIN POSE"
    color = (255, 255, 255)
    conf_text = ""

    if result.keypoints is not None and len(result.keypoints.xy) > 0:
        keypoints = result.keypoints.xy[0].cpu().numpy()

        features = normalize_pose_keypoints(keypoints)

        X_video = pd.DataFrame([features])
        X_video = X_video[feature_cols]

        proba = clf.predict_proba(X_video)[0]

        prob_normal = proba[0]
        prob_robo = proba[1]

        pred_frame = 1 if prob_robo >= 0.60 else 0
        historial_robo.append(pred_frame)

        robos_recientes = sum(historial_robo)

        if robos_recientes >= 6:
            label_text = "ROBO"
            color = (0, 0, 255)
        else:
            label_text = "NORMAL"
            color = (0, 255, 0)

        conf_text = f"Robo:{prob_robo:.2f} Normal:{prob_normal:.2f}"

    cv2.putText(
        annotated_frame,
        f"{label_text} {conf_text}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        3
    )

    cv2.imshow("YOLOv8 Pose + SVM Normalizado", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()