import cv2
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from ultralytics import YOLO

# =========================
# CARGAR MODELOS
# =========================
pose_model = YOLO("yolov8n-pose.pt")
clf = joblib.load("clasificador_pose_xgboost_normalizado.pkl")

video_path = "video.mp4"

# =========================
# KEYPOINTS COCO - YOLOv8 POSE
# =========================
KEYPOINT_NAMES = [
    'nose',
    'left_eye', 'right_eye',
    'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle'
]

KP = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

def _safe_point(keypoints, name):
    return keypoints[KP[name]].astype(float)

def _midpoint(p1, p2):
    return (p1 + p2) / 2.0

def normalize_pose_keypoints(keypoints):
    keypoints = np.asarray(keypoints, dtype=float)

    left_shoulder = _safe_point(keypoints, 'left_shoulder')
    right_shoulder = _safe_point(keypoints, 'right_shoulder')
    left_hip = _safe_point(keypoints, 'left_hip')
    right_hip = _safe_point(keypoints, 'right_hip')

    shoulder_mid = _midpoint(left_shoulder, right_shoulder)
    hip_mid = _midpoint(left_hip, right_hip)
    body_center = _midpoint(shoulder_mid, hip_mid)

    torso_scale = np.linalg.norm(shoulder_mid - hip_mid)
    shoulder_scale = np.linalg.norm(left_shoulder - right_shoulder)
    scale = torso_scale if torso_scale > 1 else shoulder_scale

    if scale <= 1:
        scale = 1.0

    features = {}

    for i, name in enumerate(KEYPOINT_NAMES):
        x, y = keypoints[i]
        features[f'{name}_x_norm'] = (x - body_center[0]) / scale
        features[f'{name}_y_norm'] = (y - body_center[1]) / scale

    pairs = [
        ('left_wrist', 'left_hip'),
        ('right_wrist', 'right_hip'),
        ('left_wrist', 'right_hip'),
        ('right_wrist', 'left_hip'),
        ('left_wrist', 'left_shoulder'),
        ('right_wrist', 'right_shoulder'),
        ('left_wrist', 'right_wrist'),
        ('left_elbow', 'left_hip'),
        ('right_elbow', 'right_hip'),
    ]

    for a, b in pairs:
        pa = _safe_point(keypoints, a)
        pb = _safe_point(keypoints, b)
        features[f'dist_{a}_to_{b}'] = np.linalg.norm(pa - pb) / scale

    features['torso_dx'] = (shoulder_mid[0] - hip_mid[0]) / scale
    features['torso_dy'] = (shoulder_mid[1] - hip_mid[1]) / scale

    return features

# =========================
# COLUMNAS DEL ENTRENAMIENTO
# =========================
feature_cols = []

for name in KEYPOINT_NAMES:
    feature_cols.append(f'{name}_x_norm')
    feature_cols.append(f'{name}_y_norm')

for a, b in [
    ('left_wrist', 'left_hip'),
    ('right_wrist', 'right_hip'),
    ('left_wrist', 'right_hip'),
    ('right_wrist', 'left_hip'),
    ('left_wrist', 'left_shoulder'),
    ('right_wrist', 'right_shoulder'),
    ('left_wrist', 'right_wrist'),
    ('left_elbow', 'left_hip'),
    ('right_elbow', 'right_hip'),
]:
    feature_cols.append(f'dist_{a}_to_{b}')

feature_cols += ['torso_dx', 'torso_dy']

# =========================
# HISTORIAL POR PERSONA
# =========================
historial_por_id = defaultdict(lambda: deque(maxlen=10))

# =========================
# SEGUIMIENTO CON BYTETRACK
# =========================
results_generator = pose_model.track(
    source=video_path,
    tracker="bytetrack.yaml",
    conf=0.25,
    persist=True,
    stream=True,
    verbose=False
)

for result in results_generator:
    frame = result.orig_img.copy()

    if result.keypoints is not None and result.boxes is not None:
        keypoints_all = result.keypoints.xy.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()

        if result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy().astype(int)
        else:
            ids = list(range(len(keypoints_all)))

        for i, keypoints in enumerate(keypoints_all):
            person_id = ids[i]
            x1, y1, x2, y2 = boxes[i].astype(int)

            features = normalize_pose_keypoints(keypoints)
            X_person = pd.DataFrame([features])[feature_cols]

            proba = clf.predict_proba(X_person)[0]
            prob_normal = proba[0]
            prob_robo = proba[1]

            pred_frame = 1 if prob_robo >= 0.60 else 0
            historial_por_id[person_id].append(pred_frame)

            robos_recientes = sum(historial_por_id[person_id])

            if robos_recientes >= 6:
                label_text = "ROBO"
                color = (0, 0, 255)
            else:
                label_text = "NORMAL"
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frame,
                f"ID {person_id} | {label_text} | R:{prob_robo:.2f} N:{prob_normal:.2f}",
                (x1, max(y1 - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    cv2.imshow("YOLOv8 Pose + ByteTrack + XGBoost", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()