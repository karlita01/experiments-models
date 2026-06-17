import cv2
import numpy as np
import joblib
from pathlib import Path
from ultralytics import YOLO


# ===============================
# CONFIGURACIÓN
# ===============================

VIDEO_PATH = Path(r"C:\Users\User\Desktop\EXP11\video.mp4")

MODEL_SVM_PATH = Path(r"C:\Users\User\Desktop\EXP11\modelo_svm_robo.pkl")
ENCODER_PATH = Path(r"C:\Users\User\Desktop\EXP11\label_encoder.pkl")

NUM_FRAMES = 40
MIN_KEYPOINTS_VALIDOS = 5

pose_model = YOLO("yolov8n-pose.pt")

svm_model = joblib.load(MODEL_SVM_PATH)
encoder = joblib.load(ENCODER_PATH)


# ===============================
# FUNCIONES
# ===============================

def normalizar_keypoints(kpts):
    kpts = np.array(kpts, dtype=np.float32)

    if kpts.shape != (17, 2):
        return None

    validos = (kpts[:, 0] > 0) & (kpts[:, 1] > 0)

    if validos.sum() < 5:
        return None

    puntos_validos = kpts[validos]

    centro = puntos_validos.mean(axis=0)
    kpts_centrados = kpts - centro

    escala = np.linalg.norm(
        puntos_validos.max(axis=0) - puntos_validos.min(axis=0)
    )

    if escala == 0:
        return None

    kpts_normalizados = kpts_centrados / escala

    return kpts_normalizados


def seleccionar_persona_principal(result):
    """
    Usa ByteTrack si hay ID disponible.
    Si hay varias personas, selecciona la persona con mayor área.
    """
    if result.keypoints is None or result.boxes is None:
        return None, None, None

    if result.keypoints.xy is None:
        return None, None, None

    keypoints = result.keypoints.xy
    boxes = result.boxes.xyxy

    if len(keypoints) == 0 or len(boxes) == 0:
        return None, None, None

    areas = []

    for box in boxes:
        x1, y1, x2, y2 = box.cpu().numpy()
        area = (x2 - x1) * (y2 - y1)
        areas.append(area)

    idx = int(np.argmax(areas))

    kpts = keypoints[idx].cpu().numpy()
    box = boxes[idx].cpu().numpy()

    track_id = None
    if result.boxes.id is not None:
        track_id = int(result.boxes.id[idx].cpu().numpy())

    return kpts, box, track_id


def calcular_features_temporales(secuencia):
    if len(secuencia) < MIN_KEYPOINTS_VALIDOS:
        return None

    arr = np.array(secuencia)  # frames x 17 x 2

    promedio = arr.mean(axis=0).flatten()
    desviacion = arr.std(axis=0).flatten()

    diferencias = np.diff(arr, axis=0)

    movimiento_promedio = np.mean(np.abs(diferencias), axis=0).flatten()
    movimiento_maximo = np.max(np.abs(diferencias), axis=0).flatten()

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

    features = np.concatenate([
        promedio,
        desviacion,
        movimiento_promedio,
        movimiento_maximo,
        features_manos
    ])

    return features


def predecir_video(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("No se pudo abrir el video:", video_path)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        print("El video no tiene frames válidos.")
        return

    posiciones_muestreo = np.linspace(
        0,
        total_frames - 1,
        NUM_FRAMES
    ).astype(int)

    posiciones_muestreo = set(posiciones_muestreo)

    secuencia = []

    frame_idx = 0

    print("Procesando video...")
    print("Presiona Q para cerrar la ventana.")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # YOLOv8-Pose con ByteTrack
        results = pose_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        result = results[0]

        # Frame anotado para mostrar en ventana
        frame_mostrado = result.plot()

        kpts, box, track_id = seleccionar_persona_principal(result)

        # Solo guardamos keypoints en frames seleccionados
        if frame_idx in posiciones_muestreo and kpts is not None:
            kpts_norm = normalizar_keypoints(kpts)

            if kpts_norm is not None:
                secuencia.append(kpts_norm)

        # Dibujar texto de persona principal
        if box is not None:
            x1, y1, x2, y2 = box.astype(int)

            texto_id = f"Persona principal"
            if track_id is not None:
                texto_id += f" | ID: {track_id}"

            cv2.putText(
                frame_mostrado,
                texto_id,
                (x1, max(y1 - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.rectangle(
                frame_mostrado,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                3
            )

        cv2.putText(
            frame_mostrado,
            f"Frames validos para prediccion: {len(secuencia)}/{NUM_FRAMES}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow("Seguimiento YOLOv8-Pose + ByteTrack", frame_mostrado)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord("q"):
            print("Ventana cerrada por el usuario.")
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    print("\nFrames válidos usados:", len(secuencia))

    features = calcular_features_temporales(secuencia)

    if features is None:
        print("No se pudo predecir: pocos keypoints válidos.")
        return

    X = features.reshape(1, -1)

    pred = svm_model.predict(X)[0]
    probas = svm_model.predict_proba(X)[0]

    clase_predicha = encoder.inverse_transform([pred])[0]

    print("\n===============================")
    print("RESULTADO FINAL")
    print("===============================")
    print("Video:", video_path.name)
    print("Predicción:", clase_predicha.upper())

    print("\nProbabilidades:")
    for clase, proba in zip(encoder.classes_, probas):
        print(f"{clase}: {proba:.4f}")

    if clase_predicha == "robo":
        print("\nALERTA: posible acción de robo detectada.")
    else:
        print("\nEl video fue clasificado como actividad normal.")


# ===============================
# EJECUTAR
# ===============================

predecir_video(VIDEO_PATH)