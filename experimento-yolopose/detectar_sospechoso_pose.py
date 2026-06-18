# ==========================================================
# DETECCION DE ACTIVIDAD SOSPECHOSA
# YOLO-Pose + Reglas + Zonas de estantes dibujables
# ==========================================================
# Controles:
# Click izquierdo  = agregar punto de zona
# Click derecho    = cerrar zona actual
# P                = pausar / continuar
# S                = guardar zonas
# L                = cargar zonas
# C                = limpiar zona actual
# R                = borrar todas las zonas
# K                = mostrar/ocultar keypoints
# Q                = salir
# ==========================================================

import cv2
import numpy as np
import json
import os
import math
import time
from collections import defaultdict, deque
from ultralytics import YOLO


# ==========================================================
# CONFIGURACION
# ==========================================================

VIDEO_PATH = "video5.mp4"          # Cambia por tu video. Para camara usa 0
OUTPUT_PATH = "salida_sospechoso.mp4"
ZONES_PATH = "zonas_estantes.json"

MODEL_PATH = "yolov8n-pose.pt"
TRACKER = "bytetrack.yaml"

CONF = 0.35
IOU = 0.45

SHOW_ALL_KEYPOINTS = True

# Reglas
KP_CONF_MIN = 0.25
MAX_TRANSFER_SECONDS = 2.0
REPEAT_WINDOW_SECONDS = 6.0
MIN_SUSPICIOUS_FRAMES = 8
ALERT_SCORE = 5

# Margen para decir que una persona esta cerca del estante
NEAR_ZONE_MARGIN = 90


# ==========================================================
# INDICES COCO POSE
# ==========================================================

NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


SKELETON = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


# ==========================================================
# ESTADO DE ZONAS
# ==========================================================

editor_state = {
    "zones": [],          # Lista de zonas. Cada zona es lista de puntos [[x,y], [x,y]...]
    "current": [],        # Zona que se esta dibujando
    "show_keypoints": SHOW_ALL_KEYPOINTS
}


# ==========================================================
# FUNCIONES DE ZONAS
# ==========================================================

def save_zones(path=ZONES_PATH):
    data = {
        "video_path": str(VIDEO_PATH),
        "zones": editor_state["zones"]
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"[OK] Zonas guardadas en: {path}")


def load_zones(path=ZONES_PATH):
    if not os.path.exists(path):
        print(f"[AVISO] No existe archivo de zonas: {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    editor_state["zones"] = data.get("zones", [])
    editor_state["current"] = []

    print(f"[OK] Zonas cargadas desde: {path}")
    print(f"[INFO] Total zonas: {len(editor_state['zones'])}")


def close_current_zone():
    if len(editor_state["current"]) >= 3:
        editor_state["zones"].append(editor_state["current"].copy())
        print(f"[OK] Zona agregada. Total zonas: {len(editor_state['zones'])}")
        editor_state["current"] = []
    else:
        print("[AVISO] Necesitas minimo 3 puntos para cerrar una zona.")


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        editor_state["current"].append([x, y])
        print(f"[PUNTO] Agregado: ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        close_current_zone()


def draw_zones(frame):
    # Zonas ya cerradas
    for idx, zone in enumerate(editor_state["zones"]):
        polygon = np.array(zone, dtype=np.int32)

        overlay = frame.copy()
        cv2.fillPoly(overlay, [polygon], (255, 180, 0))
        cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)

        cv2.polylines(frame, [polygon], True, (255, 180, 0), 2)

        x, y = polygon[0]
        cv2.putText(
            frame,
            f"Estante {idx + 1}",
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 180, 0),
            2
        )

    # Zona actual en dibujo
    current = editor_state["current"]

    if len(current) > 0:
        for p in current:
            cv2.circle(frame, tuple(p), 5, (0, 255, 255), -1)

        if len(current) >= 2:
            pts = np.array(current, dtype=np.int32)
            cv2.polylines(frame, [pts], False, (0, 255, 255), 2)

        cv2.putText(
            frame,
            "Dibujando zona: click izq puntos | click der cerrar",
            (20, frame.shape[0] - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )


def point_in_polygon(point, polygon_points):
    if point is None:
        return False

    polygon = np.array(polygon_points, dtype=np.int32)
    x, y = point

    result = cv2.pointPolygonTest(polygon, (float(x), float(y)), False)
    return result >= 0


def point_in_any_zone(point):
    for idx, zone in enumerate(editor_state["zones"]):
        if point_in_polygon(point, zone):
            return True, idx

    return False, -1


def bbox_near_zone(bbox, zone, margin=80):
    """
    Verifica si la caja de la persona esta cerca de una zona.
    Se usa un rectangulo expandido alrededor del poligono.
    """
    x1, y1, x2, y2 = bbox

    polygon = np.array(zone, dtype=np.int32)
    zx, zy, zw, zh = cv2.boundingRect(polygon)

    zx1 = zx - margin
    zy1 = zy - margin
    zx2 = zx + zw + margin
    zy2 = zy + zh + margin

    # Interseccion entre bbox de persona y bbox expandido de zona
    inter_x1 = max(x1, zx1)
    inter_y1 = max(y1, zy1)
    inter_x2 = min(x2, zx2)
    inter_y2 = min(y2, zy2)

    return inter_x2 > inter_x1 and inter_y2 > inter_y1


def bbox_near_any_zone(bbox, margin=80):
    for idx, zone in enumerate(editor_state["zones"]):
        if bbox_near_zone(bbox, zone, margin):
            return True, idx

    return False, -1


# ==========================================================
# FUNCIONES DE POSE
# ==========================================================

def get_keypoint(kpts_xy, kpts_conf, index):
    if kpts_xy is None or len(kpts_xy) <= index:
        return None

    if kpts_conf is not None:
        if kpts_conf[index] < KP_CONF_MIN:
            return None

    x, y = kpts_xy[index]

    if x <= 0 or y <= 0:
        return None

    return np.array([float(x), float(y)])


def midpoint(p1, p2):
    if p1 is None or p2 is None:
        return None

    return (p1 + p2) / 2.0


def distance(p1, p2):
    if p1 is None or p2 is None:
        return None

    return float(np.linalg.norm(p1 - p2))


def torso_angle_degrees(shoulder_center, hip_center):
    """
    Inclinacion del torso respecto a la vertical.
    """
    if shoulder_center is None or hip_center is None:
        return 0.0

    dx = shoulder_center[0] - hip_center[0]
    dy = shoulder_center[1] - hip_center[1]

    if abs(dy) < 1:
        return 0.0

    angle = math.degrees(math.atan2(abs(dx), abs(dy)))
    return angle


def wrist_near_torso(wrist, shoulder_center, hip_center, bbox):
    """
    Verifica si una muñeca esta cerca del torso/cintura.
    """
    if wrist is None or hip_center is None:
        return False

    x1, y1, x2, y2 = bbox

    box_h = max(1, y2 - y1)
    box_w = max(1, x2 - x1)

    dist_to_hip = distance(wrist, hip_center)
    threshold = max(45, 0.20 * box_h)

    near_by_distance = dist_to_hip is not None and dist_to_hip < threshold

    # Caja aproximada de torso/cintura
    torso_x1 = x1 + 0.18 * box_w
    torso_x2 = x2 - 0.18 * box_w
    torso_y1 = y1 + 0.32 * box_h
    torso_y2 = y1 + 0.88 * box_h

    wx, wy = wrist

    inside_torso_box = (
        torso_x1 <= wx <= torso_x2 and
        torso_y1 <= wy <= torso_y2
    )

    return near_by_distance or inside_torso_box


def draw_all_keypoints(frame, kpts_xy, kpts_conf=None):
    """
    Dibuja los 17 puntos corporales.
    """
    # Dibujar conexiones
    for a, b in SKELETON:
        if kpts_conf is not None:
            if kpts_conf[a] < KP_CONF_MIN or kpts_conf[b] < KP_CONF_MIN:
                continue

        xa, ya = kpts_xy[a]
        xb, yb = kpts_xy[b]

        if xa <= 0 or ya <= 0 or xb <= 0 or yb <= 0:
            continue

        cv2.line(
            frame,
            (int(xa), int(ya)),
            (int(xb), int(yb)),
            (255, 255, 255),
            2
        )

    # Dibujar puntos
    for idx, point in enumerate(kpts_xy):
        if kpts_conf is not None and kpts_conf[idx] < KP_CONF_MIN:
            continue

        x, y = point

        if x <= 0 or y <= 0:
            continue

        x, y = int(x), int(y)

        cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)

        cv2.putText(
            frame,
            str(idx),
            (x + 4, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 255, 255),
            1
        )


def clean_old_events(events, current_time, window_seconds):
    while events and current_time - events[0] > window_seconds:
        events.popleft()


# ==========================================================
# ESTADO POR PERSONA
# ==========================================================

track_state = defaultdict(lambda: {
    "last_product_time": None,
    "last_transition_time": None,
    "transitions": deque(),
    "suspicious_frames": 0,
    "last_seen": 0,
    "max_score": 0
})


# ==========================================================
# PROCESAMIENTO DE FRAME
# ==========================================================

def process_frame(frame, model, frame_idx, fps):
    current_time = frame_idx / fps

    annotated = frame.copy()

    # Si no hay zonas, avisamos
    if len(editor_state["zones"]) == 0:
        cv2.putText(
            annotated,
            "Dibuja zonas de estantes con el mouse",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    results = model.track(
        frame,
        persist=True,
        tracker=TRACKER,
        conf=CONF,
        iou=IOU,
        verbose=False
    )

    result = results[0]

    if result.boxes is None or result.keypoints is None:
        return annotated

    boxes = result.boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        return annotated

    if result.boxes.id is not None:
        track_ids = result.boxes.id.int().cpu().tolist()
    else:
        track_ids = list(range(len(boxes)))

    kpts_all_xy = result.keypoints.xy.cpu().numpy()

    if result.keypoints.conf is not None:
        kpts_all_conf = result.keypoints.conf.cpu().numpy()
    else:
        kpts_all_conf = None

    total = min(len(boxes), len(kpts_all_xy), len(track_ids))

    for i in range(total):
        track_id = track_ids[i]

        bbox = boxes[i]
        x1, y1, x2, y2 = bbox.astype(int)

        kpts_xy = kpts_all_xy[i]
        kpts_conf = kpts_all_conf[i] if kpts_all_conf is not None else None

        if editor_state["show_keypoints"]:
            draw_all_keypoints(annotated, kpts_xy, kpts_conf)

        # Keypoints principales
        left_wrist = get_keypoint(kpts_xy, kpts_conf, LEFT_WRIST)
        right_wrist = get_keypoint(kpts_xy, kpts_conf, RIGHT_WRIST)

        left_shoulder = get_keypoint(kpts_xy, kpts_conf, LEFT_SHOULDER)
        right_shoulder = get_keypoint(kpts_xy, kpts_conf, RIGHT_SHOULDER)

        left_hip = get_keypoint(kpts_xy, kpts_conf, LEFT_HIP)
        right_hip = get_keypoint(kpts_xy, kpts_conf, RIGHT_HIP)

        shoulder_center = midpoint(left_shoulder, right_shoulder)
        hip_center = midpoint(left_hip, right_hip)

        # ==================================================
        # REGLAS
        # ==================================================

        near_zone, near_zone_idx = bbox_near_any_zone(
            bbox,
            margin=NEAR_ZONE_MARGIN
        )

        left_hand_in_product, left_zone_idx = point_in_any_zone(left_wrist)
        right_hand_in_product, right_zone_idx = point_in_any_zone(right_wrist)

        hand_in_product = left_hand_in_product or right_hand_in_product

        left_hand_near_torso = wrist_near_torso(
            left_wrist,
            shoulder_center,
            hip_center,
            bbox
        )

        right_hand_near_torso = wrist_near_torso(
            right_wrist,
            shoulder_center,
            hip_center,
            bbox
        )

        hand_near_torso = left_hand_near_torso or right_hand_near_torso

        torso_angle = torso_angle_degrees(shoulder_center, hip_center)
        body_leaning = torso_angle > 18

        state = track_state[track_id]
        state["last_seen"] = current_time

        # Si la mano entro a zona de productos
        if hand_in_product:
            state["last_product_time"] = current_time

        # Transicion: mano en producto -> mano al torso/cintura
        product_to_torso = False

        if state["last_product_time"] is not None and hand_near_torso:
            delta = current_time - state["last_product_time"]

            if 0 <= delta <= MAX_TRANSFER_SECONDS:
                product_to_torso = True

                last_transition = state["last_transition_time"]

                if last_transition is None or current_time - last_transition > 0.8:
                    state["transitions"].append(current_time)
                    state["last_transition_time"] = current_time

        clean_old_events(
            state["transitions"],
            current_time,
            REPEAT_WINDOW_SECONDS
        )

        repeated_pattern = len(state["transitions"]) >= 2

        # ==================================================
        # PUNTAJE
        # ==================================================

        score = 0
        reasons = []

        if near_zone:
            score += 1
            reasons.append("cerca_estante")

        if hand_in_product:
            score += 2
            reasons.append("mano_en_estante")

        if product_to_torso:
            score += 3
            reasons.append("mano_a_torso")

        if body_leaning:
            score += 1
            reasons.append("inclinacion")

        if repeated_pattern:
            score += 2
            reasons.append("patron_repetido")

        state["max_score"] = max(state["max_score"], score)

        # Suavizado
        if score >= ALERT_SCORE or repeated_pattern:
            state["suspicious_frames"] += 1
        else:
            state["suspicious_frames"] = max(0, state["suspicious_frames"] - 1)

        is_suspicious = state["suspicious_frames"] >= MIN_SUSPICIOUS_FRAMES

        # ==================================================
        # DIBUJO DE RESULTADOS
        # ==================================================

        if is_suspicious:
            color = (0, 0, 255)
            label = f"ID {track_id} | POSIBLE SOSPECHA | score {score}"

        elif score >= 3:
            color = (0, 255, 255)
            label = f"ID {track_id} | observando | score {score}"

        else:
            color = (0, 255, 0)
            label = f"ID {track_id} | normal | score {score}"

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            annotated,
            label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

        # Dibujar muñecas destacadas
        for wrist, name in [(left_wrist, "L"), (right_wrist, "R")]:
            if wrist is not None:
                wx, wy = wrist.astype(int)

                cv2.circle(
                    annotated,
                    (wx, wy),
                    7,
                    color,
                    -1
                )

                cv2.putText(
                    annotated,
                    name,
                    (wx + 5, wy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

        # Linea torso
        if shoulder_center is not None and hip_center is not None:
            sc = shoulder_center.astype(int)
            hc = hip_center.astype(int)

            cv2.circle(annotated, tuple(sc), 5, (255, 255, 255), -1)
            cv2.circle(annotated, tuple(hc), 5, (255, 255, 255), -1)
            cv2.line(annotated, tuple(sc), tuple(hc), (255, 255, 255), 2)

        # Mostrar razones
        if len(reasons) > 0 and score >= 3:
            reason_text = ", ".join(reasons)

            cv2.putText(
                annotated,
                reason_text,
                (x1, min(annotated.shape[0] - 20, y2 + 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

    return annotated


# ==========================================================
# MAIN
# ==========================================================

def main():
    global SHOW_ALL_KEYPOINTS

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video o camara: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps is None or fps <= 0:
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        fourcc,
        fps,
        (width, height)
    )

    window_name = "YOLO-Pose | Zonas de estantes | Actividad sospechosa"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    paused = False
    frame_idx = 0
    last_processed = None
    start_time = time.time()

    print("===================================================")
    print(" SISTEMA INICIADO")
    print("===================================================")
    print("Click izquierdo  = agregar punto")
    print("Click derecho    = cerrar zona")
    print("P                = pausar / continuar")
    print("S                = guardar zonas")
    print("L                = cargar zonas")
    print("C                = limpiar zona actual")
    print("R                = borrar todas las zonas")
    print("K                = mostrar/ocultar keypoints")
    print("Q                = salir")
    print("===================================================")

    # Intentar cargar zonas si ya existen
    if os.path.exists(ZONES_PATH):
        load_zones(ZONES_PATH)

    while True:
        if not paused:
            ret, frame = cap.read()

            if not ret:
                print("[INFO] Fin del video.")
                break

            frame_idx += 1

            processed = process_frame(
                frame,
                model,
                frame_idx,
                fps
            )

            last_processed = processed.copy()

            # Guardamos video procesado sin los overlays dinamicos de texto final
            writer.write(processed)

        if last_processed is None:
            continue

        display = last_processed.copy()

        # Dibujar zonas encima
        draw_zones(display)

        # Textos generales
        status = "PAUSADO" if paused else "REPRODUCIENDO"

        cv2.putText(
            display,
            f"Estado: {status}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Zonas: {len(editor_state['zones'])} | Keypoints: {'ON' if editor_state['show_keypoints'] else 'OFF'}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            "P: pausa | S: guardar | L: cargar | C: limpiar actual | R: borrar zonas | K: keypoints | Q: salir",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("[INFO] Saliendo...")
            break

        elif key == ord("p"):
            paused = not paused
            print(f"[INFO] Pausa: {paused}")

        elif key == ord("s"):
            save_zones(ZONES_PATH)

        elif key == ord("l"):
            load_zones(ZONES_PATH)

        elif key == ord("c"):
            editor_state["current"] = []
            print("[OK] Zona actual limpiada.")

        elif key == ord("r"):
            editor_state["zones"] = []
            editor_state["current"] = []
            print("[OK] Todas las zonas fueron borradas.")

        elif key == ord("k"):
            editor_state["show_keypoints"] = not editor_state["show_keypoints"]
            print(f"[INFO] Mostrar keypoints: {editor_state['show_keypoints']}")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - start_time

    print("===================================================")
    print(" PROCESO TERMINADO")
    print("===================================================")
    print(f"Video guardado en: {OUTPUT_PATH}")
    print(f"Zonas guardadas en: {ZONES_PATH}")
    print(f"Frames procesados: {frame_idx}")
    print(f"Tiempo total: {elapsed:.2f} segundos")


if __name__ == "__main__":
    main()