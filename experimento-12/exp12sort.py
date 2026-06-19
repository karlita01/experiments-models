# ============================================================
# GENERAR DATASET CON RECORTES DE PERSONAS
# YOLO + ByteTrack + Crop ampliado por persona
# ============================================================

import os
import cv2
import random
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# ============================================================
# CONFIGURACION GENERAL
# ============================================================

INPUT_DATASETS = [
    {
        "name": "dataset1",
        "root": "dataset1",
        "classes": {
            "normal": "normal",
            "shoplifting": "shoplifting"
        }
    },
    {
        "name": "dataset2",
        "root": "dataset2",
        "classes": {
            "Normal": "normal",
            "Shoplifting": "shoplifting"
        }
    }
]

OUTPUT_ROOT = "dataset_personas"

YOLO_MODEL = "yolov8n.pt"

IMG_SIZE = 224

WINDOW_SECONDS = 5        # duración aproximada de cada clip generado
STRIDE_RATIO = 0.5        # 0.5 = ventanas con 50% de solapamiento
EXPAND_RATIO = 1.6        # aumenta el recorte de la persona para incluir contexto

MIN_TRACK_SECONDS = 2     # mínimo de segundos para guardar un track

SAVE_MODE = "main_track"  
# "main_track" = guarda solo la persona principal del video
# "all_tracks" = guarda todas las personas detectadas

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

random.seed(RANDOM_SEED)

# ============================================================
# CARGAR YOLO
# ============================================================

detector = YOLO(YOLO_MODEL)

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def clean_name(name):
    """
    Limpia nombres para evitar problemas al guardar archivos.
    """
    name = os.path.splitext(name)[0]
    name = name.replace(" ", "_")
    name = name.replace("(", "")
    name = name.replace(")", "")
    name = name.replace("[", "")
    name = name.replace("]", "")
    return name


def expand_box(x1, y1, x2, y2, frame_w, frame_h, ratio=1.6):
    """
    Amplía la caja de la persona para incluir un poco de contexto.
    Esto es importante para shoplifting porque el modelo puede necesitar ver
    manos, productos, mochila, estante, bolsa, etc.
    """
    box_w = x2 - x1
    box_h = y2 - y1

    cx = x1 + box_w / 2
    cy = y1 + box_h / 2

    new_w = box_w * ratio
    new_h = box_h * ratio

    nx1 = int(max(0, cx - new_w / 2))
    ny1 = int(max(0, cy - new_h / 2))
    nx2 = int(min(frame_w, cx + new_w / 2))
    ny2 = int(min(frame_h, cy + new_h / 2))

    return nx1, ny1, nx2, ny2


def resize_with_padding(img, size=224):
    """
    Redimensiona manteniendo proporción y rellena con negro.
    Así no deformamos demasiado a la persona.
    """
    if img is None or img.size == 0:
        return None

    h, w = img.shape[:2]

    if h == 0 or w == 0:
        return None

    scale = size / max(h, w)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h))

    canvas = np.zeros((size, size, 3), dtype=np.uint8)

    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


def save_video_clip(frames, output_path, fps):
    """
    Guarda una lista de frames como video mp4.
    """
    if len(frames) == 0:
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (IMG_SIZE, IMG_SIZE))

    if not out.isOpened():
        print("No se pudo crear el video:", output_path)
        return False

    for frame in frames:
        out.write(frame)

    out.release()
    return True


def collect_videos():
    """
    Busca todos los videos de dataset1 y dataset2.
    Devuelve una lista con path, label y dataset.
    """
    records = []

    for dataset in INPUT_DATASETS:
        dataset_name = dataset["name"]
        dataset_root = dataset["root"]
        class_map = dataset["classes"]

        print("=" * 70)
        print(f"Revisando {dataset_name}: {dataset_root}")
        print("=" * 70)

        if not os.path.exists(dataset_root):
            print(f"No existe la carpeta: {dataset_root}")
            continue

        for input_class_name, output_label in class_map.items():
            input_dir = os.path.join(dataset_root, input_class_name)

            print("Buscando en:", input_dir)

            if not os.path.exists(input_dir):
                print("No existe:", input_dir)
                continue

            videos = [
                f for f in os.listdir(input_dir)
                if f.lower().endswith(VIDEO_EXTENSIONS)
            ]

            print(f"Videos encontrados en {input_dir}: {len(videos)}")

            for video in videos:
                video_path = os.path.join(input_dir, video)

                records.append({
                    "path": video_path,
                    "label": output_label,
                    "dataset": dataset_name,
                    "original_class": input_class_name,
                    "video_name": video
                })

    return records


def split_records_by_video(records):
    """
    Divide por video original en train/val/test.
    Importante: todos los clips generados de un video quedan en el mismo split.
    """
    splits = {
        "train": [],
        "val": [],
        "test": []
    }

    labels = sorted(list(set([r["label"] for r in records])))

    for label in labels:
        label_records = [r for r in records if r["label"] == label]
        random.shuffle(label_records)

        n = len(label_records)

        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        train_records = label_records[:n_train]
        val_records = label_records[n_train:n_train + n_val]
        test_records = label_records[n_train + n_val:]

        splits["train"].extend(train_records)
        splits["val"].extend(val_records)
        splits["test"].extend(test_records)

    return splits


def save_track_as_clips(track_frames, output_dir, base_name, track_id, fps):
    """
    Guarda los frames de un track como uno o varios clips.
    """
    if len(track_frames) == 0:
        return 0

    clip_len = int(fps * WINDOW_SECONDS)
    stride = int(clip_len * STRIDE_RATIO)

    if clip_len <= 0:
        clip_len = 32

    if stride <= 0:
        stride = max(1, clip_len // 2)

    min_frames = int(fps * MIN_TRACK_SECONDS)

    if min_frames <= 0:
        min_frames = 16

    if len(track_frames) < min_frames:
        return 0

    clips_saved = 0

    # Si el track es menor a la ventana, igual guardamos un clip
    # siempre que tenga suficientes frames.
    if len(track_frames) < clip_len:
        output_path = os.path.join(
            output_dir,
            f"{base_name}_id{track_id}_clip000.mp4"
        )

        ok = save_video_clip(track_frames, output_path, fps)

        if ok:
            clips_saved += 1

        return clips_saved

    # Si el track es largo, generamos varios clips con solapamiento.
    clip_index = 0

    for start in range(0, len(track_frames) - clip_len + 1, stride):
        clip = track_frames[start:start + clip_len]

        output_path = os.path.join(
            output_dir,
            f"{base_name}_id{track_id}_clip{clip_index:03d}.mp4"
        )

        ok = save_video_clip(clip, output_path, fps)

        if ok:
            clips_saved += 1

        clip_index += 1

    return clips_saved


def process_video(record, split_name):
    """
    Procesa un video:
    1. Detecta personas con YOLO.
    2. Asigna ID con ByteTrack.
    3. Recorta a cada persona con contexto.
    4. Guarda clips por persona.
    """
    video_path = record["path"]
    label = record["label"]
    dataset_name = record["dataset"]
    video_name = record["video_name"]

    output_dir = os.path.join(OUTPUT_ROOT, split_name, label)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("No se pudo abrir:", video_path)
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps is None or fps <= 1:
        fps = 10

    fps = int(round(fps))

    track_frames = defaultdict(list)

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_h, frame_w = frame.shape[:2]

        results = detector.track(
            frame,
            persist=True,
            classes=[0],              # clase 0 = persona
            tracker="bytetrack.yaml",
            conf=0.25,
            verbose=False
        )

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = box.astype(int)

                x1, y1, x2, y2 = expand_box(
                    x1, y1, x2, y2,
                    frame_w, frame_h,
                    ratio=EXPAND_RATIO
                )

                crop = frame[y1:y2, x1:x2]
                crop = resize_with_padding(crop, IMG_SIZE)

                if crop is not None:
                    track_frames[track_id].append(crop)

        frame_count += 1

    cap.release()

    # Reiniciar predictor para que ByteTrack no arrastre IDs entre videos
    detector.predictor = None

    if len(track_frames) == 0:
        print(f"Sin personas detectadas: {video_path}")
        return 0

    safe_video_name = clean_name(video_name)
    base_name = f"{dataset_name}_{label}_{safe_video_name}"

    total_clips = 0

    if SAVE_MODE == "main_track":
        # Guarda solo la persona con más frames
        main_track_id = max(track_frames, key=lambda tid: len(track_frames[tid]))
        frames_persona = track_frames[main_track_id]

        total_clips += save_track_as_clips(
            frames_persona,
            output_dir,
            base_name,
            main_track_id,
            fps
        )

        print(
            f"[{split_name}] {label} | {video_name} | "
            f"Track principal ID {main_track_id} | "
            f"Frames: {len(frames_persona)} | Clips: {total_clips}"
        )

    elif SAVE_MODE == "all_tracks":
        # Guarda todas las personas detectadas
        for track_id, frames_persona in track_frames.items():
            clips = save_track_as_clips(
                frames_persona,
                output_dir,
                base_name,
                track_id,
                fps
            )

            total_clips += clips

        print(
            f"[{split_name}] {label} | {video_name} | "
            f"Tracks: {len(track_frames)} | Clips: {total_clips}"
        )

    return total_clips


def count_output_clips():
    """
    Cuenta cuántos clips se generaron.
    """
    print("\n" + "=" * 70)
    print("RESUMEN DEL DATASET GENERADO")
    print("=" * 70)

    total = 0

    for split in ["train", "val", "test"]:
        for label in ["normal", "shoplifting"]:
            folder = os.path.join(OUTPUT_ROOT, split, label)

            if not os.path.exists(folder):
                count = 0
            else:
                count = len([
                    f for f in os.listdir(folder)
                    if f.lower().endswith(VIDEO_EXTENSIONS)
                ])

            total += count

            print(f"{folder}: {count} clips")

    print("-" * 70)
    print(f"Total clips generados: {total}")
    print("=" * 70)


# ============================================================
# EJECUCION PRINCIPAL
# ============================================================

print("Carpeta actual:")
print(os.getcwd())

print("\nCarpetas disponibles:")
print(os.listdir())

records = collect_videos()

print("\n" + "=" * 70)
print(f"Total de videos encontrados: {len(records)}")
print("=" * 70)

normal_count = len([r for r in records if r["label"] == "normal"])
shoplifting_count = len([r for r in records if r["label"] == "shoplifting"])

print(f"Videos normal: {normal_count}")
print(f"Videos shoplifting: {shoplifting_count}")

if len(records) == 0:
    print("\nNo se encontraron videos. Revisa si dataset1 y dataset2 están en la carpeta actual.")
else:
    splits = split_records_by_video(records)

    print("\nDistribución por split:")
    for split_name, split_records in splits.items():
        n_normal = len([r for r in split_records if r["label"] == "normal"])
        n_shop = len([r for r in split_records if r["label"] == "shoplifting"])

        print(f"{split_name}: {len(split_records)} videos | normal: {n_normal} | shoplifting: {n_shop}")

    print("\nIniciando generación de clips por persona...\n")

    total_generated = 0

    for split_name, split_records in splits.items():
        print("\n" + "#" * 70)
        print(f"PROCESANDO SPLIT: {split_name}")
        print("#" * 70)

        for record in split_records:
            clips = process_video(record, split_name)
            total_generated += clips

    count_output_clips()

    print("\nProceso terminado.")
    print(f"Dataset generado en: {OUTPUT_ROOT}")