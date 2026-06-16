from ultralytics import YOLO
import cv2
import os

detector = YOLO("yolov8n.pt")

INPUT_DIR = "frames_dataset"
OUTPUT_DIR = "dataset_personas"

CLASES = ["normal", "robo"]

CONF_PERSONA = 0.40
MARGEN = 60

for clase in CLASES:
    input_folder = os.path.join(INPUT_DIR, clase)
    output_folder = os.path.join(OUTPUT_DIR, clase)

    os.makedirs(output_folder, exist_ok=True)

    total = 0

    for img_name in os.listdir(input_folder):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        h, w, _ = img.shape

        results = detector(img, verbose=False)

        persona_num = 0

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # clase 0 en COCO = persona
                if cls_id == 0 and conf >= CONF_PERSONA:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    x1 = max(0, x1 - MARGEN)
                    y1 = max(0, y1 - MARGEN)
                    x2 = min(w, x2 + MARGEN)
                    y2 = min(h, y2 + MARGEN)

                    crop = img[y1:y2, x1:x2]

                    if crop.size == 0:
                        continue

                    output_name = f"{os.path.splitext(img_name)[0]}_persona_{persona_num}.jpg"
                    output_path = os.path.join(output_folder, output_name)

                    cv2.imwrite(output_path, crop)

                    total += 1
                    persona_num += 1

    print(f"{clase}: {total} recortes generados")

print("Listo. Dataset de personas creado.")