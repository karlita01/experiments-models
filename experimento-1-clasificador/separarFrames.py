import cv2
import os

# Carpetas de entrada
DATA_DIR = "."
CLASES = ["robo", "normal"]

# Carpeta donde se guardarán los frames
OUTPUT_DIR = "frames_dataset"

# Guardar 1 frame cada N frames
FRAME_INTERVAL = 20

def extraer_frames():
    for clase in CLASES:
        input_folder = os.path.join(DATA_DIR, clase)
        output_folder = os.path.join(OUTPUT_DIR, clase)

        os.makedirs(output_folder, exist_ok=True)

        videos = os.listdir(input_folder)

        for video_name in videos:
            video_path = os.path.join(input_folder, video_name)

            if not video_name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                continue

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"No se pudo abrir: {video_path}")
                continue

            frame_count = 0
            saved_count = 0

            video_base = os.path.splitext(video_name)[0]

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                if frame_count % FRAME_INTERVAL == 0:
                    frame_filename = f"{video_base}_frame_{saved_count}.jpg"
                    frame_path = os.path.join(output_folder, frame_filename)

                    cv2.imwrite(frame_path, frame)
                    saved_count += 1

                frame_count += 1

            cap.release()
            print(f"{video_name} → {saved_count} frames guardados en {clase}")

    print("Extracción finalizada.")

if __name__ == "__main__":
    extraer_frames()