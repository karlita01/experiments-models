from ultralytics import YOLO
import os

DATASET_PATH = "frames_dataset"

def verificar_dataset():
    normal = os.path.join(DATASET_PATH, "normal")
    robo = os.path.join(DATASET_PATH, "robo")

    if not os.path.exists(normal):
        raise FileNotFoundError("No existe la carpeta frames_dataset/normal")

    if not os.path.exists(robo):
        raise FileNotFoundError("No existe la carpeta frames_dataset/robo")

    print("Dataset encontrado correctamente.")
    print("Imágenes normal:", len(os.listdir(normal)))
    print("Imágenes robo:", len(os.listdir(robo)))

def entrenar():
    model = YOLO("yolov8n-cls.pt")

    model.train(
        data=DATASET_PATH,
        epochs=30,
        imgsz=224,
        batch=16,
        name="modelo_robo_normal"
    )

if __name__ == "__main__":
    verificar_dataset()
    entrenar()