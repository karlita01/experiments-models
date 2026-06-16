from ultralytics import YOLO
import os

DATASET = "dataset_personas_balanceado"

def contar_imagenes():
    for clase in ["normal", "robo"]:
        carpeta = os.path.join(DATASET, clase)
        cantidad = len([
            img for img in os.listdir(carpeta)
            if img.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        print(f"{clase}: {cantidad} imágenes")

def entrenar():
    model = YOLO("yolov8n-cls.pt")

    model.train(
        data=DATASET,
        epochs=50,
        imgsz=224,
        batch=16,
        dropout=0.2,
        patience=10,
        name="experimento2_personas_balanceado"
    )

if __name__ == "__main__":
    contar_imagenes()
    entrenar()