from ultralytics import YOLO

model = YOLO("runs/detect/deteccion_robo/weights/best.pt")

model.predict(
    source="video2.mp4",
    conf=0.25,
    show=True,
    save=True
)