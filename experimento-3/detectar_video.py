from ultralytics import YOLO

model = YOLO("runs/detect/shoplifting_model_local-2/weights/best.pt")

model.predict(
    source="robo2.mp4",
    conf=0.25,
    show=True,
    save=False,
    imgsz=416,
    device="cpu"
)