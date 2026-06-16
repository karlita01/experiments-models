from ultralytics import YOLO

model = YOLO("runs/detect/shoplifting_model/weights/best.pt")

results = model.predict(
    source="demo1.mp4",
    conf=0.25,
    show=True,
    save=True
)