from ultralytics import YOLO

model = YOLO("runs/detect/shoplifting_model/weights/best.pt")

results = model.predict(
    source="imagen1.jpeg",
    conf=0.25,
    save=True,
    show=True
)