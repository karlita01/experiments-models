from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    patience=10,
    name="deteccion_robo"
)

// data.yaml
// train: train/images
// val: valid/images
// test: test/images

// nc: 2
// names: ['normal', 'robo']
