from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/shoplifting_model_local-2/weights/best.pt")

results = model.predict(
    source="caminando2.jpeg",
    conf=0.25,
    save=False
)

annotated_img = results[0].plot()

cv2.imshow("Deteccion YOLO", annotated_img)

cv2.waitKey(0)

cv2.destroyAllWindows()