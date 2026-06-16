from ultralytics import YOLO
import cv2

model = YOLO("runs/classify/experimento2_personas_balanceado/weights/best.pt")

video_path = "robo1.mp4"

cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(frame, verbose=False)

    probs = results[0].probs.data.tolist()

    normal_prob = probs[0]
    robo_prob = probs[1]

    clase = "ROBO" if robo_prob > normal_prob else "NORMAL"
    confianza = max(normal_prob, robo_prob)

    color = (0, 0, 255) if clase == "ROBO" else (0, 255, 0)

    texto = f"{clase} {confianza:.2f}"

    cv2.putText(
        frame,
        texto,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Prediccion en video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()