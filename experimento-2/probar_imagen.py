from ultralytics import YOLO
import cv2

# Cargar modelo entrenado
model = YOLO("runs/classify/experimento2_personas_balanceado/weights/best.pt")

# Imagen a probar
image_path = "caminando2.jpeg"

# Leer imagen
img = cv2.imread(image_path)

# Predicción
results = model.predict(img, verbose=False)

# Obtener probabilidades
probs = results[0].probs.data.tolist()

normal_prob = probs[0]
robo_prob = probs[1]

# Clase final
clase = "ROBO" if robo_prob > normal_prob else "NORMAL"

confianza = max(normal_prob, robo_prob)

# Color
color = (0, 0, 255) if clase == "ROBO" else (0, 255, 0)

# Texto
texto = f"{clase} {confianza:.2f}"

# Dibujar texto
cv2.putText(
    img,
    texto,
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    color,
    2
)

# Mostrar ventana
cv2.imshow("Prediccion", img)

# Esperar tecla
cv2.waitKey(0)

# Cerrar ventana
cv2.destroyAllWindows()