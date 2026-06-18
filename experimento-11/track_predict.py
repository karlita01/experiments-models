import cv2
import numpy as np
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
from collections import deque
from ultralytics import YOLO
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

VIDEO_PATH = "video.mp4"

MODEL_DIR       = "modelo_guardado"
NUM_FRAMES      = 16       
FRAMES_BUFFER   = 30       
RECLASIFICAR    = 15       
THRESHOLD       = 0.45     
OUTPUT_VIDEO    = "output.mp4"

# Colores (BGR)
COLOR_NORMAL      = (0, 200, 0)      
COLOR_SHOPLIFTING = (0, 0, 220)      
COLOR_ANALIZANDO  = (200, 200, 0)    

class PersonaTracker:
    def __init__(self, persona_id):
        self.id              = persona_id
        self.frames          = deque(maxlen=FRAMES_BUFFER)
        self.estado          = "Analizando..."
        self.prob_shoplifting = 0.0
        self.frames_desde_clasificacion = 0

    def agregar_frame(self, crop):
        """Agrega el recorte de la persona al buffer"""
        frame_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        self.frames.append(frame_resized)
        self.frames_desde_clasificacion += 1

    def listo_para_clasificar(self):
        """Retorna True si tiene suficientes frames y es momento de clasificar"""
        return (len(self.frames) >= NUM_FRAMES and
                self.frames_desde_clasificacion >= RECLASIFICAR)

    def clasificar(self, model, processor, device):
        """Clasifica el comportamiento con VideoMAE"""
        frames_list = list(self.frames)

        # Tomar NUM_FRAMES uniformemente del buffer
        indices = np.linspace(0, len(frames_list) - 1, NUM_FRAMES).astype(int)
        frames_seleccionados = [frames_list[i] for i in indices]

        inputs = processor(frames_seleccionados, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            probs   = F.softmax(outputs.logits, dim=-1)

        self.prob_shoplifting = probs[0][1].item()
        self.estado = "SHOPLIFTING" if self.prob_shoplifting > THRESHOLD else "Normal"
        self.frames_desde_clasificacion = 0

def dibujar_persona(frame, bbox, persona):
    x1, y1, x2, y2 = map(int, bbox)

    if persona.estado == "SHOPLIFTING":
        color = COLOR_SHOPLIFTING
    elif persona.estado == "Normal":
        color = COLOR_NORMAL
    else:
        color = COLOR_ANALIZANDO

    # Bounding box
    grosor = 3 if persona.estado == "SHOPLIFTING" else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, grosor)

    # Etiqueta
    if persona.estado == "SHOPLIFTING":
        label = f"ID{persona.id} ⚠ SOSPECHOSO {persona.prob_shoplifting*100:.0f}%"
    elif persona.estado == "Normal":
        label = f"ID{persona.id} Normal"
    else:
        label = f"ID{persona.id} Analizando..."

    # Fondo de la etiqueta
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Alerta grande si es shoplifting
    if persona.estado == "SHOPLIFTING":
        cv2.putText(frame, "! ACTIVIDAD SOSPECHOSA !",
                    (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, COLOR_SHOPLIFTING, 2)

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # Cargar YOLOv8
    print("Cargando YOLOv8...")
    yolo = YOLO("yolov8n.pt")

    # Cargar VideoMAE
    print(f"Cargando VideoMAE desde '{MODEL_DIR}'...")
    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    model     = VideoMAEForVideoClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    print("Modelos cargados.\n")

    # Abrir video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {VIDEO_PATH}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {VIDEO_PATH}")
    print(f"  {width}x{height} @ {fps:.1f}fps  |  {total} frames")
    print(f"\nProcesando... (esto puede tardar unos minutos)")
    print("-" * 50)

    # Video de salida
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    # Diccionario de personas trackeadas
    personas = {}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  Frame {frame_idx}/{total} ({frame_idx/total*100:.1f}%)")

        # ── YOLO TRACKING ──
        results = yolo.track(frame, persist=True, classes=[0], verbose=False)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids   = results[0].boxes.id.cpu().numpy().astype(int)

            for bbox, pid in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, bbox)

                # Asegura que el crop no se salga del frame
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)

                if x2 - x1 < 20 or y2 - y1 < 20:
                    continue

                # Crear tracker si es persona nueva
                if pid not in personas:
                    personas[pid] = PersonaTracker(pid)

                # Recortar persona del frame
                crop = frame[y1:y2, x1:x2]
                personas[pid].agregar_frame(crop)

                # Clasificar si tiene suficientes frames
                if personas[pid].listo_para_clasificar():
                    personas[pid].clasificar(model, processor, device)
                    print(f"  Persona {pid}: {personas[pid].estado} "
                          f"({personas[pid].prob_shoplifting*100:.1f}%)")

                # Dibujar en el frame
                dibujar_persona(frame, bbox, personas[pid])

        # Escribir frame al video de salida
        # Mostrar en pantalla
        cv2.imshow("Shoplifting Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Escribir frame al video de salida
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("-" * 50)
    print(f"\nVideo procesado guardado en: {OUTPUT_VIDEO}")

    # Resumen final
    print("\nRESUMEN:")
    for pid, persona in personas.items():
        print(f"  Persona {pid}: {persona.estado} "
              f"(shoplifting: {persona.prob_shoplifting*100:.1f}%)")
