import numpy as np
import torch
import torch.nn.functional as F
import decord
import warnings
warnings.filterwarnings("ignore")
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

VIDEO_PATH = "video.mp4"

MODEL_DIR      = "modelo_guardado"
NUM_FRAMES     = 16 
WINDOW_SECONDS = 5  
THRESHOLD      = 0.5

def analizar_video(video_path, model, processor, device):

    vr         = decord.VideoReader(video_path)
    fps        = vr.get_avg_fps()
    total_frames = len(vr)
    duracion   = total_frames / fps

    frames_por_ventana = int(fps * WINDOW_SECONDS)

    print(f"\nVideo: {video_path}")
    print(f"  Duracion:   {duracion:.1f} segundos")
    print(f"  FPS:        {fps:.1f}")
    print(f"  Frames:     {total_frames}")
    print(f"  Ventana:    {WINDOW_SECONDS} segundos ({frames_por_ventana} frames por ventana)")
    print("-" * 60)

    resultados = []
    ventana    = 0
    inicio     = 0

    while inicio < total_frames:
        fin = min(inicio + frames_por_ventana, total_frames)

        # Si la ventana es muy corta al final, ignorarla
        if (fin - inicio) < frames_por_ventana // 2:
            break

        # Tomar NUM_FRAMES uniformemente dentro de la ventana
        indices = np.linspace(inicio, fin - 1, NUM_FRAMES).astype(int)
        frames  = vr.get_batch(indices).asnumpy()
        inputs  = processor(list(frames), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            probs   = F.softmax(outputs.logits, dim=-1)

        prob_normal      = probs[0][0].item() * 100
        prob_shoplifting = probs[0][1].item() * 100
        prediccion       = "SHOPLIFTING" if probs[0][1].item() > THRESHOLD else "NORMAL"

        tiempo_inicio = inicio / fps
        tiempo_fin    = fin / fps

        # Guardar resultado
        resultados.append({
            "ventana"         : ventana + 1,
            "tiempo_inicio"   : tiempo_inicio,
            "tiempo_fin"      : tiempo_fin,
            "prediccion"      : prediccion,
            "prob_normal"     : prob_normal,
            "prob_shoplifting": prob_shoplifting,
        })

        # Mostrar en consola
        alerta = "⚠️  SOSPECHOSO" if prediccion == "SHOPLIFTING" else "✓  Normal"
        print(f"  Ventana {ventana+1:02d} | "
              f"{tiempo_inicio:5.1f}s - {tiempo_fin:5.1f}s | "
              f"Normal: {prob_normal:5.1f}%  "
              f"Shoplifting: {prob_shoplifting:5.1f}%  | "
              f"{alerta}")

        ventana += 1
        inicio  += frames_por_ventana

    # ── RESUMEN FINAL ──
    print("-" * 60)
    detecciones = [r for r in resultados if r["prediccion"] == "SHOPLIFTING"]

    if detecciones:
        print(f"\n⚠️  ACTIVIDAD SOSPECHOSA DETECTADA en {len(detecciones)} ventana(s):")
        for d in detecciones:
            print(f"   → {d['tiempo_inicio']:.1f}s - {d['tiempo_fin']:.1f}s  "
                  f"(confianza: {d['prob_shoplifting']:.1f}%)")
    else:
        print("\n✓  Sin actividad sospechosa detectada.")

    return resultados

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    print(f"Cargando modelo desde '{MODEL_DIR}'...")
    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    model     = VideoMAEForVideoClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    print("Modelo cargado.")

    analizar_video(VIDEO_PATH, model, processor, device)