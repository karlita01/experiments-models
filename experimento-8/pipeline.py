"""
=======================================================
  SHOPLIFTING DETECTION PIPELINE
  Paso 1: Extrae keypoints del dataset con YOLOv8 Pose
  Paso 2: Entrena clasificador XGBoost
=======================================================
Uso:
  python pipeline.py --step all       # Ejecuta todo
  python pipeline.py --step extract   # Solo extrae keypoints
  python pipeline.py --step train     # Solo entrena el modelo
"""

import os
import cv2
import warnings
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)

warnings.filterwarnings("ignore")

# ================================================================
#  CONFIGURACIÓN  (ajusta estas rutas si es necesario)
# ================================================================
DATASET_PATH       = "dataset"           # Carpeta raíz del dataset
YOLO_MODEL_PATH    = "yolov8n-pose.pt"  # Modelo YOLOv8 pose (se descarga si no existe)
CONFIDENCE_THRESH  = 0.50               # Confianza mínima para tomar una detección
NUM_KEYPOINTS      = 17                 # YOLOv8 detecta 17 keypoints por persona
OUTPUT_CSV         = "keypoints_dataset.csv"
XGBOOST_MODEL_PATH = "models/xgboost_model.json"

# Mapeo de clases (tal como vienen en los labels del dataset de Sparsh Goyal)
# 0 = Normal  |  1 = Suspicious
CLASS_MAP = {0: "Normal", 1: "Suspicious"}


# ================================================================
#  UTILIDADES
# ================================================================

def read_yolo_class(label_path: Path) -> int | None:
    """Lee la clase del primer objeto en un archivo de label YOLO."""
    try:
        with open(label_path) as f:
            first_line = f.readline().strip()
        if first_line:
            return int(first_line.split()[0])
    except Exception:
        pass
    return None


def keypoints_to_row(kps: list, class_id: int, img_name: str, split: str) -> dict:
    """Convierte una lista de keypoints en un diccionario (fila del CSV)."""
    row = {"image": img_name, "split": split, "label": class_id}
    for j in range(NUM_KEYPOINTS):
        if j < len(kps):
            row[f"x{j}"] = round(kps[j][0], 6)
            row[f"y{j}"] = round(kps[j][1], 6)
        else:
            row[f"x{j}"] = 0.0
            row[f"y{j}"] = 0.0
    return row


def get_feature_columns() -> list:
    """Retorna los nombres de las columnas de features (x0..x16, y0..y16)."""
    return [f"x{j}" for j in range(NUM_KEYPOINTS)] + \
           [f"y{j}" for j in range(NUM_KEYPOINTS)]


# ================================================================
#  PASO 1: EXTRAER KEYPOINTS
# ================================================================

def extract_keypoints() -> pd.DataFrame:
    print("\n" + "=" * 55)
    print("  PASO 1: Extrayendo keypoints del dataset")
    print("=" * 55)

    model = YOLO(YOLO_MODEL_PATH)
    all_rows = []
    splits = ["train", "test", "valid"]

    for split in splits:
        images_dir = Path(DATASET_PATH) / split / "images"
        labels_dir = Path(DATASET_PATH) / split / "labels"

        if not images_dir.exists():
            print(f"  ⚠ No se encontró: {images_dir} — se omite.")
            continue

        image_files = sorted(
            list(images_dir.glob("*.jpg")) +
            list(images_dir.glob("*.png")) +
            list(images_dir.glob("*.jpeg"))
        )

        normal_count = suspicious_count = skipped = 0

        for img_path in tqdm(image_files, desc=f"  {split:>5}"):
            label_path = labels_dir / (img_path.stem + ".txt")

            if not label_path.exists():
                skipped += 1
                continue

            class_id = read_yolo_class(label_path)
            if class_id is None:
                skipped += 1
                continue

            frame = cv2.imread(str(img_path))
            if frame is None:
                skipped += 1
                continue

            results = model(frame, verbose=False)

            for r in results:
                if r.keypoints is None or len(r.keypoints.xyn) == 0:
                    continue

                kps_list  = r.keypoints.xyn.tolist()
                confs     = r.boxes.conf.tolist() if r.boxes is not None else []

                # Elegir la detección con mayor confianza
                if confs:
                    best_idx = int(np.argmax(confs))
                    if confs[best_idx] < CONFIDENCE_THRESH:
                        continue
                else:
                    best_idx = 0

                row = keypoints_to_row(kps_list[best_idx], class_id,
                                       img_path.name, split)
                all_rows.append(row)

                if class_id == 0:
                    normal_count += 1
                else:
                    suspicious_count += 1
                break  # Una persona por imagen

        print(f"         Normal: {normal_count}  |  "
              f"Suspicious: {suspicious_count}  |  Omitidas: {skipped}")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    total = len(df)
    print(f"\n  ✅ Total muestras extraídas : {total}")
    print(f"     Normal     : {len(df[df['label'] == 0])}")
    print(f"     Suspicious : {len(df[df['label'] == 1])}")
    print(f"     Guardado en: {OUTPUT_CSV}\n")
    return df


# ================================================================
#  PASO 2: ENTRENAR XGBOOST
# ================================================================

def train_classifier(df: pd.DataFrame = None) -> xgb.XGBClassifier:
    print("\n" + "=" * 55)
    print("  PASO 2: Entrenando clasificador XGBoost")
    print("=" * 55)

    if df is None:
        df = pd.read_csv(OUTPUT_CSV)

    df = df.dropna().reset_index(drop=True)
    feat_cols = get_feature_columns()

    # ── Dividir por split del dataset ──────────────────────────
    if "split" in df.columns and "train" in df["split"].values:
        train_df = df[df["split"] == "train"]
        test_df  = df[df["split"].isin(["test", "valid"])]
    else:
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["label"]
        )

    X_train, y_train = train_df[feat_cols], train_df["label"]
    X_test,  y_test  = test_df[feat_cols],  test_df["label"]

    print(f"\n  Muestras de entrenamiento : {len(X_train)}")
    print(f"  Muestras de evaluación    : {len(X_test)}")

    # ── Modelo ─────────────────────────────────────────────────
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        eta=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        objective="binary:logistic",
        tree_method="hist",
        random_state=42,
        use_label_encoder=False,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── Métricas ───────────────────────────────────────────────
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    acc         = accuracy_score(y_test, y_pred)
    auc         = roc_auc_score(y_test, y_pred_prob)

    print(f"\n  ✅ Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"     ROC-AUC  : {auc:.4f}")
    print("\n  Reporte de clasificación:")
    print(classification_report(y_test, y_pred,
                                target_names=["Normal", "Suspicious"]))

    # ── Matriz de confusión ────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Suspicious"],
                yticklabels=["Normal", "Suspicious"])
    plt.title("Confusion Matrix")
    plt.ylabel("Real"); plt.xlabel("Predicho")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png", dpi=150)
    plt.close()

    # ── Guardar modelo ─────────────────────────────────────────
    model.save_model(XGBOOST_MODEL_PATH)
    print(f"\n  ✅ Modelo guardado en : {XGBOOST_MODEL_PATH}")
    print(f"     Matriz de confusión: models/confusion_matrix.png\n")
    return model


# ================================================================
#  MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Shoplifting Detection Pipeline")
    parser.add_argument(
        "--step",
        choices=["extract", "train", "all"],
        default="all",
        help="Paso a ejecutar",
    )
    args = parser.parse_args()

    df = None

    if args.step in ("extract", "all"):
        df = extract_keypoints()

    if args.step in ("train", "all"):
        if df is None:
            df = pd.read_csv(OUTPUT_CSV)
        train_classifier(df)

    print("  🏁 Pipeline completado.\n")


if __name__ == "__main__":
    main()
