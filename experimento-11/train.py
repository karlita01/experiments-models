import os
import glob
import copy
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import decord
import warnings
warnings.filterwarnings("ignore")

CONFIG = {
    "model_name"   : "MCG-NJU/videomae-base",
    "num_frames"   : 16,
    "batch_size"   : 4,
    "epochs"       : 20,
    "learning_rate": 5e-5,
    "patience"     : 5,
    "test_size"    : 0.30,
    "threshold"    : 0.5,
    "seed"         : 42,
    "output_dir"   : "modelo_guardado",
}

class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=16, model_name="MCG-NJU/videomae-base"):
        self.video_paths = video_paths
        self.labels      = labels
        self.num_frames  = num_frames
        self.processor   = VideoMAEImageProcessor.from_pretrained(model_name)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label      = self.labels[idx]
        try:
            vr           = decord.VideoReader(video_path)
            total_frames = len(vr)
            indices      = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            frames       = vr.get_batch(indices).asnumpy()
            inputs       = self.processor(list(frames), return_tensors="pt")
            return {
                "pixel_values": inputs["pixel_values"].squeeze(0),
                "labels"      : torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            print(f"  [ERROR] No se pudo cargar {video_path}: {e}")
            return {
                "pixel_values": torch.zeros(self.num_frames, 3, 224, 224),
                "labels"      : torch.tensor(label, dtype=torch.long)
            }

if __name__ == "__main__":

    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── CARGA DE VIDEOS ──
    print("\nCargando rutas de videos...")
    normal_paths = []
    normal_paths += glob.glob("dataset1/normal/*.mp4")
    normal_paths += glob.glob("dataset2/normal/*.mp4")

    shoplifting_paths = []
    shoplifting_paths += glob.glob("dataset1/shoplifting/*.mp4")
    shoplifting_paths += glob.glob("dataset2/shoplifting/*.mp4")

    all_paths  = normal_paths + shoplifting_paths
    all_labels = [0] * len(normal_paths) + [1] * len(shoplifting_paths)

    print(f"  Normal:      {len(normal_paths)} videos")
    print(f"  Shoplifting: {len(shoplifting_paths)} videos")
    print(f"  Total:       {len(all_paths)} videos")

    if len(all_paths) == 0:
        raise ValueError("No se encontraron videos. Verifica que train.py este en EXP12/")

    # Split 70/15/15
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["seed"],
        shuffle=True,
        stratify=all_labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=0.5,
        random_state=CONFIG["seed"],
        shuffle=True
    )

    print(f"\nSplit del dataset:")
    print(f"  Train: {len(train_paths)} videos")
    print(f"  Val:   {len(val_paths)} videos")
    print(f"  Test:  {len(test_paths)} videos")

    # ── DATALOADERS ──
    print("\nCreando datasets...")
    train_dataset = VideoDataset(train_paths, train_labels, CONFIG["num_frames"], CONFIG["model_name"])
    val_dataset   = VideoDataset(val_paths,   val_labels,   CONFIG["num_frames"], CONFIG["model_name"])
    test_dataset  = VideoDataset(test_paths,  test_labels,  CONFIG["num_frames"], CONFIG["model_name"])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0, pin_memory=True)

    # ── MODELO ──
    print("\nCargando modelo VideoMAE preentrenado...")
    id2label = {0: "Normal", 1: "Shoplifting"}
    label2id = {"Normal": 0, "Shoplifting": 1}

    model = VideoMAEForVideoClassification.from_pretrained(
        CONFIG["model_name"],
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    print("Modelo cargado correctamente.")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametros entrenables: {total_params:,}")

    # ── ENTRENAMIENTO ──
    optimizer        = AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    best_accuracy    = 0.0
    best_model_state = None
    patience_counter = 0
    history          = {"train_loss": [], "val_loss": [], "val_acc": []}

    print(f"\n{'='*50}")
    print("ENTRENAMIENTO")
    print(f"{'='*50}")

    for epoch in range(CONFIG["epochs"]):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        print("-" * 30)

        # Train
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc="  Training"):
            pixel_values = batch["pixel_values"].to(device)
            labels       = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item()

        avg_train_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        print(f"  Train Loss: {avg_train_loss:.4f}")

        # Validacion
        model.eval()
        val_loss = 0
        correct  = 0
        total    = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="  Validation"):
                pixel_values = batch["pixel_values"].to(device)
                labels       = batch["labels"].to(device)
                outputs      = model(pixel_values=pixel_values, labels=labels)
                probs        = F.softmax(outputs.logits, dim=-1)
                predictions  = (probs[:, 1] > CONFIG["threshold"]).long()
                correct      += (predictions == labels).sum().item()
                total        += labels.size(0)
                val_loss     += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_accuracy)
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Acc:  {val_accuracy:.2f}%")

        # Early stopping
        if val_accuracy > best_accuracy:
            print(f"  Mejora: {best_accuracy:.2f}% -> {val_accuracy:.2f}% checkmark")
            best_accuracy    = val_accuracy
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            os.makedirs(CONFIG["output_dir"], exist_ok=True)
            model.save_pretrained(CONFIG["output_dir"])
        else:
            patience_counter += 1
            print(f"  Sin mejora. Paciencia: {patience_counter}/{CONFIG['patience']}")
            if patience_counter >= CONFIG["patience"]:
                print("\n  Early stopping activado.")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"\nEntrenamiento finalizado. Mejor accuracy: {best_accuracy:.2f}%")

    # ── EVALUACION EN TEST ──
    print(f"\n{'='*50}")
    print("EVALUACION EN TEST SET")
    print(f"{'='*50}")

    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            pixel_values = batch["pixel_values"].to(device)
            labels       = batch["labels"].to(device)
            outputs      = model(pixel_values=pixel_values)
            probs        = F.softmax(outputs.logits, dim=-1)
            predictions  = (probs[:, 1] > CONFIG["threshold"]).long()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    print(f"\nTest Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")
    print("\nReporte de Clasificacion:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Shoplifting"]))

    # ── GRAFICAS ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Resultados - Shoplifting Detection", fontsize=14)

    axes[0].plot(history["train_loss"], label="Train Loss", color="blue",  marker="o")
    axes[0].plot(history["val_loss"],   label="Val Loss",   color="red",   marker="x")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["val_acc"], label="Val Accuracy", color="green", marker="s")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True)

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Shoplifting"],
                yticklabels=["Normal", "Shoplifting"],
                ax=axes[2])
    axes[2].set_title("Confusion Matrix - Test Set")
    axes[2].set_xlabel("Predicho")
    axes[2].set_ylabel("Real")

    plt.tight_layout()
    plt.savefig("resultados.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nGrafica guardada como resultados.png")
    print(f"Modelo guardado en: {CONFIG['output_dir']}/")
