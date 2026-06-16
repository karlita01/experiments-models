import os
import random
import shutil

ORIGEN = "dataset_personas"
DESTINO = "dataset_personas_balanceado"

CANTIDAD_NORMAL = 1800
CANTIDAD_ROBO = 1778

os.makedirs(os.path.join(DESTINO, "normal"), exist_ok=True)
os.makedirs(os.path.join(DESTINO, "robo"), exist_ok=True)

def copiar_muestra(clase, cantidad):
    origen_clase = os.path.join(ORIGEN, clase)
    destino_clase = os.path.join(DESTINO, clase)

    imagenes = [
        img for img in os.listdir(origen_clase)
        if img.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(imagenes)
    seleccionadas = imagenes[:cantidad]

    for img in seleccionadas:
        shutil.copy(
            os.path.join(origen_clase, img),
            os.path.join(destino_clase, img)
        )

    print(f"{clase}: {len(seleccionadas)} imágenes copiadas")

copiar_muestra("normal", CANTIDAD_NORMAL)
copiar_muestra("robo", CANTIDAD_ROBO)

print("Dataset balanceado creado correctamente.")