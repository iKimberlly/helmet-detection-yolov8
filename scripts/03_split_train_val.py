from pathlib import Path
import random
import shutil

base = Path("ppe_yolo")

train_img = base / "images/train"
train_lbl = base / "labels/train"
val_img = base / "images/val"
val_lbl = base / "labels/val"

val_img.mkdir(parents=True, exist_ok=True)
val_lbl.mkdir(parents=True, exist_ok=True)

labels = list(train_lbl.glob("*.txt"))
random.shuffle(labels)

split = int(0.2 * len(labels))
val_labels = labels[:split]

copied = 0
for lbl in val_labels:
    for ext in [".png", ".jpg"]:
        img = train_img / f"{lbl.stem}{ext}"
        if img.exists():
            shutil.copy(img, val_img / img.name)
            shutil.copy(lbl, val_lbl / lbl.name)
            copied += 1
            break

print("Validação criada:", copied)
