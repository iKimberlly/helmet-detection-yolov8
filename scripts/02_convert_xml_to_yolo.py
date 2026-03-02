import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

from pathlib import Path

ROOT = Path(r"C:\Users\LJKK\.cache\kagglehub\datasets\andrewmvd\hard-hat-detection\versions\1")


images_dir = ROOT / "images"
ann_dir = ROOT / "annotations"

out = Path("ppe_yolo")
(out / "images/train").mkdir(parents=True, exist_ok=True)
(out / "labels/train").mkdir(parents=True, exist_ok=True)

created = 0

for xml in ann_dir.glob("*.xml"):
    tree = ET.parse(xml)
    r = tree.getroot()

    img_name = r.find("filename").text
    img_path = images_dir / img_name
    if not img_path.exists():
        continue

    w = int(r.find("size/width").text)
    h = int(r.find("size/height").text)

    lines = []

    for obj in r.findall("object"):
        cls = obj.find("name").text

        if cls == "helmet":
            cls_id = 0          # helmet
        elif cls == "head":
            cls_id = 1          # no_helmet
        else:
            continue

        b = obj.find("bndbox")
        xmin, ymin = int(b.find("xmin").text), int(b.find("ymin").text)
        xmax, ymax = int(b.find("xmax").text), int(b.find("ymax").text)

        x = ((xmin + xmax) / 2) / w
        y = ((ymin + ymax) / 2) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h

        lines.append(f"{cls_id} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")

    if lines:
        shutil.copy(img_path, out / "images/train" / img_name)
        with open(out / "labels/train" / f"{img_path.stem}.txt", "w") as f:
            f.write("\n".join(lines))
        created += 1

print("Imagens anotadas criadas:", created)
