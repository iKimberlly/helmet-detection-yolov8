from ultralytics import YOLO
import os
from pathlib import Path

# ================================
# CONFIGURAÇÕES
# ================================
RUNS_DIR = "runs/detect"
CONFIDENCE = 0.5

# Caminhos de teste (altere aqui)
IMAGE_TEST = "teste/CAPACETE3.jpg"   # ex: imagem externa
VIDEO_TEST = "teste/Dia do Técnico de Segurança do Trabalho 2023 - Dimensional QSMS-I (720p, h264, youtube).mp4"    # ex: vídeo externo

# ================================
# LOCALIZAR ÚLTIMO TREINO
# ================================
train_dirs = sorted(
    [d for d in os.listdir(RUNS_DIR) if d.startswith("train")]
)

if not train_dirs:
    raise FileNotFoundError("Nenhum treino encontrado em runs/detect")

last_train = train_dirs[-1]
weights_path = os.path.join(RUNS_DIR, last_train, "weights", "best.pt")

print(f"✔ Usando pesos: {weights_path}")

# ================================
# CARREGAR MODELO
# ================================
model = YOLO(weights_path)

# ================================
# TESTE COM IMAGEM
# ================================
if Path(IMAGE_TEST).exists():
    print("🖼 Rodando inferência na imagem...")
    model.predict(
        source=IMAGE_TEST,
        conf=CONFIDENCE,
        save=True
    )
else:
    print(f"⚠ Imagem não encontrada: {IMAGE_TEST}")

# ================================
# TESTE COM VÍDEO
# ================================
if Path(VIDEO_TEST).exists():
    print("🎥 Rodando inferência no vídeo...")
    model.predict(
        source=VIDEO_TEST,
        conf=CONFIDENCE,
        save=True,
        stream=False
    )
else:
    print(f"⚠ Vídeo não encontrado: {VIDEO_TEST}")

print("✅ Inferência finalizada")
