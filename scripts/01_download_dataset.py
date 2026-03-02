import kagglehub
from pathlib import Path

path = kagglehub.dataset_download("andrewmvd/hard-hat-detection")

dst = Path("data/raw")
dst.mkdir(parents=True, exist_ok=True)

print("Dataset baixado em:", path)
print("Use esse caminho como ROOT nos próximos scripts.")
