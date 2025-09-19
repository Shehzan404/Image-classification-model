import os

folders = ["data/matar_paneer", "data/not_matar_paneer"]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created: {folder}")
