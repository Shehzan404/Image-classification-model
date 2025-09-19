import os
import shutil
import random

def split_data():
    base_dir = r"D:\matar_paneer_classifier\data"
    output_dir = r"D:\matar_paneer_classifier\dataset"

    for folder in ["train", "val", "test"]:
        for cls in ["matar_paneer", "not_matar_paneer"]:
            os.makedirs(os.path.join(output_dir, folder, cls), exist_ok=True)

    for cls in ["matar_paneer", "not_matar_paneer"]:
        src = os.path.join(base_dir, cls)
        all_images = os.listdir(src)
        random.shuffle(all_images)

        total = len(all_images)
        train_split = int(0.7 * total)
        val_split = int(0.2 * total)

        train_files = all_images[:train_split]
        val_files = all_images[train_split:train_split+val_split]
        test_files = all_images[train_split+val_split:]

        for f in train_files:
            shutil.copy(os.path.join(src, f), os.path.join(output_dir, "train", cls, f))
        for f in val_files:
            shutil.copy(os.path.join(src, f), os.path.join(output_dir, "val", cls, f))
        for f in test_files:
            shutil.copy(os.path.join(src, f), os.path.join(output_dir, "test", cls, f))

if __name__ == "__main__":
    split_data()

