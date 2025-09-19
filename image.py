import os

folder_path ="D:\matar_paneer_classifier\data\matar_paneer"



image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]

images = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in image_extensions]

print("Total images:", len(images))
