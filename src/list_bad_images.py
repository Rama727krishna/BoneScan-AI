import os
from PIL import Image

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATASET_PATH = os.path.join(PROJECT_PATH, "dataset")

folders = [
    os.path.join(DATASET_PATH, "train"),
    os.path.join(DATASET_PATH, "validation"),
    os.path.join(DATASET_PATH, "test"),
]

extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

bad_images = []

print("=" * 70)
print("SCANNING FOR PROBLEMATIC IMAGES")
print("=" * 70)

for folder in folders:

    for root, dirs, files in os.walk(folder):

        for filename in files:

            if not filename.lower().endswith(extensions):
                continue

            file_path = os.path.join(root, filename)

            try:
                with Image.open(file_path) as img:
                    img.load()
                    img.convert("RGB")

            except Exception as e:

                bad_images.append(file_path)

                print("\nBAD IMAGE:")
                print(file_path)
                print("ERROR:", e)

print("\n")
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print("Total problematic images:", len(bad_images))

print("\nProblematic files:")

for i, path in enumerate(bad_images, 1):
    print(f"{i}. {path}")

print("\n" + "=" * 70)