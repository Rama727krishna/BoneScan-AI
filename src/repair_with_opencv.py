import os
import cv2
from PIL import Image

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATASET_PATH = os.path.join(PROJECT_PATH, "dataset")

TARGET_FILES = {
    "IMG0004134.jpg",
    "IMG0004143.jpg",
    "IMG0004148.jpg",
    "IMG0004149.jpg",
    "IMG0004308.jpg",
    "IMG0004347.jpg",
}

folders = [
    os.path.join(DATASET_PATH, "train"),
    os.path.join(DATASET_PATH, "validation"),
    os.path.join(DATASET_PATH, "test"),
]

print("=" * 70)
print("OPEN CV JPEG REPAIR")
print("=" * 70)

fixed = 0
failed = 0

for folder in folders:

    for root, dirs, files in os.walk(folder):

        for filename in files:

            if filename not in TARGET_FILES:
                continue

            file_path = os.path.join(root, filename)

            print("\nChecking:")
            print(file_path)

            # Read image using OpenCV
            image = cv2.imread(file_path, cv2.IMREAD_COLOR)

            if image is None:
                print("OpenCV could NOT decode this image.")
                failed += 1
                continue

            print("OpenCV decoded image successfully.")

            # Save repaired image to temporary file
            temp_file = file_path + ".tmp.jpg"

            success = cv2.imwrite(
                temp_file,
                image,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )

            if not success:
                print("Could not save repaired image.")
                failed += 1
                continue

            # Verify the newly created JPEG using PIL
            try:

                with Image.open(temp_file) as test_image:
                    test_image.load()
                    test_image.convert("RGB")

                # Replace original only after successful verification
                os.replace(temp_file, file_path)

                print("REPAIRED SUCCESSFULLY!")
                fixed += 1

            except Exception as e:

                print("Verification failed:")
                print(e)

                if os.path.exists(temp_file):
                    os.remove(temp_file)

                failed += 1


print("\n")
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

print("Images repaired :", fixed)
print("Images failed   :", failed)

print("=" * 70)