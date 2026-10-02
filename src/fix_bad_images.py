import os
from PIL import Image

# ============================================================
# BoneScan AI - Find and Fix TensorFlow-incompatible JPEGs
# ============================================================

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATASET_PATH = os.path.join(PROJECT_PATH, "dataset")

DATASET_FOLDERS = [
    os.path.join(DATASET_PATH, "train"),
    os.path.join(DATASET_PATH, "validation"),
    os.path.join(DATASET_PATH, "test"),
]

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif")


def check_and_fix_image(file_path):
    """
    Opens an image, loads all pixel data, and rewrites it
    as a clean JPEG if necessary.
    """

    try:
        with Image.open(file_path) as img:

            # Force PIL to read the complete image
            img.load()

            # Convert to RGB
            img = img.convert("RGB")

            # Save to temporary file
            temp_path = file_path + ".fixed.jpg"

            img.save(
                temp_path,
                format="JPEG",
                quality=95,
                optimize=True
            )

        # Replace original file with repaired JPEG
        os.replace(temp_path, file_path)

        return True

    except Exception as e:

        # Remove temporary file if it exists
        temp_path = file_path + ".fixed.jpg"

        if os.path.exists(temp_path):
            os.remove(temp_path)

        print(f"FAILED: {file_path}")
        print(f"ERROR : {e}")

        return False


def process_dataset(folder):
    print("\n" + "=" * 60)
    print(f"CHECKING: {folder}")
    print("=" * 60)

    total = 0
    fixed = 0
    failed = 0

    for root, dirs, files in os.walk(folder):

        for filename in files:

            if not filename.lower().endswith(IMAGE_EXTENSIONS):
                continue

            total += 1

            file_path = os.path.join(root, filename)

            try:

                with Image.open(file_path) as img:

                    # Force complete decoding
                    img.load()

                    # Check image
                    img.convert("RGB")

            except Exception:

                print("\nPROBLEM IMAGE:")
                print(file_path)

                if check_and_fix_image(file_path):
                    print("FIXED!")
                    fixed += 1
                else:
                    failed += 1

    print("\nRESULT")
    print("-" * 40)
    print(f"Total images : {total}")
    print(f"Fixed images : {fixed}")
    print(f"Failed       : {failed}")

    return total, fixed, failed


# ============================================================
# MAIN
# ============================================================

print("=" * 60)
print("BoneScan AI - IMAGE REPAIR")
print("=" * 60)

grand_total = 0
grand_fixed = 0
grand_failed = 0

for folder in DATASET_FOLDERS:

    total, fixed, failed = process_dataset(folder)

    grand_total += total
    grand_fixed += fixed
    grand_failed += failed


print("\n")
print("=" * 60)
print("FINAL RESULT")
print("=" * 60)

print(f"Total images checked : {grand_total}")
print(f"Images repaired      : {grand_fixed}")
print(f"Images failed        : {grand_failed}")

if grand_failed == 0:
    print("\nSUCCESS!")
    print("All images were successfully processed.")
else:
    print("\nWARNING!")
    print("Some images could not be repaired.")

print("=" * 60)