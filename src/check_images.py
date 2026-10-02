import os
from PIL import Image

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATASET_PATH = os.path.join(PROJECT_PATH, "dataset")

folders = [
    os.path.join(DATASET_PATH, "train"),
    os.path.join(DATASET_PATH, "validation"),
    os.path.join(DATASET_PATH, "test")
]

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

bad_files = []
total_files = 0


print("=" * 60)
print("BONE IMAGE VALIDATION")
print("=" * 60)

for folder in folders:

    print(f"\nChecking:")
    print(folder)

    if not os.path.exists(folder):
        print("Folder not found!")
        continue

    for root, dirs, files in os.walk(folder):

        for file in files:

            if not file.lower().endswith(valid_extensions):
                continue

            file_path = os.path.join(root, file)
            total_files += 1

            try:
                with Image.open(file_path) as img:
                    img.verify()

            except Exception as e:

                bad_files.append(
                    (file_path, str(e))
                )

                print("\nBAD IMAGE:")
                print(file_path)
                print("ERROR:")
                print(e)


print("\n" + "=" * 60)
print("VALIDATION RESULT")
print("=" * 60)

print(f"\nTotal images checked: {total_files}")
print(f"Corrupted/problematic images: {len(bad_files)}")


if len(bad_files) == 0:

    print("\nSUCCESS!")
    print("No corrupted images were found.")

else:

    print("\nProblematic files:")

    for file_path, error in bad_files:

        print("\n" + file_path)
        print(error)


print("\n" + "=" * 60)