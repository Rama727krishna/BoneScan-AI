import os
from PIL import Image

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

TARGET_FILES = {
    "IMG0004134.jpg",
    "IMG0004143.jpg",
    "IMG0004148.jpg",
    "IMG0004149.jpg",
    "IMG0004308.jpg",
    "IMG0004347.jpg",
}

print("=" * 70)
print("SEARCHING FOR ORIGINAL/CLEAN COPIES")
print("=" * 70)

found = {}

for root, dirs, files in os.walk(PROJECT_PATH):

    # Don't search inside the destination dataset
    # because we already know those copies are problematic.
    if os.path.abspath(root).startswith(
        os.path.abspath(os.path.join(PROJECT_PATH, "dataset"))
    ):
        continue

    for filename in files:

        if filename not in TARGET_FILES:
            continue

        file_path = os.path.join(root, filename)

        print("\nFOUND:")
        print(file_path)

        # Test whether this copy can actually be decoded
        try:

            with Image.open(file_path) as img:
                img.load()
                img.convert("RGB")

            print("STATUS: CLEAN")

            found[filename] = file_path

        except Exception as e:

            print("STATUS: PROBLEMATIC")
            print("ERROR:", e)


print("\n")
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

for filename in sorted(TARGET_FILES):

    if filename in found:
        print(f"[CLEAN] {filename}")
        print(f"       {found[filename]}")
    else:
        print(f"[NOT FOUND] {filename}")

print("=" * 70)