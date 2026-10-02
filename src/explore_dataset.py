"""Inspect a Normal versus Fracture image classification dataset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image

SPLITS = ("train", "validation", "test")
CLASSES = ("fracture", "normal")
IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def image_files(folder: Path) -> Iterable[Path]:
    """Yield supported image files below a folder in stable order."""
    if not folder.exists():
        return ()
    return sorted(
        path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def inspect_image(path: Path) -> tuple[tuple[int, int], str | None]:
    """Return an image's dimensions, or its corruption error."""
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.size, None
    except Exception as error:
        return (0, 0), f"{type(error).__name__}: {error}"


def print_report(dataset_dir: Path, samples_per_class: int, show_samples: bool) -> None:
    """Inspect every split and class, then optionally display representative images."""
    counts: Counter[str] = Counter()
    dimensions: dict[str, Counter[tuple[int, int]]] = defaultdict(Counter)
    corrupted: list[tuple[Path, str]] = []
    samples: dict[str, list[Path]] = defaultdict(list)
    unsupported_files: list[Path] = []

    for split in SPLITS:
        for class_name in CLASSES:
            class_dir = dataset_dir / split / class_name
            files = list(image_files(class_dir))
            key = f"{split}/{class_name}"
            counts[key] = len(files)

            for path in files:
                if len(samples[key]) < samples_per_class:
                    samples[key].append(path)
                size, error = inspect_image(path)
                if error:
                    corrupted.append((path, error))
                else:
                    dimensions[key][size] += 1

            if class_dir.exists():
                unsupported_files.extend(
                    path
                    for path in class_dir.rglob("*")
                    if path.is_file() and path.suffix.lower() not in IMAGE_EXTENSIONS
                )

    print(f"Dataset: {dataset_dir.resolve()}")
    print("\nImage counts:")
    for split in SPLITS:
        split_total = sum(counts[f"{split}/{class_name}"] for class_name in CLASSES)
        print(f"  {split}: {split_total}")
        for class_name in CLASSES:
            print(f"    {class_name}: {counts[f'{split}/{class_name}']}")

    print("\nClass distribution:")
    for split in SPLITS:
        total = sum(counts[f"{split}/{class_name}"] for class_name in CLASSES)
        distribution = ", ".join(
            f"{class_name}={counts[f'{split}/{class_name}']}"
            for class_name in CLASSES
        )
        print(f"  {split}: {distribution} (total={total})")

    print("\nImage dimensions:")
    for key in (f"{split}/{class_name}" for split in SPLITS for class_name in CLASSES):
        values = ", ".join(f"{size} x {count}" for size, count in dimensions[key].most_common())
        print(f"  {key}: {values or 'no valid images'}")

    print(f"\nCorrupted files: {len(corrupted)}")
    for path, error in corrupted:
        print(f"  {path}: {error}")

    print(f"Unsupported files: {len(unsupported_files)}")
    for path in unsupported_files:
        print(f"  {path}")

    if show_samples:
        display_samples(samples)


def display_samples(samples: dict[str, list[Path]]) -> None:
    """Display collected sample images in a grid."""
    import matplotlib.pyplot as plt

    selected = [(key, path) for key, paths in samples.items() for path in paths]
    if not selected:
        print("\nNo valid images available for display.")
        return

    columns = min(4, len(selected))
    rows = (len(selected) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, squeeze=False, figsize=(4 * columns, 4 * rows))

    for index, (key, path) in enumerate(selected):
        axis = axes[index // columns][index % columns]
        with Image.open(path) as image:
            axis.imshow(image, cmap="gray")
        axis.set_title(key)
        axis.axis("off")

    for index in range(len(selected), rows * columns):
        axes[index // columns][index % columns].axis("off")

    figure.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset",
        help="Path to the dataset root.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=2,
        help="Number of sample images to collect per split and class.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display collected sample images with Matplotlib.",
    )
    args = parser.parse_args()
    if args.samples_per_class < 0:
        parser.error("--samples-per-class must be zero or greater")
    return args


def main() -> None:
    args = parse_args()
    print_report(args.dataset_dir, args.samples_per_class, args.show)


if __name__ == "__main__":
    main()
