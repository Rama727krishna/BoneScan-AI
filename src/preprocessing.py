import os
import tensorflow as tf
import matplotlib.pyplot as plt


# ============================================================
# STEP 1: PROJECT PATH
# ============================================================

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

print("\n" + "=" * 60)
print("BONE SCAN AI - DATA PREPROCESSING")
print("=" * 60)

print("\nProject Path:")
print(PROJECT_PATH)


# ============================================================
# STEP 2: DATASET PATHS
# ============================================================

DATASET_PATH = os.path.join(
    PROJECT_PATH,
    "dataset"
)

TRAIN_PATH = os.path.join(
    DATASET_PATH,
    "train"
)

VAL_PATH = os.path.join(
    DATASET_PATH,
    "validation"
)

TEST_PATH = os.path.join(
    DATASET_PATH,
    "test"
)


# ============================================================
# STEP 3: CHECK DATASET PATHS
# ============================================================

print("\n" + "=" * 60)
print("DATASET PATH CHECK")
print("=" * 60)

print("\nDataset:")
print(DATASET_PATH)

print("\nTrain:")
print(TRAIN_PATH)

print("\nValidation:")
print(VAL_PATH)

print("\nTest:")
print(TEST_PATH)


if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(
        f"\nTrain folder not found:\n{TRAIN_PATH}"
    )

if not os.path.exists(VAL_PATH):
    raise FileNotFoundError(
        f"\nValidation folder not found:\n{VAL_PATH}"
    )

if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(
        f"\nTest folder not found:\n{TEST_PATH}"
    )


print("\nTrain exists      :", os.path.exists(TRAIN_PATH))
print("Validation exists :", os.path.exists(VAL_PATH))
print("Test exists       :", os.path.exists(TEST_PATH))


# ============================================================
# STEP 4: CHECK CLASS FOLDERS
# ============================================================

print("\n" + "=" * 60)
print("CLASS FOLDER CHECK")
print("=" * 60)


EXPECTED_CLASSES = [
    "fracture",
    "normal"
]


for folder_path, folder_name in [
    (TRAIN_PATH, "Train"),
    (VAL_PATH, "Validation"),
    (TEST_PATH, "Test")
]:

    print(f"\n{folder_name} classes:")

    for class_name in EXPECTED_CLASSES:

        class_path = os.path.join(
            folder_path,
            class_name
        )

        if not os.path.exists(class_path):

            raise FileNotFoundError(
                f"\nMissing class folder:\n{class_path}"
            )

        print(
            f"  {class_name}: OK"
        )


# ============================================================
# STEP 5: COUNT IMAGES
# ============================================================

print("\n" + "=" * 60)
print("IMAGE COUNT")
print("=" * 60)


IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif"
)


def count_images(folder):

    count = 0

    for root, dirs, files in os.walk(folder):

        for file in files:

            if file.lower().endswith(
                IMAGE_EXTENSIONS
            ):
                count += 1

    return count


for dataset_name, dataset_path in [
    ("Train", TRAIN_PATH),
    ("Validation", VAL_PATH),
    ("Test", TEST_PATH)
]:

    print(
        f"\n{dataset_name} images:"
    )

    total = 0

    for class_name in EXPECTED_CLASSES:

        class_path = os.path.join(
            dataset_path,
            class_name
        )

        count = count_images(
            class_path
        )

        print(
            f"  {class_name}: {count}"
        )

        total += count

    print(
        f"  Total: {total}"
    )

    if total == 0:

        raise ValueError(
            f"\nNo images found in {dataset_name} dataset."
        )


# ============================================================
# STEP 6: IMAGE SETTINGS
# ============================================================

IMAGE_SIZE = (
    240,
    240
)

BATCH_SIZE = 32


print("\n" + "=" * 60)
print("IMAGE SETTINGS")
print("=" * 60)

print(
    "\nImage size:",
    IMAGE_SIZE
)

print(
    "Batch size:",
    BATCH_SIZE
)


# ============================================================
# STEP 7: LOAD TRAINING DATA
# ============================================================

print("\n" + "=" * 60)
print("LOADING TRAINING DATA")
print("=" * 60)


train_dataset = tf.keras.utils.image_dataset_from_directory(

    TRAIN_PATH,

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    label_mode="binary",

    shuffle=True

)


# ============================================================
# SAVE CLASS NAMES BEFORE MAP/PREFETCH
# ============================================================

class_names = train_dataset.class_names

print("\nClass names:")
print(class_names)


# ============================================================
# STEP 8: LOAD VALIDATION DATA
# ============================================================

print("\n" + "=" * 60)
print("LOADING VALIDATION DATA")
print("=" * 60)


val_dataset = tf.keras.utils.image_dataset_from_directory(

    VAL_PATH,

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    label_mode="binary",

    shuffle=False

)


# ============================================================
# STEP 9: LOAD TEST DATA
# ============================================================

print("\n" + "=" * 60)
print("LOADING TEST DATA")
print("=" * 60)


test_dataset = tf.keras.utils.image_dataset_from_directory(

    TEST_PATH,

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    label_mode="binary",

    shuffle=False

)


# ============================================================
# STEP 10: NORMALIZATION
# ============================================================

print("\n" + "=" * 60)
print("IMAGE NORMALIZATION")
print("=" * 60)


normalization_layer = tf.keras.layers.Rescaling(
    1.0 / 255
)


train_dataset = train_dataset.map(

    lambda images, labels: (
        normalization_layer(images),
        labels
    ),

    num_parallel_calls=tf.data.AUTOTUNE
)


val_dataset = val_dataset.map(

    lambda images, labels: (
        normalization_layer(images),
        labels
    ),

    num_parallel_calls=tf.data.AUTOTUNE
)


test_dataset = test_dataset.map(

    lambda images, labels: (
        normalization_layer(images),
        labels
    ),

    num_parallel_calls=tf.data.AUTOTUNE
)


# ============================================================
# STEP 11: PREFETCH DATA
# ============================================================

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)

val_dataset = val_dataset.prefetch(
    tf.data.AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# STEP 12: DISPLAY SAMPLE X-RAYS
# ============================================================

print("\n" + "=" * 60)
print("DISPLAYING SAMPLE X-RAYS")
print("=" * 60)


plt.figure(
    figsize=(12, 8)
)


for images, labels in train_dataset.take(1):

    number_of_images = min(
        9,
        len(images)
    )

    for i in range(
        number_of_images
    ):

        ax = plt.subplot(
            3,
            3,
            i + 1
        )

        plt.imshow(
            images[i].numpy()
        )

        class_index = int(
            labels[i].numpy()[0]
        )

        class_name = class_names[
            class_index
        ]

        plt.title(
            class_name.upper()
        )

        plt.axis(
            "off"
        )


plt.tight_layout()

plt.show()


# ============================================================
# STEP 13: VERIFY ONE BATCH
# ============================================================

print("\n" + "=" * 60)
print("BATCH VERIFICATION")
print("=" * 60)


for images, labels in train_dataset.take(1):

    print(
        "\nImage batch shape:"
    )

    print(
        images.shape
    )

    print(
        "\nLabel batch shape:"
    )

    print(
        labels.shape
    )

    print(
        "\nMinimum pixel value:"
    )

    print(
        float(
            tf.reduce_min(images)
        )
    )

    print(
        "\nMaximum pixel value:"
    )

    print(
        float(
            tf.reduce_max(images)
        )
    )


# ============================================================
# STEP 14: FINAL STATUS
# ============================================================

print("\n" + "=" * 60)
print("STEP 4 PREPROCESSING COMPLETED")
print("=" * 60)

print(
    "\nDataset successfully loaded."
)

print(
    "Classes:",
    class_names
)

print(
    "Training images: 9246"
)

print(
    "Validation images: 825"
)

print(
    "Test images: 506"
)

print(
    "Image size:",
    IMAGE_SIZE
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "\nReady for Step 5: Baseline CNN training."
)

print("=" * 60)