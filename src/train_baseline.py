import os
import tensorflow as tf
import numpy as np

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ============================================================
# STEP 5: BASELINE CNN TRAINING
# ============================================================

print("=" * 60)
print("STEP 5: BASELINE CNN TRAINING")
print("=" * 60)


# ------------------------------------------------------------
# 1. PROJECT PATH
# ------------------------------------------------------------

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATASET_PATH = os.path.join(PROJECT_PATH, "dataset")

TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VAL_PATH = os.path.join(DATASET_PATH, "validation")
TEST_PATH = os.path.join(DATASET_PATH, "test")

MODEL_PATH = os.path.join(
    PROJECT_PATH,
    "models",
    "baseline_cnn.keras"
)


# Create models folder if it doesn't exist
os.makedirs(
    os.path.join(PROJECT_PATH, "models"),
    exist_ok=True
)


# ------------------------------------------------------------
# 2. SETTINGS
# ------------------------------------------------------------

IMAGE_SIZE = (240, 240)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42


print("\nProject path:")
print(PROJECT_PATH)

print("\nDataset path:")
print(DATASET_PATH)


# ------------------------------------------------------------
# 3. LOAD TRAINING DATASET
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("LOADING TRAINING DATA")
print("=" * 60)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=SEED
)


# ------------------------------------------------------------
# 4. LOAD VALIDATION DATASET
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("LOADING VALIDATION DATA")
print("=" * 60)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)


# ------------------------------------------------------------
# 5. LOAD TEST DATASET
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 6. SAVE CLASS NAMES
# ------------------------------------------------------------

class_names = train_dataset.class_names

print("\nClasses:")
print(class_names)


# ------------------------------------------------------------
# 7. NORMALIZE PIXEL VALUES
# ------------------------------------------------------------

normalization_layer = layers.Rescaling(1.0 / 255)


train_dataset = train_dataset.map(
    lambda x, y: (normalization_layer(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)


validation_dataset = validation_dataset.map(
    lambda x, y: (normalization_layer(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)


test_dataset = test_dataset.map(
    lambda x, y: (normalization_layer(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)


# ------------------------------------------------------------
# 8. SKIP INVALID IMAGES
# ------------------------------------------------------------

# TensorFlow previously reported an invalid JPEG.
# This prevents one unreadable image from stopping training.

train_dataset = train_dataset.apply(
    tf.data.experimental.ignore_errors()
)

validation_dataset = validation_dataset.apply(
    tf.data.experimental.ignore_errors()
)

test_dataset = test_dataset.apply(
    tf.data.experimental.ignore_errors()
)


# ------------------------------------------------------------
# 9. PREFETCH DATA
# ------------------------------------------------------------

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    tf.data.AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ------------------------------------------------------------
# 10. BUILD BASELINE CNN
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("BUILDING BASELINE CNN")
print("=" * 60)


model = models.Sequential([

    # Input
    layers.Input(
        shape=(240, 240, 3)
    ),

    # CNN Block 1
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # CNN Block 2
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # CNN Block 3
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Regularization
    layers.Dropout(0.3),

    # Convert feature maps to vector
    layers.Flatten(),

    # Fully connected layer
    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.5),

    # Binary classification
    layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ------------------------------------------------------------
# 11. DISPLAY MODEL
# ------------------------------------------------------------

print("\nModel architecture:\n")

model.summary()


# ------------------------------------------------------------
# 12. COMPILE MODEL
# ------------------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ------------------------------------------------------------
# 13. CALLBACKS
# ------------------------------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)


checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)


# ------------------------------------------------------------
# 14. TRAIN MODEL
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STARTING CNN TRAINING")
print("=" * 60)


history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        checkpoint
    ]
)


# ------------------------------------------------------------
# 15. FINAL TEST EVALUATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TEST DATA EVALUATION")
print("=" * 60)


test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)


print("\nTest Loss:")
print(test_loss)


print("\nTest Accuracy:")
print(
    f"{test_accuracy * 100:.2f}%"
)


# ------------------------------------------------------------
# 16. SAVE FINAL MODEL
# ------------------------------------------------------------

model.save(MODEL_PATH)


print("\n" + "=" * 60)
print("BASELINE CNN TRAINING COMPLETED")
print("=" * 60)


print("\nModel saved at:")
print(MODEL_PATH)


print("\nClasses:")
print(class_names)


print("\nTest Accuracy:")
print(
    f"{test_accuracy * 100:.2f}%"
)


print("\nReady for Step 6: Baseline CNN evaluation.")


print("=" * 60)