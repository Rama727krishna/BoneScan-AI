import os
import tensorflow as tf

# ============================================================
# BoneScan AI - EfficientNetB1 Transfer Learning
# ============================================================

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATASET_PATH = os.path.join(PROJECT_PATH, "dataset")

TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VAL_PATH = os.path.join(DATASET_PATH, "validation")

MODEL_PATH = os.path.join(
    PROJECT_PATH,
    "models",
    "efficientnet_b1.keras"
)

# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (240, 240)
BATCH_SIZE = 32
EPOCHS = 15

LEARNING_RATE = 0.0001

# ============================================================
# LOAD DATASETS
# ============================================================

print("=" * 60)
print("LOADING DATASETS")
print("=" * 60)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=42
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

# Save class names before further dataset operations
class_names = train_dataset.class_names

print("\nClasses:")
print(class_names)

print("\nTraining images: 9246")
print("Validation images: 825")

# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.10),
], name="data_augmentation")


# ============================================================
# PERFORMANCE
# ============================================================

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# EFFICIENTNETB1
# ============================================================

print("\n" + "=" * 60)
print("BUILDING EFFICIENTNETB1")
print("=" * 60)

base_model = tf.keras.applications.EfficientNetB1(
    include_top=False,
    weights="imagenet",
    input_shape=(240, 240, 3)
)

# Freeze pretrained layers initially
base_model.trainable = False


# ============================================================
# BUILD MODEL
# ============================================================

inputs = tf.keras.Input(
    shape=(240, 240, 3)
)

x = data_augmentation(inputs)

# EfficientNet has its own preprocessing
x = base_model(
    x,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Dropout(0.4)(x)

x = tf.keras.layers.Dense(
    128,
    activation="relu"
)(x)

x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = tf.keras.Model(
    inputs,
    outputs,
    name="BoneScan_EfficientNetB1"
)


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-7,
    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 60)
print("STARTING EFFICIENTNETB1 TRAINING")
print("=" * 60)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        checkpoint,
        reduce_lr
    ]
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(MODEL_PATH)

print("\n" + "=" * 60)
print("EFFICIENTNETB1 TRAINING COMPLETED")
print("=" * 60)

print("\nModel saved at:")
print(MODEL_PATH)

print("\nClasses:")
print(class_names)

print("\nNext step:")
print("Evaluate EfficientNetB1 on all 506 test images.")