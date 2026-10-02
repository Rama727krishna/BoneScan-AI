import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model


# ============================================================
# BoneScan AI - Single Image Prediction
# ============================================================

print("=" * 60)
print("BoneScan AI - PREDICTION")
print("=" * 60)


# ============================================================
# Project Paths
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "efficientnet_b1.keras"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "models",
    "prediction_results"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


IMAGE_SIZE = (240, 240)


# ============================================================
# Get Image Path
# ============================================================

if len(sys.argv) < 2:

    print("\nUsage:")
    print(
        "python src\\predict.py "
        "\"path\\to\\xray.jpg\""
    )

    print("\nExample:")
    print(
        "python src\\predict.py "
        "\"dataset\\test\\fracture\\0.png\""
    )

    sys.exit(1)


IMAGE_PATH = sys.argv[1]

# Convert relative path to absolute path
if not os.path.isabs(IMAGE_PATH):

    IMAGE_PATH = os.path.join(
        PROJECT_DIR,
        IMAGE_PATH
    )

IMAGE_PATH = os.path.abspath(
    IMAGE_PATH
)


# ============================================================
# Check Files
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )


if not os.path.exists(IMAGE_PATH):

    raise FileNotFoundError(
        f"\nImage not found:\n{IMAGE_PATH}"
    )


# ============================================================
# Load Model
# ============================================================

print("\nLoading EfficientNetB1 model...")

model = load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# Find EfficientNet Base Model
# ============================================================

base_model = model.get_layer(
    "efficientnetb1"
)

print(
    "\nEfficientNet base model:",
    base_model.name
)


# ============================================================
# Find Last Convolutional Layer
# ============================================================

last_conv_layer = None
last_conv_index = None

for i in range(
    len(base_model.layers) - 1,
    -1,
    -1
):

    layer = base_model.layers[i]

    if isinstance(
        layer,
        tf.keras.layers.Conv2D
    ):

        last_conv_layer = layer
        last_conv_index = i

        break


if last_conv_layer is None:

    raise ValueError(
        "Could not find last convolutional layer."
    )


print(
    "Last convolutional layer:",
    last_conv_layer.name
)

print(
    "Layer index:",
    last_conv_index
)

print(
    "Last convolutional output:",
    last_conv_layer.output.shape
)


# ============================================================
# Feature Model
# ============================================================

feature_model = Model(
    inputs=base_model.input,
    outputs=last_conv_layer.output
)


# ============================================================
# Create Connected Classifier
# ============================================================

print(
    "\nCreating Grad-CAM classifier..."
)

conv_shape = tuple(
    last_conv_layer.output.shape[1:]
)

classifier_input = tf.keras.Input(
    shape=conv_shape,
    name="gradcam_classifier_input"
)

x = classifier_input


# ------------------------------------------------------------
# EfficientNet layers after top_conv
# ------------------------------------------------------------

for layer in base_model.layers[
    last_conv_index + 1:
]:

    if isinstance(
        layer,
        tf.keras.layers.InputLayer
    ):
        continue

    x = layer(x)


# ------------------------------------------------------------
# Find EfficientNet position in outer model
# ------------------------------------------------------------

base_index = None

for i, layer in enumerate(
    model.layers
):

    if layer.name == base_model.name:

        base_index = i
        break


if base_index is None:

    raise ValueError(
        "Could not find EfficientNet base model "
        "inside outer model."
    )


# ------------------------------------------------------------
# Outer model layers after EfficientNet
# ------------------------------------------------------------

for layer in model.layers[
    base_index + 1:
]:

    if isinstance(
        layer,
        tf.keras.layers.InputLayer
    ):
        continue

    x = layer(x)


classifier_model = Model(
    classifier_input,
    x,
    name="gradcam_classifier"
)

print(
    "Grad-CAM classifier created."
)


# ============================================================
# Load Image
# ============================================================

print(
    "\nLoading image:"
)

print(
    IMAGE_PATH
)

original_image = cv2.imread(
    IMAGE_PATH
)

if original_image is None:

    raise ValueError(
        "OpenCV could not read the image."
    )


# Keep original for final visualization
display_image = original_image.copy()


# Convert BGR → RGB
image_rgb = cv2.cvtColor(
    original_image,
    cv2.COLOR_BGR2RGB
)


# Resize
image_resized = cv2.resize(
    image_rgb,
    IMAGE_SIZE
)


# IMPORTANT:
# Do NOT divide by 255.
# EfficientNetB1 contains its own preprocessing.
image_array = np.expand_dims(
    image_resized.astype(
        np.float32
    ),
    axis=0
)


print(
    "Image prepared successfully."
)


# ============================================================
# Prediction
# ============================================================

print(
    "\nRunning prediction..."
)

prediction = model.predict(
    image_array,
    verbose=0
)[0][0]


# Dataset class mapping:
#
# 0 = fracture
# 1 = normal
#
# Model sigmoid output = probability of normal


if prediction >= 0.5:

    predicted_class = "normal"

    confidence = prediction

else:

    predicted_class = "fracture"

    confidence = 1.0 - prediction


confidence_percent = (
    confidence * 100
)


print("\n" + "-" * 60)
print("PREDICTION RESULT")
print("-" * 60)

print(
    "Prediction  :",
    predicted_class.upper()
)

print(
    "Confidence  :",
    f"{confidence_percent:.2f}%"
)

print("-" * 60)


# ============================================================
# Grad-CAM
# ============================================================

print(
    "\nGenerating Grad-CAM..."
)


with tf.GradientTape() as tape:

    # Last convolutional activation
    conv_output = feature_model(
        image_array,
        training=False
    )

    # Tell TensorFlow to track this tensor
    tape.watch(
        conv_output
    )

    # Send activation through classifier
    classifier_prediction = classifier_model(
        conv_output,
        training=False
    )

    # Select predicted class score
    if predicted_class == "normal":

        class_score = (
            classifier_prediction[:, 0]
        )

    else:

        # Fracture score
        class_score = (
            1.0 -
            classifier_prediction[:, 0]
        )


# ============================================================
# Gradients
# ============================================================

grads = tape.gradient(
    class_score,
    conv_output
)


if grads is None:

    raise ValueError(
        "Gradients could not be calculated."
    )


print(
    "Gradients calculated successfully."
)


# ============================================================
# Grad-CAM Calculation
# ============================================================

# Average gradients over height and width
pooled_grads = tf.reduce_mean(
    grads,
    axis=(1, 2)
)


conv_output = conv_output[0]

pooled_grads = pooled_grads[0]


# Weight feature maps
weighted_features = (
    conv_output *
    pooled_grads
)


# Combine feature maps
heatmap = tf.reduce_sum(
    weighted_features,
    axis=-1
)


# Remove negative values
heatmap = tf.maximum(
    heatmap,
    0
)


# Normalize
max_value = tf.reduce_max(
    heatmap
).numpy()


if max_value > 0:

    heatmap = (
        heatmap /
        max_value
    )


heatmap = heatmap.numpy()


# ============================================================
# Resize Heatmap
# ============================================================

heatmap = cv2.resize(
    heatmap,
    (
        display_image.shape[1],
        display_image.shape[0]
    )
)


# Convert to 8-bit
heatmap_uint8 = np.uint8(
    255 * heatmap
)


# Apply color map
heatmap_color = cv2.applyColorMap(
    heatmap_uint8,
    cv2.COLORMAP_JET
)


# ============================================================
# Create Overlay
# ============================================================

overlay = cv2.addWeighted(
    display_image,
    0.60,
    heatmap_color,
    0.40,
    0
)


# ============================================================
# Add Prediction Text
# ============================================================

text = (
    f"{predicted_class.upper()} "
    f"({confidence_percent:.2f}%)"
)


cv2.putText(
    overlay,
    text,
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)


# ============================================================
# Save Result
# ============================================================

image_name = os.path.splitext(
    os.path.basename(IMAGE_PATH)
)[0]


output_filename = (
    f"{image_name}_prediction_gradcam.jpg"
)


output_path = os.path.join(
    OUTPUT_DIR,
    output_filename
)


success = cv2.imwrite(
    output_path,
    overlay
)


if not success:

    raise ValueError(
        "Failed to save result image."
    )


# ============================================================
# Final Output
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    "\nImage      :",
    os.path.basename(IMAGE_PATH)
)

print(
    "Prediction :",
    predicted_class.upper()
)

print(
    "Confidence :",
    f"{confidence_percent:.2f}%"
)

print(
    "Grad-CAM   :",
    "Generated"
)

print(
    "\nResult saved to:"
)

print(
    output_path
)

print("\n" + "=" * 60)