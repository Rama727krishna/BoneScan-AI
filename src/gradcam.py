import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model

# ============================================================
# BoneScan AI - GRAD-CAM
# ============================================================

print("=" * 60)
print("BoneScan AI - GRAD-CAM")
print("=" * 60)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "efficientnet_b1.keras"
)

IMAGE_PATH = os.path.join(
    PROJECT_DIR,
    "dataset",
    "test",
    "fracture",
    "0.png"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "models",
    "gradcam_results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "0_gradcam.jpg"
)

IMAGE_SIZE = (240, 240)


# ============================================================
# Load Model
# ============================================================

print("\nLoading EfficientNetB1 model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# Find EfficientNet Base Model
# ============================================================

base_model = model.get_layer("efficientnetb1")

print("\nEfficientNet base model:")
print(base_model.name)


# ============================================================
# Find Last Convolutional Layer
# ============================================================

last_conv_layer = None
last_conv_index = None

for i in range(len(base_model.layers) - 1, -1, -1):

    layer = base_model.layers[i]

    if isinstance(layer, tf.keras.layers.Conv2D):

        last_conv_layer = layer
        last_conv_index = i
        break


if last_conv_layer is None:
    raise ValueError("Could not find a Conv2D layer.")

print("Last convolutional layer:", last_conv_layer.name)
print("Layer index:", last_conv_index)

print(
    "Last convolutional output shape:",
    last_conv_layer.output.shape
)


# ============================================================
# Create Feature Model
# ============================================================

print("\nCreating feature extraction model...")

feature_model = Model(
    inputs=base_model.input,
    outputs=last_conv_layer.output
)

print("Feature model created successfully.")


# ============================================================
# Create Classifier Model
# ============================================================

print("\nCreating connected classifier graph...")

# IMPORTANT:
# The classifier input MUST have the shape of top_conv OUTPUT,
# not the original image input.

conv_shape = tuple(last_conv_layer.output.shape[1:])

print("Classifier input shape:", conv_shape)

classifier_input = tf.keras.Input(
    shape=conv_shape,
    name="gradcam_classifier_input"
)

x = classifier_input


# ------------------------------------------------------------
# Continue through EfficientNet layers after top_conv
# ------------------------------------------------------------

for layer in base_model.layers[last_conv_index + 1:]:

    # Ignore InputLayer
    if isinstance(layer, tf.keras.layers.InputLayer):
        continue

    x = layer(x)


# ------------------------------------------------------------
# Continue through the outer model head
# ------------------------------------------------------------

base_index = None

for i, layer in enumerate(model.layers):

    if layer.name == base_model.name:
        base_index = i
        break


if base_index is None:
    raise ValueError(
        "Could not find EfficientNet base model in outer model."
    )


print("EfficientNet position in outer model:", base_index)


for layer in model.layers[base_index + 1:]:

    if isinstance(layer, tf.keras.layers.InputLayer):
        continue

    x = layer(x)


classifier_model = Model(
    classifier_input,
    x,
    name="gradcam_classifier"
)

print("Classifier graph created successfully.")


# ============================================================
# Load Image
# ============================================================

print("\nLoading test image...")

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not load image:\n{IMAGE_PATH}"
    )

original_image = image.copy()

image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

image = cv2.resize(
    image,
    IMAGE_SIZE
)

# IMPORTANT:
# EfficientNet model already contains its own preprocessing.
# Therefore DO NOT divide by 255 here.

image_array = np.expand_dims(
    image.astype(np.float32),
    axis=0
)

print("Image loaded successfully.")
print("Image shape:", image_array.shape)


# ============================================================
# Prediction
# ============================================================

print("\nRunning prediction...")

prediction = model.predict(
    image_array,
    verbose=0
)[0][0]

# Dataset class order:
# 0 = fracture
# 1 = normal
#
# Sigmoid output represents probability of class 1 (normal).

if prediction >= 0.5:

    predicted_class = "normal"
    confidence = prediction

else:

    predicted_class = "fracture"
    confidence = 1.0 - prediction


print("Prediction :", predicted_class)
print("Confidence :", f"{confidence * 100:.2f}%")


# ============================================================
# Generate Grad-CAM
# ============================================================

print("\nGenerating Grad-CAM heatmap...")

with tf.GradientTape() as tape:

    # Get activation from last convolutional layer
    conv_output = feature_model(
        image_array,
        training=False
    )

    # Explicitly watch the activation
    tape.watch(conv_output)

    # Pass activation through the remaining classifier
    classifier_prediction = classifier_model(
        conv_output,
        training=False
    )

    # --------------------------------------------------------
    # Select score for predicted class
    # --------------------------------------------------------

    if predicted_class == "normal":

        # Normal = sigmoid output
        class_score = classifier_prediction[:, 0]

    else:

        # Fracture = 1 - sigmoid(normal probability)
        class_score = 1.0 - classifier_prediction[:, 0]


# ============================================================
# Calculate Gradients
# ============================================================

grads = tape.gradient(
    class_score,
    conv_output
)

if grads is None:

    raise ValueError(
        "Gradients could not be calculated."
    )

print("Gradients calculated successfully.")


# ============================================================
# Grad-CAM Calculation
# ============================================================

# Average gradients across spatial dimensions
pooled_grads = tf.reduce_mean(
    grads,
    axis=(1, 2)
)

conv_output = conv_output[0]

pooled_grads = pooled_grads[0]


# Weight each feature map
weighted_features = (
    conv_output * pooled_grads
)


# Sum feature maps
heatmap = tf.reduce_sum(
    weighted_features,
    axis=-1
)


# ReLU
heatmap = tf.maximum(
    heatmap,
    0
)


# Normalize
max_value = tf.reduce_max(
    heatmap
).numpy()

if max_value > 0:

    heatmap = heatmap / max_value


heatmap = heatmap.numpy()


# ============================================================
# Resize Heatmap
# ============================================================

heatmap = cv2.resize(
    heatmap,
    (
        original_image.shape[1],
        original_image.shape[0]
    )
)


# Convert heatmap to 0-255
heatmap_uint8 = np.uint8(
    255 * heatmap
)


# Apply color map
heatmap_color = cv2.applyColorMap(
    heatmap_uint8,
    cv2.COLORMAP_JET
)


# ============================================================
# Overlay
# ============================================================

overlay = cv2.addWeighted(
    original_image,
    0.6,
    heatmap_color,
    0.4,
    0
)


# ============================================================
# Add Prediction Text
# ============================================================

label = (
    f"Prediction: {predicted_class.upper()} "
    f"({confidence * 100:.2f}%)"
)

cv2.putText(
    overlay,
    label,
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)


# ============================================================
# Save Result
# ============================================================

success = cv2.imwrite(
    OUTPUT_PATH,
    overlay
)

if not success:

    raise ValueError(
        "Failed to save Grad-CAM image."
    )


print("\n" + "=" * 60)
print("GRAD-CAM COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nPrediction :", predicted_class)
print("Confidence :", f"{confidence * 100:.2f}%")
print("Output     :", OUTPUT_PATH)

print("\nHeatmap generated successfully.")