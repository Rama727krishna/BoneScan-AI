from flask import Flask, render_template, request
import os
import uuid
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "efficientnet_b1.keras"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

RESULT_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "results"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading EfficientNetB1 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# BUILD GRAD-CAM GRAPH
# ============================================================

base_model = model.get_layer("efficientnetb1")

last_conv_layer = base_model.get_layer("top_conv")

print("Grad-CAM layer:", last_conv_layer.name)


# Feature model
feature_model = tf.keras.Model(
    inputs=base_model.input,
    outputs=last_conv_layer.output
)


# ============================================================
# BUILD CLASSIFIER FROM TOP_CONV OUTPUT
# ============================================================

classifier_input = tf.keras.Input(
    shape=last_conv_layer.output.shape[1:]
)

x = classifier_input

# Continue through EfficientNet layers after top_conv
start_index = base_model.layers.index(last_conv_layer) + 1

for layer in base_model.layers[start_index:]:
    x = layer(x)


# Continue through outer model layers
base_index = model.layers.index(base_model) + 1

for layer in model.layers[base_index:]:
    x = layer(x)


classifier_model = tf.keras.Model(
    classifier_input,
    x
)

print("Grad-CAM graph created successfully.")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Unable to read image.")

    # Convert BGR -> RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Resize to model input size
    image = cv2.resize(
        image,
        (240, 240)
    )

    # Keep image as float32 for TensorFlow
    image = image.astype(np.float32)

    # EfficientNet already contains its preprocessing.
    # DO NOT divide by 255 here.

    image_batch = np.expand_dims(
        image,
        axis=0
    )

    return image, image_batch


# ============================================================
# PREDICTION
# ============================================================

def predict_image(image_batch):

    prediction = model.predict(
        image_batch,
        verbose=0
    )[0][0]

    # Dataset class order:
    # 0 = fracture
    # 1 = normal

    if prediction >= 0.5:

        label = "NORMAL"
        confidence = prediction * 100

    else:

        label = "FRACTURE"
        confidence = (1 - prediction) * 100

    return label, confidence, prediction


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(image_batch, prediction):

    print("Generating Grad-CAM...")

    with tf.GradientTape() as tape:

        features = feature_model(
            image_batch
        )

        tape.watch(features)

        classifier_output = classifier_model(
            features
        )

        # Sigmoid output:
        # class 1 = NORMAL
        # class 0 = FRACTURE

        if prediction >= 0.5:

            class_score = classifier_output[:, 0]

        else:

            class_score = 1 - classifier_output[:, 0]

    gradients = tape.gradient(
        class_score,
        features
    )

    if gradients is None:

        raise ValueError(
            "Gradients could not be calculated."
        )

    print("Gradients calculated successfully.")

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    features = features[0]

    pooled_gradients = pooled_gradients[0]

    heatmap = tf.reduce_sum(
        features * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = heatmap / (
        max_value + 1e-8
    )

    heatmap = heatmap.numpy()

    return heatmap


# ============================================================
# CREATE GRAD-CAM OVERLAY
# ============================================================

def create_overlay(original_image, heatmap):

    # --------------------------------------------------------
    # IMPORTANT FIX:
    # OpenCV requires compatible image data types.
    # --------------------------------------------------------

    # Ensure original image is uint8
    original_image = np.clip(
        original_image,
        0,
        255
    ).astype(np.uint8)

    # Resize heatmap
    heatmap = cv2.resize(
        heatmap,
        (
            original_image.shape[1],
            original_image.shape[0]
        )
    )

    # Ensure heatmap values are valid
    heatmap = np.clip(
        heatmap,
        0.0,
        1.0
    )

    # Convert heatmap to uint8
    heatmap = np.uint8(
        heatmap * 255
    )

    # Create colored heatmap
    heatmap_color = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    # Convert RGB -> BGR
    original_bgr = cv2.cvtColor(
        original_image,
        cv2.COLOR_RGB2BGR
    )

    # Explicitly make both uint8
    original_bgr = original_bgr.astype(
        np.uint8
    )

    heatmap_color = heatmap_color.astype(
        np.uint8
    )

    # Verify same shape
    if original_bgr.shape != heatmap_color.shape:

        heatmap_color = cv2.resize(
            heatmap_color,
            (
                original_bgr.shape[1],
                original_bgr.shape[0]
            )
        )

    # Combine images
    overlay = cv2.addWeighted(
        original_bgr,
        0.6,
        heatmap_color,
        0.4,
        0
    )

    return overlay


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    if "image" not in request.files:

        return render_template(
            "index.html",
            error="Please select an X-ray image."
        )

    file = request.files["image"]

    if file.filename == "":

        return render_template(
            "index.html",
            error="Please select an image."
        )

    # Generate unique filename
    unique_id = uuid.uuid4().hex

    filename = (
        unique_id
        + "_"
        + file.filename
    )

    upload_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    file.save(upload_path)

    try:

        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        original_image, image_batch = preprocess_image(
            upload_path
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        label, confidence, raw_prediction = predict_image(
            image_batch
        )

        # ----------------------------------------------------
        # GRAD-CAM
        # ----------------------------------------------------

        heatmap = generate_gradcam(
            image_batch,
            raw_prediction
        )

        overlay = create_overlay(
            original_image,
            heatmap
        )

        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        result_filename = (
            unique_id
            + "_gradcam.jpg"
        )

        result_path = os.path.join(
            RESULT_FOLDER,
            result_filename
        )

        success = cv2.imwrite(
            result_path,
            overlay
        )

        if not success:

            raise ValueError(
                "Failed to save Grad-CAM result."
            )

        # ----------------------------------------------------
        # TERMINAL RESULT
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)
        print(
            "Prediction :",
            label
        )
        print(
            "Confidence : "
            f"{confidence:.2f}%"
        )
        print(
            "Grad-CAM   : Generated"
        )
        print("=" * 60)

        # ----------------------------------------------------
        # SEND RESULT TO WEB PAGE
        # ----------------------------------------------------

        return render_template(
            "index.html",

            prediction=label,

            confidence=f"{confidence:.2f}",

            image_file=(
                "uploads/"
                + filename
            ),

            result_file=(
                "results/"
                + result_filename
            )
        )

    except Exception as e:

        print()
        print("=" * 60)
        print("PREDICTION ERROR")
        print("=" * 60)
        print(e)
        print("=" * 60)

        return render_template(
            "index.html",
            error=str(e)
        )


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("BoneScan AI Web Application")
    print("=" * 60)
    print("Starting Flask server...")
    print("Open: http://127.0.0.1:5000")
    print("=" * 60)

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )