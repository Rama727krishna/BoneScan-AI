import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

import matplotlib.pyplot as plt


# ============================================================
# BoneScan AI - EfficientNetB1 Evaluation
# ============================================================

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

MODEL_PATH = os.path.join(
    PROJECT_PATH,
    "models",
    "efficientnet_b1.keras"
)

TEST_PATH = os.path.join(
    PROJECT_PATH,
    "dataset",
    "test"
)

IMAGE_SIZE = (240, 240)
BATCH_SIZE = 32


# ============================================================
# 1. LOAD MODEL
# ============================================================

print("=" * 60)
print("LOADING EFFICIENTNETB1 MODEL")
print("=" * 60)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# 2. LOAD TEST DATASET
# ============================================================

print("\n" + "=" * 60)
print("LOADING TEST DATASET")
print("=" * 60)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

class_names = test_dataset.class_names

print("\nClasses:")
print(class_names)


# ============================================================
# 3. PREPROCESSING
# ============================================================
#
# IMPORTANT:
#
# EfficientNetB1 already contains the required input
# preprocessing/rescaling inside the Keras model.
#
# Therefore, DO NOT add:
#
# Rescaling(1.0 / 255)
#
# here.
# ============================================================

test_dataset = test_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# 4. COLLECT TRUE LABELS
# ============================================================

print("\nCollecting test labels...")

y_true = []

for images, labels in test_dataset:

    y_true.extend(
        labels.numpy().flatten()
    )

y_true = np.array(y_true).astype(int)

print("Number of test labels:")
print(len(y_true))


# ============================================================
# 5. GENERATE PREDICTIONS
# ============================================================

print("\nGenerating EfficientNetB1 predictions...")

y_probability = model.predict(
    test_dataset,
    verbose=1
).flatten()

y_pred = (
    y_probability >= 0.5
).astype(int)


# ============================================================
# 6. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_true,
    y_probability
)


# ============================================================
# 7. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("EFFICIENTNETB1 EVALUATION RESULTS")
print("=" * 60)

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")
print(f"ROC-AUC   : {roc_auc:.4f}")


# ============================================================
# 8. CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n")
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


# Create confusion matrix figure
fig, ax = plt.subplots(
    figsize=(7, 6)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    ax=ax
)

ax.set_title(
    "BoneScan AI - EfficientNetB1 Confusion Matrix"
)

plt.tight_layout()

cm_path = os.path.join(
    PROJECT_PATH,
    "models",
    "efficientnet_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("\nConfusion matrix saved:")
print(cm_path)


# ============================================================
# 10. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_true,
    y_probability
)

fig = plt.figure(
    figsize=(7, 6)
)

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "BoneScan AI - EfficientNetB1 ROC Curve"
)

plt.legend()

plt.tight_layout()

roc_path = os.path.join(
    PROJECT_PATH,
    "models",
    "efficientnet_roc_curve.png"
)

plt.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("\nROC curve saved:")
print(roc_path)


# ============================================================
# 11. FINAL VERIFICATION
# ============================================================

print("\n")
print("=" * 60)
print("FINAL VERIFICATION")
print("=" * 60)

print("Expected test images : 506")
print(f"Evaluated test images: {len(y_true)}")

if len(y_true) == 506:

    print("\nSUCCESS!")
    print("All 506 test images were evaluated.")

else:

    print("\nWARNING!")
    print("Not all test images were evaluated.")


# ============================================================
# 12. COMPLETION
# ============================================================

print("\nEfficientNetB1 evaluation completed.")

print("=" * 60)