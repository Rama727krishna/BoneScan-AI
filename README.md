# 🦴 BoneScan AI — Hierarchical Bone Fracture Detection System

> An AI-powered educational and research prototype for detecting bone fractures from X-ray images using Deep Learning, EfficientNetB1, and Explainable AI with Grad-CAM.

---

## 📌 Overview

**BoneScan AI** is a deep-learning-based image classification project designed to analyze X-ray images and classify them into two categories:

- **Fracture**
- **Normal**

The system uses **EfficientNetB1 with transfer learning** as its primary deep-learning model.

To make the prediction more understandable, the project also implements **Grad-CAM (Gradient-weighted Class Activation Mapping)**. Grad-CAM generates a visual heatmap showing the regions of the X-ray that contributed to the model's prediction.

The trained model is integrated into a **Flask web application**, allowing users to upload an X-ray image and receive:

1. AI prediction
2. Prediction confidence
3. Original X-ray image
4. Grad-CAM visualization

---

# 🎯 Project Objective

The main objective of BoneScan AI is to demonstrate how modern deep-learning techniques can be applied to medical-image classification in an educational and research environment.

The project focuses on:

- Image preprocessing
- Dataset validation
- Deep-learning classification
- Transfer learning
- Model evaluation
- Explainable AI
- Grad-CAM visualization
- Flask deployment
- Web-based image prediction

---

# 🧠 How the System Works

```text
                    X-Ray Image
                         │
                         ▼
                Image Preprocessing
                         │
                         ▼
                  EfficientNetB1
                         │
                         ▼
              Binary Classification
                    /          \
                   /            \
                  ▼              ▼
             FRACTURE         NORMAL
                  │              │
                  └──────┬───────┘
                         │
                         ▼
                      Grad-CAM
                         │
                         ▼
                Visual Explanation
                         │
                         ▼
                  Flask Web App
```

---

# 🚀 Main Features

### 1. X-Ray Classification

The model classifies an uploaded X-ray into:

```text
FRACTURE
NORMAL
```

### 2. Confidence Score

The application displays the confidence associated with the prediction.

Example:

```text
Prediction : FRACTURE
Confidence : 80.56%
```

### 3. Grad-CAM

Grad-CAM provides a visual explanation of the model's prediction.

The resulting heatmap helps visualize the image regions that contributed to the prediction.

### 4. Web Interface

The project includes a modern Flask-based interface supporting:

- X-ray upload
- Drag-and-drop
- Image preview
- Prediction result
- Confidence visualization
- Grad-CAM visualization
- Responsive design

### 5. Dataset Validation

The project includes scripts for identifying and repairing problematic images before training.

---

# 📊 Dataset

This project uses the:

**Bone Fracture Multi-Region X-ray Data**

Dataset source:

https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data

The dataset contains X-ray images categorized into:

```text
fractured
not fractured
```

For this project, the categories were mapped to:

```text
fractured      → fracture
not fractured  → normal
```

The original validation directory was mapped to:

```text
val → validation
```

### Dataset distribution used

| Dataset | Fracture | Normal | Total |
|---|---:|---:|---:|
| Training | 4,606 | 4,640 | 9,246 |
| Validation | 333 | 492 | 825 |
| Testing | 238 | 268 | 506 |
| **Total** | **5,177** | **5,400** | **10,577** |

The dataset itself is **not included in this GitHub repository**.

Users should download the dataset from the original source and arrange it according to the project structure.

---

# 🧹 Data Cleaning

During development, TensorFlow detected several JPEG decoding problems.

Although standard image validation initially reported the images as readable, TensorFlow's JPEG decoder identified corrupted/problematic files.

A dedicated image-repair workflow was therefore implemented.

### Problematic files

The following filenames appeared as problematic:

```text
IMG0004134.jpg
IMG0004143.jpg
IMG0004148.jpg
IMG0004149.jpg
IMG0004308.jpg
IMG0004347.jpg
```

They occurred across the dataset splits.

### Repair process

OpenCV was used to decode and rewrite the affected images.

Final verification:

```text
Total images checked: 10577
Corrupted/problematic images: 0
```

This ensured that the complete dataset could be evaluated without silently skipping images.

---

# 🏗️ Models

Two deep-learning approaches were evaluated.

## Model 1 — Baseline CNN

A custom Convolutional Neural Network was developed as the baseline.

Architecture:

```text
Input: 240 × 240 × 3

Conv2D (32)
     ↓
MaxPooling
     ↓
Conv2D (64)
     ↓
MaxPooling
     ↓
Conv2D (128)
     ↓
MaxPooling
     ↓
Dropout
     ↓
Flatten
     ↓
Dense (128)
     ↓
Dropout
     ↓
Sigmoid Output
```

Training configuration:

```text
Optimizer       : Adam
Learning Rate   : 0.001
Loss            : Binary Crossentropy
Batch Size      : 32
Maximum Epochs  : 15
Image Size      : 240 × 240
```

---

# 🧠 Model 2 — EfficientNetB1

The main model uses **EfficientNetB1 with ImageNet transfer learning**.

Architecture:

```text
Input Image
     ↓
EfficientNetB1
     ↓
Global Average Pooling
     ↓
Batch Normalization
     ↓
Dropout
     ↓
Dense 128
     ↓
Dropout
     ↓
Sigmoid
     ↓
Fracture / Normal
```

Training configuration:

```text
Model           : EfficientNetB1
Input Size      : 240 × 240
Optimizer       : Adam
Learning Rate   : 0.0001
Loss            : Binary Crossentropy
Batch Size      : 32
Maximum Epochs  : 15
```

Data augmentation was also used during training:

```text
Random Horizontal Flip
Random Rotation
Random Zoom
```

---

# 📈 Model Evaluation

Both models were evaluated using the same **506-image test set**.

| Metric | Baseline CNN | EfficientNetB1 |
|---|---:|---:|
| Accuracy | 99.21% | 94.47% |
| Precision | 100.00% | 98.39% |
| Recall | 98.51% | 91.04% |
| F1-Score | 99.25% | 94.57% |
| ROC-AUC | 0.9999 | 0.9889 |
| Test Images | 506 | 506 |

These results describe performance on this particular dataset and test split. They should not be interpreted as evidence of clinical performance or generalization to other hospitals, populations, scanners, or clinical settings.

---

# 📊 Baseline CNN Confusion Matrix

```text
                 Predicted
              Fracture  Normal

Actual
Fracture         238       0
Normal             4     264
```

Total test images:

```text
506
```

---

# 📊 EfficientNetB1 Confusion Matrix

```text
                 Predicted
              Fracture  Normal

Actual
Fracture         234       4
Normal            24     244
```

Total test images:

```text
506
```

---

# 🔍 Explainable AI — Grad-CAM

The project uses **Grad-CAM** to provide visual explanations of predictions.

The process is:

```text
X-Ray
  ↓
EfficientNetB1
  ↓
Last Convolutional Layer
  ↓
Class Prediction
  ↓
Gradients
  ↓
Weighted Feature Maps
  ↓
Heatmap
  ↓
Original X-Ray + Heatmap
```

The current Grad-CAM implementation uses the EfficientNetB1 `top_conv` layer.

Example output:

```text
Prediction : FRACTURE
Confidence : 80.56%
Grad-CAM   : Generated
```

The generated visualization is saved locally by the application.

---

# 🌐 Web Application

The model is integrated into a Flask web application.

### Technology stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow / Keras | Deep learning |
| EfficientNetB1 | Transfer-learning model |
| OpenCV | Image processing |
| NumPy | Numerical operations |
| Scikit-learn | Evaluation metrics |
| Matplotlib | Evaluation plots |
| Flask | Web backend |
| HTML/CSS | Web interface |
| JavaScript | Upload and UI interaction |
| Grad-CAM | Explainable AI |

---

# 🖥️ Web Application Workflow

```text
Open Browser
     ↓
Upload X-Ray
     ↓
Flask receives image
     ↓
OpenCV preprocessing
     ↓
EfficientNetB1 prediction
     ↓
Calculate confidence
     ↓
Generate Grad-CAM
     ↓
Save visualization
     ↓
Display result
```

---

# 📁 Project Structure

```text
BoneScan-AI/
│
├── dataset/
│   └── .gitkeep
│
├── models/
│   ├── baseline_cnn.keras
│   ├── efficientnet_b1.keras
│   ├── baseline_confusion_matrix.png
│   ├── baseline_roc_curve.png
│   ├── efficientnet_confusion_matrix.png
│   └── efficientnet_roc_curve.png
│
├── notebooks/
│
├── src/
│   ├── check_images.py
│   ├── fix_bad_images.py
│   ├── list_bad_images.py
│   ├── repair_with_opencv.py
│   ├── preprocessing.py
│   ├── train_baseline.py
│   ├── evaluate_baseline.py
│   ├── train_efficientnet.py
│   ├── evaluate_efficientnet.py
│   ├── gradcam.py
│   └── predict.py
│
├── static/
│   ├── uploads/
│   └── results/
│
├── templates/
│   └── index.html
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/BoneScan-AI.git
```

Move into the project:

```bash
cd BoneScan-AI
```

---

# 🐍 Create Virtual Environment

Windows:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

---

# 📦 Install Dependencies

```powershell
pip install -r requirements.txt
```

---

# 📂 Dataset Setup

Download the dataset from Kaggle and arrange it as:

```text
dataset/
│
├── train/
│   ├── fracture/
│   └── normal/
│
├── validation/
│   ├── fracture/
│   └── normal/
│
└── test/
    ├── fracture/
    └── normal/
```

The original dataset labels should be mapped as:

```text
fractured      → fracture
not fractured  → normal
```

---

# ▶️ Run the Web Application

From the project root:

```powershell
python app.py
```

The terminal will display:

```text
BoneScan AI Web Application
Starting Flask server...
Open: http://127.0.0.1:5000
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

# 🧪 Command-Line Prediction

The project also supports prediction without the Flask interface.

Example:

```powershell
python src/predict.py "dataset\test\fracture\0.png"
```

Example output:

```text
PREDICTION RESULT

Prediction  : FRACTURE
Confidence  : 80.56%

Grad-CAM   : Generated
```

For a normal image:

```powershell
python src/predict.py "dataset\test\normal\0.png"
```

---

# 📊 Evaluation

To evaluate the baseline CNN:

```powershell
python src/evaluate_baseline.py
```

To evaluate EfficientNetB1:

```powershell
python src/evaluate_efficientnet.py
```

---

# 🔬 Research Notes

This project demonstrates an end-to-end machine-learning workflow:

```text
Dataset
   ↓
Data Validation
   ↓
Data Repair
   ↓
Preprocessing
   ↓
Baseline CNN
   ↓
Evaluation
   ↓
EfficientNetB1
   ↓
Evaluation
   ↓
Grad-CAM
   ↓
Flask Deployment
```

The project also demonstrates why data-quality validation is important. A dataset can appear valid under one image library while still contain images that fail during TensorFlow decoding.

---

# ⚠️ Important Disclaimer

**BoneScan AI is an educational and research prototype.**

It is **not a medical device** and must not be used as a substitute for professional medical diagnosis, treatment, or clinical decision-making.

The reported metrics were obtained from a specific dataset and test split. Real-world medical performance can differ substantially because of differences in imaging equipment, patient populations, image quality, acquisition protocols, and other factors.

Predictions should therefore be treated as experimental outputs rather than medical diagnoses.

---

# 🔮 Future Improvements

Possible future development includes:

- Larger and more diverse datasets
- External validation datasets
- Multi-region bone classification
- Bone-region localization
- Advanced segmentation
- Additional CNN architectures
- Model ensemble techniques
- Better uncertainty estimation
- Improved Grad-CAM visualization
- User authentication
- Prediction history
- Database integration
- Docker deployment
- Cloud deployment
- REST API
- Automated model monitoring

---

# 👨‍💻 Skills Demonstrated

This project demonstrates practical experience with:

- Python
- Machine Learning
- Deep Learning
- Computer Vision
- TensorFlow
- Keras
- Transfer Learning
- EfficientNet
- CNN
- OpenCV
- NumPy
- Scikit-learn
- Explainable AI
- Grad-CAM
- Flask
- HTML
- CSS
- JavaScript
- Model Evaluation
- Data Cleaning
- Git
- GitHub

---

# 📌 Project Status

```text
✅ Dataset preparation
✅ Image validation
✅ Image repair
✅ Preprocessing
✅ Baseline CNN
✅ EfficientNetB1
✅ Model evaluation
✅ Confusion matrix
✅ ROC curve
✅ Grad-CAM
✅ Command-line prediction
✅ Flask backend
✅ Web interface
✅ Upload workflow
✅ Prediction visualization
```

---

# 📜 License

The source code of this project can be published under an appropriate open-source license selected by the project author.

The dataset is subject to the license and terms specified by its original dataset provider. Please review those terms before redistributing dataset files.

---

# ⭐ Acknowledgement

This project was developed as an educational/research implementation exploring deep learning and explainable AI for X-ray image classification.

If you find the project useful, consider giving the repository a ⭐ on GitHub.
```