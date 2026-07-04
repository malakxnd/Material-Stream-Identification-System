# Material Stream Identification (MSI) System

A machine learning pipeline that classifies waste materials — **cardboard, glass, metal, paper, plastic, and trash** — from images and live webcam video. Built as the final project for the Machine Learning course, Faculty of Computers and Artificial Intelligence, Cairo University (Spring 2026).

## Overview

The system combines **handcrafted (HOG)** and **deep learned (ResNet50)** image features and feeds them into classical classifiers — **SVM** and **k-NN** — to identify the material type of an object. A real-time camera application then uses the trained SVM model to classify objects live from a webcam feed, with an "Unknown" rejection mechanism for low-confidence predictions.

This README walks through the full process end to end: setting up your environment, running the notebooks in order to build the dataset and train the models, and finally running the live camera app with the models you produced.

## Project Structure

```
.
├── models/                          # Trained models are saved here once you run the notebooks
├── src/
│   ├── 1. data_aug.ipynb            # Data augmentation & dataset balancing pipeline
│   ├── 2. feature_extraction.ipynb  # HOG + ResNet50 feature extraction
│   ├── SVM_Model.ipynb              # SVM training, tuning, and evaluation
│   ├── KNN_Model.ipynb              # k-NN training, tuning, and evaluation
│   └── camera_app.py                # Real-time webcam classification app
├── Report_and_Repo.pdf              # Full project report (methodology, results, discussion)
├── SVM report.pdf                   # SVM-specific report
└── KNN report.pdf                   # k-NN-specific report
└── dataset/
```

The `models/` folder starts empty (or absent) in a fresh clone — it's populated automatically when you run the notebooks in Step 2 below.

## Pipeline at a Glance

1. **Data Augmentation** — corrects class imbalance and expands the dataset.
2. **Feature Extraction** — converts each image into a fixed-length numerical vector combining HOG and ResNet50 CNN features.
3. **Model Training** — trains and evaluates SVM and k-NN classifiers, and saves the final models.
4. **Real-Time Inference** — runs the trained SVM model live on webcam input.

Each stage is explained in detail below, in the order you should run it.

---

## Step 1: Set Up Your Environment

You'll need **Python 3.9+** and Jupyter Notebook (or JupyterLab / VS Code with the Jupyter extension) to run the `.ipynb` files.

Install the required libraries:

```bash
pip install opencv-python numpy pandas matplotlib scikit-learn tensorflow joblib scikit-image jupyter
```

**What each library is used for:**

| Library | Purpose |
|---|---|
| `opencv-python` (`cv2`) | Image loading, preprocessing, augmentation, webcam capture |
| `numpy` | Numerical operations and array handling |
| `pandas` | Dataset organization and analysis |
| `matplotlib` | Visualizing class distributions and sample images |
| `scikit-learn` | Train/validation/test splitting, scaling, SVM, k-NN, PCA |
| `scikit-image` | HOG feature computation |
| `tensorflow` | ResNet50 backbone for deep feature extraction |
| `joblib` | Saving and loading trained models and preprocessing objects |
| `jupyter` | Running the `.ipynb` notebooks |

If you're using a GPU for faster training, install a GPU-enabled build of TensorFlow appropriate for your system instead of the default CPU package.

You'll also need your own image dataset of the six material classes (`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`), organized into one folder per class, to feed into the augmentation notebook.

---

## Step 2: Run the Notebooks in Order

Open the `src/` folder in Jupyter and run the following notebooks **in this exact order**. Each notebook produces outputs that the next one depends on.

### 2.1 — `1. data_aug.ipynb`

Balances and augments your raw dataset:

- Applies rotation (±12°), horizontal flipping, scaling (0.90–1.00 zoom), brightness/contrast adjustment, and low-level Gaussian noise.
- Augmentation is applied **only to the training split**, with duplicate and leakage checks across splits, to avoid data leakage.
- Balances every class to roughly the same number of images (~500 per class in the original run).
- Produces a structured `augmented_dataset/` directory with one subfolder per class.

Point this notebook at your raw dataset folder before running it, and let it finish fully — later steps depend on its output directory.

### 2.2 — `2. feature_extraction.ipynb`

Converts every image into a single fixed-length feature vector:

- **HOG** — captures shape, edges, and structural texture (8×8 cells, 2×2 blocks, 9 orientation bins, L2-Hys normalization).
- **ResNet50 + GlobalAveragePooling2D** — a frozen, ImageNet-pretrained backbone that captures texture, transparency, and color-related semantic cues.
- Both feature sets are scaled (fit on training data only, then applied to validation/test) and concatenated into one combined vector per image.
- Saves the extracted feature matrices, labels, and the fitted scalers/encoders needed to reproduce this exact transformation later — including at inference time in the camera app.

Run this notebook fully before moving on; the model-training notebooks load its saved outputs directly.

### 2.3 — `SVM_Model.ipynb` and `KNN_Model.ipynb`

Train and evaluate the two classifiers on the extracted features:

- Both notebooks compare performance on HOG-only, CNN-only, and combined HOG+CNN feature sets.
- Both apply a confidence-based rejection mechanism (threshold = 0.5): predictions with low confidence are labeled **Unknown** instead of being forced into one of the six classes.
- Run **`SVM_Model.ipynb`** in full — this is the notebook that produces the models used by the live camera app. Run `KNN_Model.ipynb` as well if you want to compare or use k-NN separately.
- At the end of `SVM_Model.ipynb`, the trained classifier, its PCA transform, feature scalers, and label encoder are all saved into the `models/` folder automatically. Once this notebook completes, everything the camera app needs is in place.

---

## Step 3: Run the Real-Time Camera App

Once `SVM_Model.ipynb` has finished and populated `models/`, you're ready to classify materials live.

```bash
cd src
python camera_app.py
```

**How it works:**

- Opens your webcam and captures video at 640×480, 30 FPS.
- Draws a green bounding box (a 250×250 pixel region of interest) in the center of the frame — place the material inside this box.
- Every 10th frame (~3 predictions/second), it extracts features from the region using the same HOG + ResNet50 pipeline used during training, then classifies with the trained SVM.
- Displays the predicted class, a confidence percentage, and a color-coded confidence bar on screen. Predictions below the 0.5 confidence threshold are labeled **Unknown**.
- Press **Q** to quit.

`camera_app.py` automatically loads its required model files from the `models/` folder relative to its own location, so no extra configuration is needed as long as Step 2 completed successfully.

---

## Results

| Feature Set | SVM Val Acc | SVM Test Acc | k-NN Val Acc | k-NN Test Acc |
|---|---|---|---|---|
| HOG only | 0.606 | 0.601 | 0.580 | 0.615 |
| CNN only | 0.864 | 0.922 | 0.867 | 0.883 |
| HOG + CNN | 0.835 | 0.908 | 0.885 | 0.887 |

**With rejection (threshold = 0.5):**

| Model | Accuracy with Rejection |
|---|---|
| SVM | 91.17% |
| k-NN | 90.07% |

CNN features alone provide the strongest signal, while combining HOG and CNN features improves k-NN performance further. See `Report_and_Repo.pdf` for full methodology, ablations, and discussion.

## Reports

- **`Report_and_Repo.pdf`** — Full technical report covering data augmentation, feature extraction methodology, real-time integration, and SVM vs. k-NN comparison.
- **`SVM report.pdf`** — Detailed SVM training and evaluation report.
- **`KNN report.pdf`** — Detailed k-NN training and evaluation report.

## Troubleshooting

- **Camera app can't find model files** — make sure `SVM_Model.ipynb` ran to completion; it's responsible for saving everything the app needs into `models/`.
- **Feature extraction is slow** — ResNet50 feature extraction is the heaviest step; a GPU-enabled TensorFlow install will speed this up significantly.
- **Webcam doesn't open** — check that no other application is using the camera, and that OpenCV has permission to access it on your OS.
- **Predictions are mostly "Unknown"** — try improving lighting, keeping the material fully inside the bounding box, or reviewing the confidence threshold in `camera_app.py`.
