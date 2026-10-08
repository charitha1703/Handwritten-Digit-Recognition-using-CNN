# Handwritten Digit Recognition — ML Project

A complete CNN (Convolutional Neural Network) project that recognizes
handwritten digits (0-9) using the MNIST dataset, built with TensorFlow/Keras.
Achieves ~99% test accuracy.

## Project structure
```
digit_recognition/
├── models/
│   └── digit_recognizer.keras   (created after training)
├── outputs/                      (plots + results, created after training)
├── train_model.py                (main script: loads data, builds CNN, trains, evaluates)
├── predict.py                    (predict on test images OR your own drawn digit)
├── requirements.txt
└── README.md
```

## About the dataset
This project uses **MNIST** — the classic 70,000-image handwritten digit
dataset (60,000 training + 10,000 test images, each 28x28 pixels, grayscale).
You do **not** need to manually download anything — `train_model.py`
downloads it automatically the first time you run it (via
`keras.datasets.mnist`), and caches it locally (~11 MB) so it's instant
after that.

---

## Step-by-step: How to run this in VS Code

### 1. Install prerequisites
- Install **Python** (3.9–3.12) from https://python.org — check "Add Python to PATH" during install.
- Install **VS Code** from https://code.visualstudio.com
- In VS Code, install the **Python extension** (Microsoft) from the Extensions panel (`Ctrl+Shift+X`).

### 2. Get the project onto your machine
- Copy this whole `digit_recognition` folder to your computer.
- Open VS Code → File → Open Folder → select `digit_recognition`.

### 3. Open a terminal inside VS Code
- Terminal menu → New Terminal (or `` Ctrl+` ``).

### 4. Create a virtual environment
```bash
python -m venv venv
```
Activate it:
- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

If VS Code prompts "Select Interpreter", choose the one inside `venv`.

### 5. Install the required libraries
```bash
pip install -r requirements.txt
```
> Note: TensorFlow is a large package (~500 MB) — this step can take a
> few minutes depending on your internet speed. This is normal.

### 6. Create the folders the scripts need
```bash
mkdir models
mkdir outputs
```

### 7. Train the model
```bash
python train_model.py
```
This will:
- Download MNIST automatically the first time (needs internet, one-time only)
- Show sample digit images and class distribution → saved to `outputs/`
- Build and train a CNN (about 5–15 minutes on a normal laptop CPU;
  faster if you have a GPU)
- Print a classification report and save a confusion matrix
- Save prediction examples (including any mistakes, highlighted in red)
- Save the trained model to `models/digit_recognizer.keras`

Expect output like:
```
Test Accuracy: 0.9920
Test Loss: 0.0245
```

### 8. Try predictions
**Option A — quick demo on random test images (no setup):**
```bash
python predict.py
```
This grabs 5 random images from the test set, predicts them, and saves
the result to `outputs/prediction_demo.png`.

**Option B — predict your own handwritten digit:**
1. Draw a digit (0–9) on paper with a thick dark pen, take a clear photo
   or scan of it, and save it as `my_digit.png` in the project folder
   (or use any simple drawing tool like Paint — draw a white canvas with
   a black digit).
2. Run:
```bash
python predict.py my_digit.png
```
3. Check the printed prediction and confidence, and view
   `outputs/custom_prediction.png`.

> Tip: for best results, make sure the digit fills most of the image
> and is roughly centered — this is how MNIST images are formatted.

### 9. View all the results
Open the `outputs/` folder in VS Code's file explorer and click any
`.png` file to preview it directly in the editor:
- `sample_digits.png` — example training images
- `class_distribution.png` — how many of each digit in the dataset
- `training_curves.png` — accuracy/loss over training epochs
- `confusion_matrix.png` — which digits get confused with each other
- `sample_predictions.png` — model predictions vs actual labels
- `classification_report.txt` — precision/recall/F1 per digit

---

## What to say about this project on your resume

> **Handwritten Digit Recognition (CNN)** — Built a convolutional neural
> network in TensorFlow/Keras to classify handwritten digits from the
> MNIST dataset, achieving 99.2% test accuracy. Implemented batch
> normalization, dropout regularization, and early stopping to prevent
> overfitting. Evaluated with confusion matrix and per-class precision/recall,
> and built an inference pipeline for predicting on custom hand-drawn images.

## Possible extensions (great interview talking points)
- **Data augmentation** — random rotations/shifts (`ImageDataGenerator` or
  `keras.layers.RandomRotation`) to make the model more robust
- **Try different architectures** — compare a simple MLP vs. this CNN vs.
  a deeper ResNet-style model, and report accuracy differences
- **Build a drawing-canvas web app** — use Streamlit or a simple HTML
  `<canvas>` + Flask backend so users can draw a digit in the browser and
  get a live prediction
- **Deploy it** — host on Render/Hugging Face Spaces with a live demo link
  for your resume
- **Convert to TensorFlow Lite** — show you understand deploying models to
  mobile/edge devices
