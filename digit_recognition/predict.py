"""
predict.py
----------
Two ways to use the trained digit recognizer:

1. Predict on a few random images from the MNIST test set (no setup needed)
2. Predict on YOUR OWN image file (e.g. a photo of a handwritten digit
   you drew and saved as digit.png)

Run: python predict.py
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras

MODEL_PATH = "models/digit_recognizer.keras"
model = keras.models.load_model(MODEL_PATH)


def predict_from_mnist_test_set(n=5):
    """Predict n random digits from the MNIST test set (quick demo, no file needed)."""
    (_, _), (X_test, y_test) = keras.datasets.mnist.load_data()
    idx = np.random.choice(len(X_test), n, replace=False)

    images = X_test[idx].astype("float32") / 255.0
    images = np.expand_dims(images, -1)

    preds = model.predict(images, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    confidences = np.max(preds, axis=1)

    plt.figure(figsize=(3 * n, 3))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(X_test[idx[i]], cmap="gray")
        color = "green" if pred_labels[i] == y_test[idx[i]] else "red"
        plt.title(f"Pred: {pred_labels[i]} ({confidences[i]*100:.1f}%)\nTrue: {y_test[idx[i]]}", color=color)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("outputs/prediction_demo.png")
    plt.close()
    print("Saved outputs/prediction_demo.png")

    for i in range(n):
        print(f"Image {i+1}: predicted = {pred_labels[i]}, actual = {y_test[idx[i]]}, confidence = {confidences[i]*100:.1f}%")


def preprocess_custom_image(image_path):
    """
    Converts any hand-drawn digit photo/scan into MNIST-style format:
    - grayscale, inverted if needed (white digit on black background)
    - cropped tightly to the digit itself (removes empty margins)
    - resized so the digit's longest side is ~20px (MNIST digits aren't
      edge-to-edge, they sit inside a 28x28 canvas with padding)
    - centered on a 28x28 black canvas using the digit's center of mass
      (MNIST images are centered this way, not just centered by bounding box)
    - lines are thickened slightly, since a thin drawn/photographed line
      often becomes a faint 1-2px line after resizing, which the model
      can easily confuse for other digits (commonly a "1")
    This centering/cropping step is the single biggest fix for custom
    images being misclassified - without it, accuracy on hand-drawn
    digits is often poor even though test-set accuracy is ~99%.
    """
    from PIL import Image, ImageFilter

    img = Image.open(image_path).convert("L")  # grayscale
    img_array = np.array(img).astype("float32")

    # Ensure white digit on black background (MNIST format)
    if img_array.mean() > 127:
        img_array = 255 - img_array

    # Threshold to find the actual digit strokes vs background noise
    threshold = 30
    mask = img_array > threshold

    if mask.sum() == 0:
        raise ValueError(
            "No digit detected in the image (it looks blank). "
            "Make sure the digit is drawn clearly and the file is correct."
        )

    # Crop tightly to the bounding box of the digit
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped = img_array[rmin:rmax + 1, cmin:cmax + 1]

    # Resize so the longest side becomes 20px, preserving aspect ratio
    # (MNIST digits occupy roughly the center 20x20 area of the 28x28 image)
    h, w = cropped.shape
    if h > w:
        new_h, new_w = 20, max(1, round(20 * w / h))
    else:
        new_w, new_h = 20, max(1, round(20 * h / w))

    cropped_img = Image.fromarray(cropped.astype("uint8")).resize(
        (new_w, new_h), Image.LANCZOS
    )
    # Thicken the strokes slightly so thin lines survive downscaling
    cropped_img = cropped_img.filter(ImageFilter.MaxFilter(3))
    resized = np.array(cropped_img).astype("float32")

    # Paste onto a 28x28 black canvas, centered by center of mass
    canvas = np.zeros((28, 28), dtype="float32")
    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    img_array = canvas / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array


def predict_from_image_file(image_path):
    """
    Predict the digit in a custom image file.
    The image should be a single handwritten digit, ideally:
    - dark digit on light background OR light digit on dark background
      (auto-detected and inverted if needed)
    - reasonably clear, thick strokes (thin pencil lines may not survive
      the resize to 28x28)
    """
    img_array = preprocess_custom_image(image_path)

    pred = model.predict(img_array, verbose=0)[0]
    pred_label = np.argmax(pred)
    confidence = np.max(pred) * 100

    plt.figure(figsize=(4, 4))
    plt.imshow(img_array.squeeze(), cmap="gray")
    plt.title(f"Predicted: {pred_label} ({confidence:.1f}% confident)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("outputs/custom_prediction.png")
    plt.close()

    print(f"\nPredicted digit: {pred_label}")
    print(f"Confidence: {confidence:.1f}%")
    print("Saved visualization to outputs/custom_prediction.png")

    print("\nAll class probabilities:")
    for digit, prob in enumerate(pred):
        print(f"  {digit}: {prob*100:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Usage: python predict.py path/to/your_digit.png
        predict_from_image_file(sys.argv[1])
    else:
        print("No image path given — running demo on random MNIST test images.\n")
        predict_from_mnist_test_set(n=5)
        print("\nTip: to predict your own image, run:")
        print("   python predict.py path/to/your_digit.png")