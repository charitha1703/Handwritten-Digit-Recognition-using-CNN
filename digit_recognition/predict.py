 
 

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras

MODEL_PATH = "models/digit_recognizer.keras"
model = keras.models.load_model(MODEL_PATH)


def predict_from_mnist_test_set(n=5):
     
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
    
    from PIL import Image, ImageFilter

    img = Image.open(image_path).convert("L")  # grayscale
    img_array = np.array(img).astype("float32")

     
    if img_array.mean() > 127:
        img_array = 255 - img_array

 
    threshold = 30
    mask = img_array > threshold

    if mask.sum() == 0:
        raise ValueError(
            "No digit detected in the image (it looks blank). "
            "Make sure the digit is drawn clearly and the file is correct."
        )

    
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped = img_array[rmin:rmax + 1, cmin:cmax + 1]
 
   
    h, w = cropped.shape
    if h > w:
        new_h, new_w = 20, max(1, round(20 * w / h))
    else:
        new_w, new_h = 20, max(1, round(20 * h / w))

    cropped_img = Image.fromarray(cropped.astype("uint8")).resize(
        (new_w, new_h), Image.LANCZOS
    )
   
    cropped_img = cropped_img.filter(ImageFilter.MaxFilter(3))
    resized = np.array(cropped_img).astype("float32")

     
    canvas = np.zeros((28, 28), dtype="float32")
    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    img_array = canvas / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array


def predict_from_image_file(image_path):
    
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
