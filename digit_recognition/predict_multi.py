import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras

MODEL_PATH = "models/digit_recognizer.keras"
model = keras.models.load_model(MODEL_PATH)
 


def sort_reading_order(boxes):
    
    if not boxes:
        return boxes
  
    boxes_by_y = sorted(boxes, key=lambda b: b[1] + b[3] / 2)
    median_height = float(np.median([b[3] for b in boxes]))
    row_gap_threshold = 0.6 * median_height  # how close in y counts as "same row"
 
    rows = []
    current_row = [boxes_by_y[0]]
    current_row_y_center = boxes_by_y[0][1] + boxes_by_y[0][3] / 2
 
    for box in boxes_by_y[1:]:
        y_center = box[1] + box[3] / 2
        if abs(y_center - current_row_y_center) <= row_gap_threshold:
            current_row.append(box)
            # keep the row's reference y-center updated as a running average
            current_row_y_center = np.mean([b[1] + b[3] / 2 for b in current_row])
        else:
            rows.append(current_row)
            current_row = [box]
            current_row_y_center = y_center
    rows.append(current_row)
 
    
    ordered = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda b: b[0]))
 
    return ordered




def segment_digits(image_path, min_area=40):
     
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    
    if img.mean() > 127:
        img = 255 - img

 
    _, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

     
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area:  # skip tiny noise specks
            continue
        boxes.append((x, y, w, h))

    if not boxes:
        raise ValueError(
            "No digits detected in the image. Make sure digits are drawn "
            "clearly with a thick, dark pen on a plain background."
        )

    
    boxes = sort_reading_order(boxes)

     

    digits = []
    for (x, y, w, h) in boxes:
        # crop with a little padding around each digit
        pad = max(4, int(0.15 * max(w, h)))
        y0, y1 = max(0, y - pad), min(img.shape[0], y + h + pad)
        x0, x1 = max(0, x - pad), min(img.shape[1], x + w + pad)
        digit_crop = img[y0:y1, x0:x1]
        digits.append((x, y, w, h, digit_crop))

    return digits, thresh


def preprocess_digit(digit_array):
    
    h, w = digit_array.shape
 
    _, binary = cv2.threshold(digit_array, 30, 255, cv2.THRESH_BINARY)

     
    kernel_size = max(3, int(min(h, w) * 0.04))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    thickened = cv2.dilate(binary, kernel, iterations=1)

     
    if h > w:
        new_h, new_w = 20, max(1, round(20 * w / h))
    else:
        new_w, new_h = 20, max(1, round(20 * h / w))
    resized = cv2.resize(thickened, (new_w, new_h), interpolation=cv2.INTER_AREA)

   
    canvas = np.zeros((28, 28), dtype="float32")
    y_off, x_off = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    canvas = canvas / 255.0
    return canvas.reshape(1, 28, 28, 1)


def predict_multi_digit(image_path):
    digits, thresh = segment_digits(image_path)

    predicted_sequence = []
    confidences = []
    crops_for_display = []

    for (x, y, w, h, digit_crop) in digits:
        processed = preprocess_digit(digit_crop)
        pred = model.predict(processed, verbose=0)[0]
        label = np.argmax(pred)
        confidence = np.max(pred) * 100

        predicted_sequence.append(str(label))
        confidences.append(confidence)
        crops_for_display.append(processed.squeeze())

    result_number = "".join(predicted_sequence)

    print(f"\nDetected {len(digits)} digit(s)")
    print(f"Predicted sequence: {result_number}")
    for i, (d, c) in enumerate(zip(predicted_sequence, confidences)):
        print(f"  Digit {i+1}: {d}  (confidence: {c:.1f}%)")

   
    img_color = cv2.imread(image_path)
    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 1, figsize=(max(6, len(digits) * 1.5), 6))

    axes[0].imshow(img_color)
    for (x, y, w, h), label in zip([(d[0], d[1], d[2], d[3]) for d in digits], predicted_sequence):
        axes[0].add_patch(plt.Rectangle((x, y), w, h, edgecolor="lime", facecolor="none", linewidth=2))
        axes[0].text(x, y - 8, label, color="lime", fontsize=14, fontweight="bold")
    axes[0].set_title(f"Detected sequence: {result_number}")
    axes[0].axis("off")

    n = len(crops_for_display)
    for i, crop in enumerate(crops_for_display):
        ax = fig.add_subplot(2, n, n + i + 1)
        ax.imshow(crop, cmap="gray")
        ax.set_title(predicted_sequence[i])
        ax.axis("off")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("outputs/multi_digit_prediction.png")
    plt.close()
    print("\nSaved visualization to outputs/multi_digit_prediction.png")

    return result_number


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_multi.py path/to/image_with_multiple_digits.png")
        sys.exit(1)

    predict_multi_digit(sys.argv[1])
