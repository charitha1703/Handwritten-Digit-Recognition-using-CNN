"""
train_model.py
---------------
Handwritten Digit Recognition using a Convolutional Neural Network (CNN)
trained on the MNIST dataset (70,000 images of handwritten digits 0-9).

The MNIST dataset is downloaded automatically the first time you run this
script (via keras.datasets.mnist) - no manual dataset file needed. It's
cached locally afterwards, so it only downloads once.

What this script does:
1. Loads and explores the MNIST dataset
2. Preprocesses images (normalize, reshape)
3. Builds a CNN using TensorFlow/Keras
4. Trains the model with validation
5. Evaluates on the test set (accuracy, confusion matrix, classification report)
6. Saves training curves and sample predictions as images in outputs/
7. Saves the trained model to models/digit_recognizer.keras

Run:  python train_model.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # save plots to file, no display needed
import matplotlib.pyplot as plt
import seaborn as sns
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# -----------------------------------------------------------------
# 1. LOAD DATA (downloads automatically the first time, then caches)
# -----------------------------------------------------------------
print("Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print(f"Training images: {X_train.shape}")   # (60000, 28, 28)
print(f"Test images: {X_test.shape}")         # (10000, 28, 28)
print(f"Classes: {np.unique(y_train)}")       # digits 0-9

# -----------------------------------------------------------------
# 2. EDA: visualize a few sample digits
# -----------------------------------------------------------------
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i], cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")
plt.tight_layout()
plt.savefig("outputs/sample_digits.png")
plt.close()
print("Saved sample digit images to outputs/sample_digits.png")

# Class distribution
plt.figure(figsize=(8, 5))
sns.countplot(x=y_train)
plt.title("Digit Class Distribution (Training Set)")
plt.xlabel("Digit")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/class_distribution.png")
plt.close()

# -----------------------------------------------------------------
# 3. PREPROCESSING
# -----------------------------------------------------------------
# Normalize pixel values from 0-255 to 0-1 (helps the network train faster)
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Reshape to add the "channels" dimension CNNs expect: (28, 28) -> (28, 28, 1)
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

# Set aside part of the training set for validation during training
val_split = 5000
X_val, y_val = X_train[:val_split], y_train[:val_split]
X_train_final, y_train_final = X_train[val_split:], y_train[val_split:]

print(f"\nAfter preprocessing:")
print(f"Train: {X_train_final.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

# -----------------------------------------------------------------
# 4. BUILD THE CNN MODEL
# -----------------------------------------------------------------
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, kernel_size=3, activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(32, kernel_size=3, activation="relu"),
    layers.MaxPooling2D(pool_size=2),
    layers.Dropout(0.25),

    layers.Conv2D(64, kernel_size=3, activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(64, kernel_size=3, activation="relu"),
    layers.MaxPooling2D(pool_size=2),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),  # 10 output classes: digits 0-9
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# -----------------------------------------------------------------
# 5. TRAIN THE MODEL
# -----------------------------------------------------------------
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

EPOCHS = 15
BATCH_SIZE = 128

print("\nTraining model...")
history = model.fit(
    X_train_final, y_train_final,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=2,
)

# -----------------------------------------------------------------
# 6. PLOT TRAINING CURVES
# -----------------------------------------------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("outputs/training_curves.png")
plt.close()
print("Saved training curves to outputs/training_curves.png")

# -----------------------------------------------------------------
# 7. EVALUATE ON TEST SET
# -----------------------------------------------------------------
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report:")
report = classification_report(y_test, y_pred, digits=4)
print(report)
with open("outputs/classification_report.txt", "w") as f:
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n\n")
    f.write(report)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Digit")
plt.ylabel("Actual Digit")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.close()
print("Saved confusion matrix to outputs/confusion_matrix.png")

# -----------------------------------------------------------------
# 8. VISUALIZE SOME PREDICTIONS (including any mistakes)
# -----------------------------------------------------------------
plt.figure(figsize=(12, 6))
for i in range(15):
    plt.subplot(3, 5, i + 1)
    plt.imshow(X_test[i].squeeze(), cmap="gray")
    color = "green" if y_pred[i] == y_test[i] else "red"
    plt.title(f"Pred: {y_pred[i]} | True: {y_test[i]}", color=color)
    plt.axis("off")
plt.tight_layout()
plt.savefig("outputs/sample_predictions.png")
plt.close()
print("Saved sample predictions to outputs/sample_predictions.png")

# -----------------------------------------------------------------
# 9. SAVE THE TRAINED MODEL
# -----------------------------------------------------------------
model.save("models/digit_recognizer.keras")
print("\nSaved trained model to models/digit_recognizer.keras")
print("\nDone! Check the 'outputs/' folder for plots and 'models/' for the saved model.")
