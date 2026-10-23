"""Q: FFNN (3 hidden layers) on CIFAR-10. Train/test; print training accuracy,
validation accuracy and plot loss curves.

Algorithm:
1. Load CIFAR-10, normalise to 0-1, one-hot encode labels.
2. Flatten 32x32x3 images to 3072 inputs.
3. Build Dense 512-256-128 (ReLU) + Dense 10 (Softmax).
4. Compile (Adam, categorical cross-entropy) and train with 20% validation.
5. Print train/val/test accuracy, plot accuracy and loss curves.
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.utils import to_categorical

tf.keras.utils.set_random_seed(42)

# ---- Load & preprocess CIFAR-10 ----
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
y_train_int, y_test_int = y_train.flatten(), y_test.flatten()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

model = Sequential([
    Input(shape=(32, 32, 3)), Flatten(),
    Dense(512, activation="relu"),
    Dense(256, activation="relu"),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(X_train, y_train, validation_split=0.2, epochs=15, batch_size=64, verbose=1)
h = history.history
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"\nFinal Training Accuracy  : {h['accuracy'][-1]*100:.2f}%")
print(f"Final Validation Accuracy: {h['val_accuracy'][-1]*100:.2f}%")
print(f"Test Accuracy            : {test_acc*100:.2f}%  (Test loss {test_loss:.4f})")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(h["accuracy"], label="Training Accuracy"); ax[0].plot(h["val_accuracy"], label="Validation Accuracy")
ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend(); ax[0].grid(True)
ax[1].plot(h["loss"], label="Training Loss"); ax[1].plot(h["val_loss"], label="Validation Loss")
ax[1].set_title("Loss Curves"); ax[1].set_xlabel("Epoch"); ax[1].legend(); ax[1].grid(True)
plt.tight_layout(); plt.show()
