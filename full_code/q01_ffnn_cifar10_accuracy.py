"""
Q1. Feed Forward Neural Network (FFNN) for classification; evaluate using accuracy

Dataset: CIFAR-10
Architecture: Dense 256 -> Dense 256 -> Dense 128 (ReLU) -> Dense 10 (Softmax)
Evaluation Metric: Accuracy

Algorithm:
1. Import libraries and load CIFAR-10.
2. Normalize pixels to 0–1; one-hot encode labels.
3. Flatten images (32x32x3 = 3072).
4. Build FFNN: Dense 256, 256, 128 (ReLU) + Dense 10 (Softmax).
5. Compile (Adam, categorical cross-entropy, accuracy).
6. Train 10 epochs, batch 64, 20% validation.
7. Evaluate test accuracy; plot training/validation accuracy.
"""

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1. Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# 2. Normalize pixels to 0-1 and one-hot encode labels
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 3. Flatten images (32 x 32 x 3 = 3072)
X_train = X_train.reshape(-1, 3072)
X_test = X_test.reshape(-1, 3072)

# 4. Build FFNN model
model = Sequential([
    Input(shape=(3072,)),
    Dense(256, activation="relu"),
    Dense(256, activation="relu"),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])

# 5. Compile model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 6. Train model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=64, verbose=1)

# 7. Evaluate test accuracy
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Plot training vs validation accuracy
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("FFNN Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()
