"""
Q2. FFNN for MNIST digit classification; display test accuracy and plot training/validation accuracy

Dataset: MNIST
Architecture: Dense 128 -> Dense 64 (ReLU) -> Dense 10 (Softmax)
Evaluation Metric: Accuracy

Algorithm:
1. Load MNIST; normalize to 0–1.
2. Flatten 28x28 to 784; one-hot encode labels.
3. Build Dense 128, 64 (ReLU) + Dense 10 (Softmax).
4. Compile (Adam, categorical cross-entropy).
5. Train with 20% validation split.
6. Print test accuracy; plot accuracy curves.
"""

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1 & 2. Load MNIST, normalize, and flatten
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32").reshape(-1, 784) / 255.0
x_test = x_test.astype("float32").reshape(-1, 784) / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 3. Build FFNN model
model = Sequential([
    Input(shape=(784,)),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])

# 4. Compile model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 5. Train model with 20% validation split
history = model.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=64, verbose=1)

# 6. Evaluate and print test accuracy
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Plot accuracy curves
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MNIST FFNN: Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()
