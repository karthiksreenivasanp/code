"""
Q5. CNN for MNIST digit classification; evaluate using test accuracy

Dataset: MNIST
Architecture: Conv2D(6, 5x5) -> MaxPool(2x2) -> Conv2D(16, 5x5) -> MaxPool(2x2) -> Flatten -> Dense(120) -> Dense(84) -> Dense(10)
Goal: Train CNN, evaluate test accuracy, and display sample predictions.

Algorithm:
1. Load MNIST; normalize; add channel dimension.
2. Conv(6, 5x5) -> MaxPool(2x2) -> Conv(16, 5x5) -> MaxPool(2x2).
3. Flatten -> Dense 120 -> Dense 84 -> Dense 10 (Softmax).
4. Compile (Adam, categorical cross-entropy).
5. Train 10 epochs, batch 64, 20% validation.
6. Evaluate and print test accuracy.
"""

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# 1. Load MNIST, normalize, and add channel dimension
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train.astype("float32") / 255.0, -1)
x_test = np.expand_dims(x_test.astype("float32") / 255.0, -1)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2 & 3. Build LeNet-style CNN model
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(6, (5, 5), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(16, (5, 5), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(120, activation="relu"),
    Dense(84, activation="relu"),
    Dense(10, activation="softmax")
])

# 4. Compile model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 5. Train model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1)

# 6. Evaluate on test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Predict and compare first 10 samples
pred = model.predict(x_test[:10], verbose=0)
print("\nSample Predictions:")
print("-" * 35)
for i in range(10):
    print(f"Sample {i}: Predicted={np.argmax(pred[i])}, Actual={np.argmax(y_test[i])}")
