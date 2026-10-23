"""
Q8. CNN for MNIST with convolution, pooling, flattening and fully connected layers; train and evaluate

Dataset: MNIST
Architecture:
- Input: 28x28x1
- Conv2D (6 filters, 5x5, ReLU)
- MaxPooling2D (2x2)
- Conv2D (16 filters, 5x5, ReLU)
- MaxPooling2D (2x2)
- Flatten
- Dense (120, ReLU)
- Dense (84, ReLU)
- Dense (10, Softmax)
Goal: Display model.summary(), train, and evaluate test loss and accuracy.

Algorithm:
1. Load MNIST; normalize; add channel dimension.
2. Add Conv + MaxPool stages.
3. Flatten feature maps.
4. Add fully connected layers and 10-class Softmax output.
5. Compile (Adam), train with validation data.
6. Evaluate on test set.
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

# 2, 3, 4. Build CNN with Conv, Pool, Flatten, and Dense layers
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

# 5. Compile model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Display architecture summary
print("=" * 60)
print("Model Architecture Summary")
print("=" * 60)
model.summary()

# 5. Train with validation data
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1)

# 6. Evaluate on test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("\n" + "=" * 40)
print(f"Test Loss    : {test_loss:.4f}")
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print("=" * 40)
