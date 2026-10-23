"""
Question 17
Implement a CNN for MNIST digit recognition and visualize correctly and incorrectly classified test images.

From your files: Experiment 6 (CNN) plus code for showing correct and wrong images.
Algorithm:
1. Load and preprocess MNIST dataset with channel dimension (28, 28, 1).
2. Build LeNet-like CNN (Conv 6 -> MaxPool -> Conv 16 -> MaxPool -> Dense 120 -> Dense 84 -> Dense 10).
3. Train model with 20% validation split.
4. Predict on test set and identify correct vs incorrect predictions.
5. Display counts and visualize 5 correctly classified (green) and 5 incorrectly classified (red) images.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1. Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# 2. Create LeNet-like CNN model
model = Sequential([
    Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(16, kernel_size=(5, 5), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(120, activation='relu'),
    Dense(84, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 3. Train model
print("Training CNN model on MNIST...")
history = model.fit(
    x_train, y_train_cat, epochs=10,
    batch_size=64, validation_split=0.2, verbose=1
)

# 4. Evaluate and Predict
test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"\nTest accuracy: {test_accuracy * 100:.2f}%")

predictions = model.predict(x_test, verbose=0)
pred_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test_cat, axis=1)

correct_idx = np.where(pred_classes == true_classes)[0]
wrong_idx = np.where(pred_classes != true_classes)[0]

print("=" * 40)
print("Correctly classified   :", len(correct_idx))
print("Incorrectly classified :", len(wrong_idx))
print("=" * 40)

# 5. Show 5 correctly classified images
plt.figure(figsize=(12, 3))
for k, i in enumerate(correct_idx[:5]):
    plt.subplot(1, 5, k + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {pred_classes[i]}\nTrue: {true_classes[i]}", color='green')
    plt.axis('off')
plt.suptitle("Correctly Classified Images (Green)", fontsize=12)
plt.tight_layout()
plt.show()

# 6. Show 5 incorrectly classified images
plt.figure(figsize=(12, 3))
for k, i in enumerate(wrong_idx[:5]):
    plt.subplot(1, 5, k + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {pred_classes[i]}\nTrue: {true_classes[i]}", color='red')
    plt.axis('off')
plt.suptitle("Incorrectly Classified Images (Red)", fontsize=12)
plt.tight_layout()
plt.show()
