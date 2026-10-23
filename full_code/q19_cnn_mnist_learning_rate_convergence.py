"""
Question 19
Implement a CNN for MNIST classification and study the effect of changing the learning rate on model convergence.

Not in your files - adapted from Experiment 6 (same CNN, trained at four learning rates).
Algorithm:
1. Load and preprocess MNIST dataset.
2. Define a function to build the LeNet-like CNN given a specific learning rate for Adam.
3. Train the model across learning rates: [0.1, 0.01, 0.001, 0.0001].
4. Compare final test loss and test accuracy.
5. Plot Training Loss and Validation Accuracy convergence curves for each learning rate.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1. Load and preprocess MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (x_train / 255.0).reshape(-1, 28, 28, 1)
x_test = (x_test / 255.0).reshape(-1, 28, 28, 1)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. Build CNN function parameterized by learning rate
def create_cnn(learning_rate):
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(6, (5, 5), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(16, (5, 5), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(120, activation='relu'),
        Dense(84, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

learning_rates = [0.1, 0.01, 0.001, 0.0001]
history = {}
results = {}

# 3. Train models across learning rates
for lr in learning_rates:
    print(f"\nTraining with learning rate = {lr}...")
    model = create_cnn(lr)
    history[lr] = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    results[lr] = (loss, acc)

# 4. Compare test results
print("\n" + "=" * 55)
print("EFFECT OF LEARNING RATE ON CNN CONVERGENCE")
print("=" * 55)
for lr in results:
    loss, acc = results[lr]
    print(f"LR = {lr:<8} | Test Loss: {loss:.4f} | Test Accuracy: {acc * 100:.2f}%")
print("=" * 55)

# 5. Plot convergence curves
plt.figure(figsize=(14, 5))

# Subplot 1: Training Loss
plt.subplot(1, 2, 1)
for lr in history:
    plt.plot(history[lr].history['loss'], label=f"LR = {lr}")
plt.title("Training Loss for Different Learning Rates")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Subplot 2: Validation Accuracy
plt.subplot(1, 2, 2)
for lr in history:
    plt.plot(history[lr].history['val_accuracy'], label=f"LR = {lr}")
plt.title("Validation Accuracy for Different Learning Rates")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
