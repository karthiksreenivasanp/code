"""
Q3. FFNN with at least one hidden layer; compare different activation functions

Dataset: CIFAR-10
Activations compared: ReLU, Sigmoid, Tanh
Goal: Evaluate and compare train, validation, and test accuracy across activations.

Algorithm:
1. Load, normalize and flatten CIFAR-10; one-hot encode labels.
2. Write a function that builds the FFNN with a chosen hidden activation.
3. Train separate models with ReLU, Sigmoid and Tanh.
4. Evaluate each on the same test set.
5. Print and compare train, validation and test accuracy.
"""

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical

# 1. Load, normalize, and flatten CIFAR-10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype("float32").reshape(-1, 3072) / 255.0
X_test = X_test.astype("float32").reshape(-1, 3072) / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. Function to build FFNN with variable activation function
def build_model(act):
    m = Sequential([
        Input(shape=(3072,)),
        Dense(256, activation=act),
        Dense(128, activation=act),
        Dense(10, activation="softmax")
    ])
    m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return m

# 3 & 4. Train and evaluate models with ReLU, Sigmoid, and Tanh
results = {}
for act in ["relu", "sigmoid", "tanh"]:
    print(f"Training model with '{act}' activation...")
    model = build_model(act)
    h = model.fit(X_train, y_train, validation_split=0.2, epochs=5, batch_size=64, verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    results[act] = (h.history["accuracy"][-1], h.history["val_accuracy"][-1], test_acc)

# 5. Print and compare results
print("\n" + "=" * 50)
print("Activation Function Comparison (CIFAR-10)")
print("=" * 50)
for act, (tr, va, te) in results.items():
    print(f"{act:>7} | Train: {tr:.4f} | Val: {va:.4f} | Test: {te:.4f}")
print("=" * 50)
