"""
Q6. CIFAR-10 CNN with Xavier and Kaiming initialization + L2 and Dropout; compare accuracy and loss curves

Dataset: CIFAR-10
Initializers: Xavier (Glorot Normal) vs. Kaiming (He Normal)
Regularization: L2 regularization (0.001) + Dropout (0.25, 0.25, 0.5)
Goal: Train both configurations and compare training/validation accuracy and loss curves.

Algorithm:
1. Load, normalize CIFAR-10; one-hot encode labels.
2. Build a CNN function taking an initializer; use L2(0.001) and Dropout (0.25, 0.25, 0.5).
3. Create Xavier and Kaiming models with identical settings.
4. Compile (SGD, categorical cross-entropy) and train 10 epochs.
5. Plot training/validation accuracy and loss.
6. Evaluate test accuracy and compare.
"""

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.initializers import glorot_normal, he_normal
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

# 1. Load and preprocess CIFAR-10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 2. Function to build regularized CNN
def build_cnn(init):
    reg = l2(0.001)
    m = Sequential([
        Input(shape=(32, 32, 3)),
        Conv2D(32, (3, 3), activation="relu", kernel_initializer=init, kernel_regularizer=reg),
        Conv2D(64, (3, 3), activation="relu", kernel_initializer=init, kernel_regularizer=reg),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation="relu", kernel_initializer=init, kernel_regularizer=reg),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation="relu", kernel_initializer=init, kernel_regularizer=reg),
        Dropout(0.5),
        Dense(10, activation="softmax")
    ])
    # 4. Compile with SGD
    m.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
    return m

# 3. Instantiate both models
models = {
    "Xavier": build_cnn(glorot_normal()),
    "Kaiming": build_cnn(he_normal())
}

# 4. Train both models
histories = {}
for name, model in models.items():
    print(f"Training {name} CNN model...")
    histories[name] = model.fit(
        X_train, y_train, validation_split=0.2,
        epochs=10, batch_size=64, verbose=0
    )
    # 6. Evaluate test accuracy
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"{name}: Test Loss={loss:.4f}, Test Accuracy={acc:.4f}")

# 5. Plot training/validation accuracy curves
plt.figure(figsize=(9, 5))
for name in histories:
    plt.plot(histories[name].history["accuracy"], label=f"{name} Train Acc")
    plt.plot(histories[name].history["val_accuracy"], "--", label=f"{name} Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curves (Xavier vs Kaiming with L2 + Dropout)")
plt.legend()
plt.grid(True)
plt.show()

# 5. Plot training/validation loss curves
plt.figure(figsize=(9, 5))
for name in histories:
    plt.plot(histories[name].history["loss"], label=f"{name} Train Loss")
    plt.plot(histories[name].history["val_loss"], "--", label=f"{name} Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curves (Xavier vs Kaiming with L2 + Dropout)")
plt.legend()
plt.grid(True)
plt.show()
