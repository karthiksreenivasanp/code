"""Q: FFNN (3 hidden layers) on CIFAR-10. Compare ReLU vs Sigmoid activation.

Algorithm:
1. Load and preprocess CIFAR-10.
2. Build 3-hidden-layer FFNN with a chosen hidden activation (ReLU / Sigmoid).
3. Train both under identical conditions.
4. Compare train/val/test accuracy and plot curves.
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

def build(act):
    m = Sequential([
        Input(shape=(32, 32, 3)), Flatten(),
        Dense(512, activation=act), Dense(256, activation=act), Dense(128, activation=act),
        Dense(10, activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return m

histories, tests = {}, {}
for act in ["relu", "sigmoid"]:
    print(f"\n=== {act.upper()} ===")
    model = build(act)
    histories[act] = model.fit(X_train, y_train, validation_split=0.2, epochs=15, batch_size=64, verbose=1)
    tests[act] = model.evaluate(X_test, y_test, verbose=0)[1]

print("\n--- ReLU vs Sigmoid ---")
for a, hh in histories.items():
    h = hh.history
    print(f"{a:8s} | Train: {h['accuracy'][-1]:.4f} | Val: {h['val_accuracy'][-1]:.4f} | Test: {tests[a]:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for a, hh in histories.items():
    ax[0].plot(hh.history["val_accuracy"], label=f"{a} val acc")
    ax[1].plot(hh.history["loss"], label=f"{a} train loss")
    ax[1].plot(hh.history["val_loss"], "--", label=f"{a} val loss")
ax[0].set_title("Validation Accuracy"); ax[1].set_title("Loss")
for x in ax: x.set_xlabel("Epoch"); x.legend(); x.grid(True)
plt.tight_layout(); plt.show()
print("Observation: ReLU avoids vanishing gradients, so it usually trains faster and reaches higher accuracy than Sigmoid.")
