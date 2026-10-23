"""Q: FFNN (3 hidden layers) on CIFAR-10. Apply Dropout (p = 0.5) and evaluate impact.

Algorithm:
1. Load/preprocess CIFAR-10.
2. Build FFNN without dropout (baseline) and with Dropout(0.5) after each hidden layer.
3. Train both identically.
4. Compare train/val/test accuracy and the train-val gap.
"""
from tensorflow.keras.layers import Dropout
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

def build(p):
    layers = [Input(shape=(32, 32, 3)), Flatten()]
    for u in (512, 256, 128):
        layers.append(Dense(u, activation="relu"))
        if p > 0: layers.append(Dropout(p))
    layers.append(Dense(10, activation="softmax"))
    m = Sequential(layers)
    m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return m

histories, tests = {}, {}
for name, p in [("No Dropout", 0.0), ("Dropout 0.5", 0.5)]:
    print(f"\n=== {name} ===")
    model = build(p)
    histories[name] = model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=64, verbose=0)
    tests[name] = model.evaluate(X_test, y_test, verbose=0)[1]

print("\n--- Dropout Impact ---")
for n, hh in histories.items():
    h = hh.history
    gap = h["accuracy"][-1] - h["val_accuracy"][-1]
    print(f"{n:12s} | Train: {h['accuracy'][-1]:.4f} | Val: {h['val_accuracy'][-1]:.4f} | Test: {tests[n]:.4f} | Gap: {gap:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for n, hh in histories.items():
    ax[0].plot(hh.history["val_accuracy"], label=f"{n} val")
    ax[0].plot(hh.history["accuracy"], "--", label=f"{n} train")
    ax[1].plot(hh.history["val_loss"], label=f"{n} val")
    ax[1].plot(hh.history["loss"], "--", label=f"{n} train")
ax[0].set_title("Accuracy"); ax[1].set_title("Loss")
for a in ax: a.set_xlabel("Epoch"); a.legend(); a.grid(True)
plt.tight_layout(); plt.show()
