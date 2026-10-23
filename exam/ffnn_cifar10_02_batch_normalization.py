"""Q: FFNN (3 hidden layers) on CIFAR-10. Add Batch Normalization and observe its
effect on training speed and accuracy.

Algorithm:
1. Load and preprocess CIFAR-10.
2. Build the same network twice: without BN and with BN (Dense -> BN -> ReLU).
3. Train both identically, timing each run.
4. Compare time, final accuracy and convergence curves.
"""
import time
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

def build(use_bn):
    layers = [Input(shape=(32, 32, 3)), Flatten()]
    for units in (512, 256, 128):
        if use_bn:
            layers += [Dense(units), tf.keras.layers.BatchNormalization(), tf.keras.layers.Activation("relu")]
        else:
            layers += [Dense(units, activation="relu")]
    layers += [Dense(10, activation="softmax")]
    m = Sequential(layers)
    m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return m

EPOCHS = 15
histories, times, tests = {}, {}, {}
for name, bn in [("Without BatchNorm", False), ("With BatchNorm", True)]:
    print(f"\n=== {name} ===")
    model = build(bn)
    t0 = time.perf_counter()
    histories[name] = model.fit(X_train, y_train, validation_split=0.2, epochs=EPOCHS, batch_size=64, verbose=1)
    times[name] = time.perf_counter() - t0
    tests[name] = model.evaluate(X_test, y_test, verbose=0)[1]

print("\n--- Comparison ---")
for n in histories:
    h = histories[n].history
    print(f"{n:18s} | Time: {times[n]:6.1f}s | Train: {h['accuracy'][-1]:.4f} | "
          f"Val: {h['val_accuracy'][-1]:.4f} | Test: {tests[n]:.4f} | Best val loss: {min(h['val_loss']):.4f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for n, hh in histories.items():
    ax[0].plot(hh.history["val_accuracy"], label=n)
    ax[1].plot(hh.history["val_loss"], label=n)
ax[0].set_title("Validation Accuracy"); ax[1].set_title("Validation Loss")
for a in ax: a.set_xlabel("Epoch"); a.legend(); a.grid(True)
plt.tight_layout(); plt.show()
