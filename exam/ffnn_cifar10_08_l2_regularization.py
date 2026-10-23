"""Q: FFNN (3 hidden layers) on CIFAR-10. Demonstrate L2 regularisation and show how
it reduces overfitting.

Algorithm:
1. Load/preprocess CIFAR-10.
2. Build FFNN without regularisation and with L2(0.001) on Dense kernels.
3. Train both for 25 epochs.
4. Compare train-val gap and plot curves.
"""
from tensorflow.keras.regularizers import l2
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

def build(lam):
    reg = l2(lam) if lam > 0 else None
    m = Sequential([
        Input(shape=(32, 32, 3)), Flatten(),
        Dense(512, activation="relu", kernel_regularizer=reg),
        Dense(256, activation="relu", kernel_regularizer=reg),
        Dense(128, activation="relu", kernel_regularizer=reg),
        Dense(10, activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return m

histories, tests = {}, {}
for name, lam in [("No L2", 0.0), ("L2 (0.001)", 0.001)]:
    print(f"\n=== {name} ===")
    model = build(lam)
    histories[name] = model.fit(X_train, y_train, validation_split=0.2, epochs=25, batch_size=64, verbose=0)
    tests[name] = model.evaluate(X_test, y_test, verbose=0)[1]

print("\n--- L2 Regularisation ---")
for n, hh in histories.items():
    h = hh.history
    print(f"{n:11s} | Train acc: {h['accuracy'][-1]:.4f} | Val acc: {h['val_accuracy'][-1]:.4f} | "
          f"Gap: {h['accuracy'][-1]-h['val_accuracy'][-1]:.4f} | Test: {tests[n]:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for n, hh in histories.items():
    ax[0].plot(hh.history["accuracy"], "--", label=f"{n} train"); ax[0].plot(hh.history["val_accuracy"], label=f"{n} val")
    ax[1].plot(hh.history["loss"], "--", label=f"{n} train"); ax[1].plot(hh.history["val_loss"], label=f"{n} val")
ax[0].set_title("Accuracy"); ax[1].set_title("Loss (includes L2 penalty for training)")
for a in ax: a.set_xlabel("Epoch"); a.legend(); a.grid(True)
plt.tight_layout(); plt.show()
print("L2 penalises large weights, giving smoother decision boundaries and a smaller train-validation gap.")
