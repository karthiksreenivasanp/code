"""Q: FFNN (3 hidden layers) on CIFAR-10. Train with SGD, Adam, RMSprop and
compare convergence.

Algorithm:
1. Load/preprocess CIFAR-10.
2. Build the same FFNN for each optimiser (fresh weights, same seed).
3. Train 15 epochs each, 20% validation.
4. Plot training loss / validation accuracy; compare test accuracy.
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

def build(opt):
    tf.keras.utils.set_random_seed(42)
    m = Sequential([
        Input(shape=(32, 32, 3)), Flatten(),
        Dense(512, activation="relu"), Dense(256, activation="relu"), Dense(128, activation="relu"),
        Dense(10, activation="softmax"),
    ])
    m.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return m

optimizers = {
    "SGD": lambda: tf.keras.optimizers.SGD(learning_rate=0.01),
    "Adam": lambda: tf.keras.optimizers.Adam(learning_rate=0.001),
    "RMSprop": lambda: tf.keras.optimizers.RMSprop(learning_rate=0.001),
}
histories, tests = {}, {}
for name, make in optimizers.items():
    print(f"\n=== {name} ===")
    model = build(make())
    histories[name] = model.fit(X_train, y_train, validation_split=0.2, epochs=15, batch_size=64, verbose=0)
    tests[name] = model.evaluate(X_test, y_test, verbose=0)[1]

print("\n--- Optimizer Comparison ---")
for n, hh in histories.items():
    h = hh.history
    print(f"{n:8s} | Train loss: {h['loss'][-1]:.4f} | Val acc: {h['val_accuracy'][-1]:.4f} | Test acc: {tests[n]:.4f}")

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for n, hh in histories.items():
    ax[0].plot(hh.history["loss"], label=n)
    ax[1].plot(hh.history["val_loss"], label=n)
    ax[2].plot(hh.history["val_accuracy"], label=n)
for a, t in zip(ax, ["Training Loss", "Validation Loss", "Validation Accuracy"]):
    a.set_title(t); a.set_xlabel("Epoch"); a.legend(); a.grid(True)
plt.tight_layout(); plt.show()
