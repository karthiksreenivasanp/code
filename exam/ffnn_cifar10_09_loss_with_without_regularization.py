"""Q: FFNN (3 hidden layers) on CIFAR-10. Visualise training loss vs validation loss
for models with and without regularisation.

Algorithm:
1. Load/preprocess CIFAR-10.
2. Build FFNN without regularisation, and with L2(0.001) + Dropout(0.3).
3. Train both for 30 epochs.
4. Plot train vs val loss side by side and compare.
"""
from tensorflow.keras.layers import Dropout
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

def build(regularized):
    reg = l2(0.001) if regularized else None
    layers = [Input(shape=(32, 32, 3)), Flatten()]
    for u in (512, 256, 128):
        layers.append(Dense(u, activation="relu", kernel_regularizer=reg))
        if regularized: layers.append(Dropout(0.3))
    layers.append(Dense(10, activation="softmax"))
    m = Sequential(layers)
    m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return m

histories, tests = {}, {}
for name, r in [("Without Regularization", False), ("With Regularization (L2 + Dropout)", True)]:
    print(f"\n=== {name} ===")
    model = build(r)
    histories[name] = model.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=64, verbose=0)
    tests[name] = model.evaluate(X_test, y_test, verbose=0)[1]
    h = histories[name].history
    print(f"Train loss {h['loss'][-1]:.4f} | Val loss {h['val_loss'][-1]:.4f} | Test acc {tests[name]:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for a, (n, hh) in zip(ax, histories.items()):
    a.plot(hh.history["loss"], label="Training Loss"); a.plot(hh.history["val_loss"], label="Validation Loss")
    a.set_title(n); a.set_xlabel("Epoch"); a.set_ylabel("Loss"); a.legend(); a.grid(True)
plt.tight_layout(); plt.show()
print("Without regularisation the validation loss rises while training loss keeps falling (overfitting).")
print("With regularisation the two curves stay closer together.")
