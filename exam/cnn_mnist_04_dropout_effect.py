"""Q: Design and train a CNN for MNIST. Modify the CNN by adding a Dropout layer and
evaluate its effect on validation accuracy.

Algorithm:
1. Load/preprocess MNIST.
2. Train CNN without dropout, then CNN with Dropout(0.5) before the output layer.
3. Compare validation accuracy, test accuracy and train-val gap.
4. Plot validation accuracy / loss for both.
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

tf.keras.utils.set_random_seed(42)

# ---- Load & preprocess MNIST ----
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train.astype("float32") / 255.0, -1)
x_test = np.expand_dims(x_test.astype("float32") / 255.0, -1)
y_train_int, y_test_int = y_train.copy(), y_test.copy()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

def build_cnn(dropout=0.0):
    layers = [
        Input(shape=(28, 28, 1)),
        Conv2D(32, (3, 3), activation="relu", name="conv1"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu", name="conv2"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
    ]
    if dropout > 0: layers.append(Dropout(dropout))
    layers.append(Dense(10, activation="softmax"))
    m = Sequential(layers)
    m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return m

histories, tests = {}, {}
for name, p in [("No Dropout", 0.0), ("With Dropout (0.5)", 0.5)]:
    print(f"\n=== {name} ===")
    tf.keras.utils.set_random_seed(42)
    model = build_cnn(p)
    histories[name] = model.fit(x_train, y_train, validation_split=0.2, epochs=12, batch_size=128, verbose=0)
    tests[name] = model.evaluate(x_test, y_test, verbose=0)[1]

print("\n--- Effect of Dropout ---")
for n, hh in histories.items():
    h = hh.history
    print(f"{n:20s} | Train: {h['accuracy'][-1]:.4f} | Val: {h['val_accuracy'][-1]:.4f} | "
          f"Best Val: {max(h['val_accuracy']):.4f} | Test: {tests[n]:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for n, hh in histories.items():
    ax[0].plot(hh.history["val_accuracy"], label=n); ax[1].plot(hh.history["val_loss"], label=n)
ax[0].set_title("Validation Accuracy"); ax[1].set_title("Validation Loss")
for a in ax: a.set_xlabel("Epoch"); a.legend(); a.grid(True)
plt.tight_layout(); plt.show()
