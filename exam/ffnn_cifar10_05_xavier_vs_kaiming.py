"""Q: FFNN (3 hidden layers) on CIFAR-10. Implement and compare Xavier and Kaiming
weight initialisation.

Algorithm:
1. Load/preprocess CIFAR-10.
2. Build identical FFNNs: GlorotNormal (Xavier) and HeNormal (Kaiming).
3. Train with SGD, 20 epochs, 20% validation.
4. Plot val loss and accuracy; compare best val loss / epoch and test accuracy.
"""
from tensorflow.keras.initializers import GlorotNormal, HeNormal
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

def build(init):
    m = Sequential([
        Input(shape=(32, 32, 3)), Flatten(),
        Dense(512, activation="relu", kernel_initializer=init),
        Dense(256, activation="relu", kernel_initializer=init),
        Dense(128, activation="relu", kernel_initializer=init),
        Dense(10, activation="softmax"),
    ])
    m.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss="categorical_crossentropy", metrics=["accuracy"])
    return m

inits = {"Xavier (Glorot)": GlorotNormal(seed=42), "Kaiming (He)": HeNormal(seed=42)}
histories = {}
for name, init in inits.items():
    print(f"\n=== {name} ===")
    model = build(init)
    histories[name] = model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=64, verbose=0)
    vl = histories[name].history["val_loss"]
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Acc={acc:.4f} | Best Val Loss={min(vl):.4f} at epoch {vl.index(min(vl))+1} | "
          f"Final Train Acc={histories[name].history['accuracy'][-1]:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for n, hh in histories.items():
    ax[0].plot(hh.history["val_loss"], label=n)
    ax[1].plot(hh.history["val_accuracy"], label=n)
ax[0].set_title("Validation Loss"); ax[1].set_title("Validation Accuracy")
for a in ax: a.set_xlabel("Epoch"); a.legend(); a.grid(True)
plt.tight_layout(); plt.show()
print("Lower and earlier minimum validation loss = faster convergence. Kaiming suits ReLU; Xavier suits tanh/sigmoid.")
