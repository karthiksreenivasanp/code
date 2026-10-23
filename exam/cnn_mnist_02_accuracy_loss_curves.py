"""Q: Design and train a CNN for MNIST. Plot accuracy and loss curves and interpret the trend.

Algorithm:
1. Load/preprocess MNIST, build the CNN (Conv-Pool-Conv-Pool-Flatten-Dense).
2. Train 12 epochs with 20% validation, keeping the History object.
3. Plot training vs validation accuracy and loss.
4. Print an automatic interpretation of the trend.
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

model = build_cnn()
history = model.fit(x_train, y_train, validation_split=0.2, epochs=12, batch_size=128, verbose=1)
h = history.history
_, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc*100:.2f}%")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(h["accuracy"], label="Training Accuracy"); ax[0].plot(h["val_accuracy"], label="Validation Accuracy")
ax[0].set_title("Accuracy"); ax[1].plot(h["loss"], label="Training Loss"); ax[1].plot(h["val_loss"], label="Validation Loss")
ax[1].set_title("Loss")
for a in ax: a.set_xlabel("Epoch"); a.legend(); a.grid(True)
plt.tight_layout(); plt.show()

best = int(np.argmin(h["val_loss"])) + 1
print("\n--- Interpretation ---")
print(f"Training accuracy rose from {h['accuracy'][0]:.3f} to {h['accuracy'][-1]:.3f}; "
      f"validation accuracy from {h['val_accuracy'][0]:.3f} to {h['val_accuracy'][-1]:.3f}.")
print(f"Lowest validation loss {min(h['val_loss']):.4f} at epoch {best}.")
if best < len(h["val_loss"]) - 2 or h["accuracy"][-1] - h["val_accuracy"][-1] > 0.01:
    print("Validation loss stops improving while training keeps improving -> mild overfitting begins after that epoch.")
else:
    print("Both curves improve together and stay close -> good fit, no significant overfitting.")
