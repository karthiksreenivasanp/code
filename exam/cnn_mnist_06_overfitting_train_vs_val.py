"""Q: Design a CNN for MNIST and compare its training and validation performance to
identify overfitting.

Algorithm:
1. Load/preprocess MNIST; build CNN (no regularisation) so overfitting can appear.
2. Train 20 epochs with 20% validation.
3. Plot train vs validation accuracy/loss.
4. Compute the accuracy gap and the epoch where val loss is lowest; report overfitting.
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
history = model.fit(x_train, y_train, validation_split=0.2, epochs=20, batch_size=128, verbose=1)
h = history.history
print(f"\nTest accuracy: {model.evaluate(x_test, y_test, verbose=0)[1]*100:.2f}%")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(h["accuracy"], label="Training"); ax[0].plot(h["val_accuracy"], label="Validation"); ax[0].set_title("Accuracy")
ax[1].plot(h["loss"], label="Training"); ax[1].plot(h["val_loss"], label="Validation"); ax[1].set_title("Loss")
for a in ax: a.set_xlabel("Epoch"); a.legend(); a.grid(True)
plt.tight_layout(); plt.show()

gap = h["accuracy"][-1] - h["val_accuracy"][-1]
best = int(np.argmin(h["val_loss"])) + 1
print("\n--- Overfitting Analysis ---")
print(f"Final train acc: {h['accuracy'][-1]:.4f} | Final val acc: {h['val_accuracy'][-1]:.4f} | Gap: {gap:.4f}")
print(f"Lowest val loss ({min(h['val_loss']):.4f}) at epoch {best}; final val loss {h['val_loss'][-1]:.4f}")
if h["val_loss"][-1] > min(h["val_loss"]) * 1.1 and gap > 0.005:
    print(f"OVERFITTING detected: val loss rises after epoch {best} while training accuracy keeps improving.")
else:
    print("No strong overfitting: curves stay close. Use more epochs or remove regularisation to see it more clearly.")
print("Remedies: Dropout, L2, EarlyStopping, data augmentation.")
