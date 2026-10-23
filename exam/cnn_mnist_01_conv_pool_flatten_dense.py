"""Q: Design and train a CNN for MNIST using Conv2D, MaxPooling, Flatten and Dense layers.

Algorithm:
1. Load MNIST, normalise to 0-1, add channel dimension, one-hot encode labels.
2. Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool -> Flatten -> Dense(128) -> Dense(10, softmax).
3. Compile (Adam, categorical cross-entropy), train 10 epochs with 20% validation.
4. Evaluate on test set and show sample predictions.
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
model.summary()
history = model.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=128, verbose=1)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest loss: {test_loss:.4f} | Test accuracy: {test_acc*100:.2f}%")

pred = np.argmax(model.predict(x_test[:10], verbose=0), axis=1)
print("Predicted:", pred); print("Actual   :", y_test_int[:10])
fig, ax = plt.subplots(1, 10, figsize=(15, 2))
for i in range(10):
    ax[i].imshow(x_test[i].squeeze(), cmap="gray"); ax[i].set_title(f"P:{pred[i]}"); ax[i].axis("off")
plt.show()
