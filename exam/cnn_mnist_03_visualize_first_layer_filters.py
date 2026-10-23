"""Q: Design and train a CNN for MNIST. Visualise the feature maps (filters) from the
first convolutional layer after training.

Algorithm:
1. Load/preprocess MNIST, build and train the CNN.
2. Plot the learned 3x3 kernels of conv1 (the filters).
3. Build a sub-model from the input to conv1 and pass one test image.
4. Plot the resulting feature maps (one per filter).
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
model.fit(x_train, y_train, validation_split=0.2, epochs=5, batch_size=128, verbose=1)
print(f"Test accuracy: {model.evaluate(x_test, y_test, verbose=0)[1]*100:.2f}%")

# ---- 1. Learned filters (kernels) ----
filters, biases = model.get_layer("conv1").get_weights()   # (3, 3, 1, 32)
print("conv1 filter shape:", filters.shape)
fig, axes = plt.subplots(4, 8, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(filters[:, :, 0, i], cmap="gray"); ax.axis("off"); ax.set_title(f"F{i+1}", fontsize=8)
plt.suptitle("Conv1 learned filters (3x3)"); plt.tight_layout(); plt.show()

# ---- 2. Feature maps for a sample image ----
sample = x_test[0:1]
feat_model = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer("conv1").output)
fmaps = feat_model.predict(sample, verbose=0)               # (1, 26, 26, 32)
print("Feature map shape:", fmaps.shape)

plt.figure(figsize=(3, 3)); plt.imshow(sample[0].squeeze(), cmap="gray")
plt.title(f"Input (label {y_test_int[0]})"); plt.axis("off"); plt.show()

fig, axes = plt.subplots(4, 8, figsize=(14, 7))
for i, ax in enumerate(axes.flat):
    ax.imshow(fmaps[0, :, :, i], cmap="viridis"); ax.axis("off"); ax.set_title(f"Map {i+1}", fontsize=8)
plt.suptitle("Conv1 feature maps"); plt.tight_layout(); plt.show()
