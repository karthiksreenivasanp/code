"""
Q4. FFNN with Xavier and Kaiming initialization; plot loss curves and decide better convergence

Dataset: CIFAR-10
Initializers: Glorot Normal (Xavier) vs. He Normal (Kaiming)
Optimizer: SGD
Goal: Compare validation loss curves and identify which converges better/faster.

Algorithm:
1. Load and preprocess CIFAR-10.
2. Build the same FFNN twice: Glorot Normal (Xavier) and He Normal (Kaiming).
3. Compile both with SGD and categorical cross-entropy.
4. Train 20 epochs, 20% validation.
5. Plot validation loss for both.
6. Compare minimum validation loss and the epoch it occurs; lower and earlier = better convergence.
"""

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.initializers import glorot_normal, he_normal
import matplotlib.pyplot as plt

# 1. Load and preprocess CIFAR-10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 2. Build model with specified kernel initializer
def create_model(init):
    m = Sequential([
        Input(shape=(32, 32, 3)),
        Flatten(),
        Dense(256, activation="relu", kernel_initializer=init),
        Dense(128, activation="relu", kernel_initializer=init),
        Dense(64, activation="relu", kernel_initializer=init),
        Dense(10, activation="softmax")
    ])
    # 3. Compile with SGD
    m.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
    return m

models = {
    "Xavier": create_model(glorot_normal()),
    "Kaiming": create_model(he_normal())
}

# 4. Train models
histories = {}
for name, model in models.items():
    print(f"Training {name} initialized model...")
    histories[name] = model.fit(
        X_train, y_train, epochs=20, batch_size=64,
        validation_split=0.2, verbose=0
    )
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    vl = histories[name].history["val_loss"]
    # 6. Compare minimum validation loss
    best_loss = min(vl)
    best_epoch = vl.index(best_loss) + 1
    print(f"{name}: Test Acc={acc:.4f}, Best Val Loss={best_loss:.4f} at Epoch {best_epoch}")

# 5. Plot validation loss for both
for name in histories:
    plt.plot(histories[name].history["val_loss"], label=f"{name} Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Xavier vs Kaiming Validation Loss")
plt.legend()
plt.grid(True)
plt.show()
