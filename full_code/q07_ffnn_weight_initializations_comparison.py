"""
Q7. Neural network with different weight initializations (Xavier, Kaiming); compare training performance

Dataset: CIFAR-10
Initializers: Default baseline, Glorot Normal (Xavier), He Normal (Kaiming)
Goal: Compare final training, validation, and test accuracy across weight initializations.

Algorithm:
1. Load and preprocess CIFAR-10.
2. Build FFNN with default (baseline), Glorot Normal and He Normal initializers.
3. Compile all with SGD and categorical cross-entropy.
4. Train all under identical conditions (20 epochs).
5. Record and compare final training, validation and test accuracy.
"""

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.initializers import glorot_normal, he_normal

# 1. Load and preprocess CIFAR-10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 2. Build model with optional initializer
def create_model(init=None):
    kw = {} if init is None else {"kernel_initializer": init}
    m = Sequential([
        Input(shape=(32, 32, 3)),
        Flatten(),
        Dense(256, activation="relu", **kw),
        Dense(128, activation="relu", **kw),
        Dense(64, activation="relu", **kw),
        Dense(10, activation="softmax")
    ])
    # 3. Compile with SGD
    m.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
    return m

# Models to compare
models = {
    "Baseline": create_model(),
    "Xavier": create_model(glorot_normal()),
    "Kaiming": create_model(he_normal())
}

# 4 & 5. Train and record performance
print("=" * 60)
print("Weight Initialization Performance Comparison (CIFAR-10)")
print("=" * 60)
for name, model in models.items():
    print(f"Training {name} initialized model...")
    h = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2, verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"{name:>8} | Train={h.history['accuracy'][-1]:.4f} | "
          f"Val={h.history['val_accuracy'][-1]:.4f} | Test={test_acc:.4f}")
print("=" * 60)
