import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

tf.keras.utils.set_random_seed(42)

# 1. Load and preprocess MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# 2. Build CNN
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])

# 3. Compile and train
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, batch_size=64,
          validation_split=0.2, verbose=1)

# 4. Evaluate
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# 5. Predict test images
probabilities = model.predict(x_test, verbose=0)
predicted_labels = np.argmax(probabilities, axis=1)

correct_indices = np.where(predicted_labels == y_test)[0]
incorrect_indices = np.where(predicted_labels != y_test)[0]

print("Correctly classified images:", len(correct_indices))
print("Incorrectly classified images:", len(incorrect_indices))

# 6. Display correctly classified examples
if len(correct_indices) > 0:
    selected = correct_indices[:10]
    plt.figure(figsize=(12, 5))
    for plot_number, index in enumerate(selected, start=1):
        plt.subplot(2, 5, plot_number)

