import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense

tf.keras.utils.set_random_seed(42)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Use the same subset and training settings for each activation.
x_train = x_train[:20000]
y_train = y_train[:20000]

activations = ["relu", "sigmoid", "tanh"]
results = []

for activation in activations:
    print(f"\nTraining with {activation} activation")
    tf.keras.utils.set_random_seed(42)

    model = Sequential([
        Input(shape=(28, 28)),
        Flatten(),
        Dense(128, activation=activation),
        Dense(10, activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    history = model.fit(
        x_train, y_train, epochs=5, batch_size=128,
        validation_split=0.1, verbose=1
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    results.append((activation, test_accuracy, test_loss))
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test loss: {test_loss:.4f}")

names = [item[0] for item in results]
accuracies = [item[1] * 100 for item in results]

print("\nActivation\tTest accuracy\tTest loss")
for name, accuracy, loss in results:
    print(f"{name}\t\t{accuracy:.4f}\t\t{loss:.4f}")

plt.figure(figsize=(8, 5))
plt.bar(names, accuracies)
plt.title("Test Accuracy for Different Activation Functions")
plt.xlabel("Activation function")
plt.ylabel("Test accuracy (%)")
plt.ylim(0, 100)
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("Q3_activation_comparison.png")
plt.show()
print("Experiment completed successfully!")
