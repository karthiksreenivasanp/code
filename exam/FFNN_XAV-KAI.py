import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense

tf.keras.utils.set_random_seed(42)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Smaller subset keeps execution time manageable.
x_train = x_train[:20000]
y_train = y_train[:20000]

initializers = {
    "Xavier": "glorot_uniform",
    "Kaiming": "he_normal"
}
histories = {}
results = {}

for name, initializer in initializers.items():
    print(f"\nTraining with {name} initialization")
    tf.keras.utils.set_random_seed(42)

    model = Sequential([
        Input(shape=(28, 28)),
        Flatten(),
        Dense(128, activation="relu", kernel_initializer=initializer),
        Dense(10, activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    history = model.fit(
        x_train, y_train, epochs=8, batch_size=128,
        validation_split=0.1, verbose=1
    )
    histories[name] = history.history
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    results[name] = (test_loss, test_accuracy)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test loss: {test_loss:.4f}")

plt.figure(figsize=(8, 5))
for name in initializers:
    plt.plot(histories[name]["loss"], label=f"{name} training")
plt.title("Training Loss: Xavier vs Kaiming")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Q4_training_loss.png")
plt.show()

plt.figure(figsize=(8, 5))
for name in initializers:
    plt.plot(histories[name]["val_loss"], label=f"{name} validation")
plt.title("Validation Loss: Xavier vs Kaiming")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Q4_validation_loss.png")
plt.show()

print("\nMethod\t\tTest accuracy\tTest loss")
for name, (loss, accuracy) in results.items():
    print(f"{name}\t\t{accuracy * 100:.2f}%\t\t{loss:.4f}")

best = min(histories, key=lambda key: min(histories[key]["val_loss"]))
print("Lowest validation loss during training:", best)
print("Compare the curves; do not assume one initializer always wins.")
print("Experiment completed successfully!")
