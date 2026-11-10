import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.utils import to_categorical

tf.keras.utils.set_random_seed(42)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("Training data:", x_train.shape)
print("Testing data:", x_test.shape)

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential([
    Input(shape=(32, 32, 3)),
    Flatten(),
    Dense(512, activation="relu"),
    Dense(10, activation="softmax")
])

model.summary()
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(x_train, y_train, epochs=15, batch_size=128,
          validation_split=0.1, verbose=1)

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print("Experiment completed successfully!")
