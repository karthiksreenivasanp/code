
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist

# Load Dataset
tf.keras.utils.set_random_seed(42)
(x_tr, y_tr), (x_te, y_te) = mnist.load_data()

# Preprocessing
x_tr = x_tr.reshape(-1, 28, 28, 1) / 255.0
x_te = x_te.reshape(-1, 28, 28, 1) / 255.0

# Build CNN
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(x_tr, y_tr, epochs=5, batch_size=64,
          validation_split=0.2)

# Test
loss, acc = model.evaluate(x_te, y_te)
print("Test Accuracy:", acc * 100)
print("Test Loss:", loss)

# Predict
p = model.predict(x_te)
pred = np.argmax(p, axis=1)

# Correct and Incorrect
correct = np.where(pred == y_te)[0]
incorrect = np.where(pred != y_te)[0]

print("Correct predictions:", len(correct))
print("Incorrect predictions:", len(incorrect))

# Correct Images
plt.figure(figsize=(12, 5))
plt.suptitle("Correctly Classified Images")

for i, n in enumerate(correct[:10]):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_te[n].reshape(28, 28), cmap='gray')
    plt.title("P:" + str(pred[n]) + " A:" + str(y_te[n]))
    plt.axis('off')

plt.show()

# Incorrect Images
plt.figure(figsize=(12, 5))
plt.suptitle("Incorrectly Classified Images")

for i, n in enumerate(incorrect[:10]):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_te[n].reshape(28, 28), cmap='gray')
    plt.title("P:" + str(pred[n]) + " A:" + str(y_te[n]))
    plt.axis('off')

plt.show()

print("Experiment completed successfully!")
