
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, layers, datasets, utils

# Load MNIST dataset
(x_tr, y_tr), (x_te, y_te) = datasets.mnist.load_data()

# Preprocessing
x_tr = x_tr.reshape(-1, 28, 28, 1) / 255.0
x_te = x_te.reshape(-1, 28, 28, 1) / 255.0

y_tr = utils.to_categorical(y_tr, 10)
y_te = utils.to_categorical(y_te, 10)

# Build CNN Model
model = Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(6, (5, 5), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(16, (5, 5), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(120, activation='relu'),
    layers.Dense(84, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train Model
h = model.fit(x_tr, y_tr, epochs=10,
              batch_size=64, validation_split=0.2)

# Evaluate Model
test_loss, test_acc = model.evaluate(x_te, y_te)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc * 100)

# Plot Loss
plt.plot(h.history['loss'], label='Training Loss')
plt.plot(h.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()
plt.show()

# Predict Test Images
preds = model.predict(x_te)
pred_labels = np.argmax(preds, axis=1)
actual_labels = np.argmax(y_te, axis=1)

# Find Correct and Incorrect Predictions
correct = np.where(pred_labels == actual_labels)[0]
incorrect = np.where(pred_labels != actual_labels)[0]

# Visualize Correctly Classified Images
plt.figure(figsize=(10, 4))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_te[correct[i]].reshape(28, 28), cmap='gray')
    plt.title("Pred: " + str(pred_labels[correct[i]]))
    plt.axis('off')

plt.suptitle("Correctly Classified Images")
plt.tight_layout()
plt.show()

# Visualize Incorrectly Classified Images
plt.figure(figsize=(10, 4))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_te[incorrect[i]].reshape(28, 28), cmap='gray')
    plt.title("Pred: " + str(pred_labels[incorrect[i]]) +
              " Actual: " + str(actual_labels[incorrect[i]]))
    plt.axis('off')

plt.suptitle("Incorrectly Classified Images")
plt.tight_layout()
plt.show()
