
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

# Test Model
test_loss, test_acc = model.evaluate(x_te, y_te)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc * 100)

# Plot Loss
plt.plot(h.history['loss'], label='Training Loss')
plt.plot(h.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()
plt.show()

# Plot Accuracy
plt.plot(h.history['accuracy'], label='Training Accuracy')
plt.plot(h.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()
plt.show()

# Predict first 10 images
preds = model.predict(x_te[:10])

# Display Predictions
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_te[i].reshape(28, 28), cmap='gray')
    plt.title(f"P:{np.argmax(preds[i])} A:{np.argmax(y_te[i])}")
    plt.axis('off')

plt.tight_layout()
plt.show()
