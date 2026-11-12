
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, layers, datasets, utils

# Load CIFAR-10 dataset
(X_tr, y_tr), (X_te, y_te) = datasets.cifar10.load_data()

# Preprocess data
X_tr = X_tr.reshape(-1, 3072) / 255.0
X_te = X_te.reshape(-1, 3072) / 255.0

y_tr = utils.to_categorical(y_tr, 10)
y_te = utils.to_categorical(y_te, 10)

# Build model
model = Sequential([
    layers.Dense(256, activation='relu', input_shape=(3072,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
h = model.fit(X_tr, y_tr,
              validation_split=0.2,
              epochs=10,
              batch_size=64)

# Evaluate test data
test_loss, test_accuracy = model.evaluate(X_te, y_te)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Plot Loss
plt.plot(h.history['loss'], label='Training Loss')
plt.plot(h.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot Accuracy
plt.plot(h.history['accuracy'], label='Training Accuracy')
plt.plot(h.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
