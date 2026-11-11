
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, layers, datasets, utils

# Load CIFAR-10 dataset
(X_tr, y_tr), (X_te, y_te) = datasets.cifar10.load_data()

# Preprocessing
X_tr, X_te = X_tr / 255.0, X_te / 255.0

y_tr = utils.to_categorical(y_tr, 10)
y_te = utils.to_categorical(y_te, 10)

# Xavier Initialization
model1 = Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Flatten(),
    layers.Dense(256, activation='relu', kernel_initializer='glorot_normal'),
    layers.Dense(128, activation='relu', kernel_initializer='glorot_normal'),
    layers.Dense(64, activation='relu', kernel_initializer='glorot_normal'),
    layers.Dense(10, activation='softmax')
])

model1.compile('sgd', 'categorical_crossentropy', metrics=['accuracy'])

h1 = model1.fit(X_tr, y_tr, epochs=20,
                batch_size=64, validation_split=0.2)

loss1, acc1 = model1.evaluate(X_te, y_te)

print("Xavier Test Loss:", loss1)
print("Xavier Test Accuracy:", acc1 * 100)

# Kaiming Initialization
model2 = Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Flatten(),
    layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
    layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
    layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
    layers.Dense(10, activation='softmax')
])

model2.compile('sgd', 'categorical_crossentropy', metrics=['accuracy'])

h2 = model2.fit(X_tr, y_tr, epochs=20,
                batch_size=64, validation_split=0.2)

loss2, acc2 = model2.evaluate(X_te, y_te)

print("Kaiming Test Loss:", loss2)
print("Kaiming Test Accuracy:", acc2 * 100)

# Plot Training Accuracy
plt.plot(h1.history['accuracy'], label='Xavier')
plt.plot(h2.history['accuracy'], label='Kaiming')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot Training Loss
plt.plot(h1.history['loss'], label='Xavier')
plt.plot(h2.history['loss'], label='Kaiming')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Compare Test Accuracy
plt.bar(['Xavier', 'Kaiming'], [acc1 * 100, acc2 * 100])
plt.title('Test Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.show()
