
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, layers, datasets, utils

# Load Dataset
(X_tr, y_tr), (X_te, y_te) = datasets.cifar10.load_data()

# Preprocessing
X_tr = X_tr / 255.0
X_te = X_te / 255.0

y_tr = utils.to_categorical(y_tr, 10)
y_te = utils.to_categorical(y_te, 10)

# Create Two Models
model1 = Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Flatten(),
    layers.Dense(256, activation='relu', kernel_initializer='glorot_normal'),
    layers.Dense(128, activation='relu', kernel_initializer='glorot_normal'),
    layers.Dense(64, activation='relu', kernel_initializer='glorot_normal'),
    layers.Dense(10, activation='softmax')
])

model2 = Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Flatten(),
    layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
    layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
    layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
    layers.Dense(10, activation='softmax')
])

# Compile Both Models
model1.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
model2.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

# Train Both Models
print("Training Xavier")
h1 = model1.fit(X_tr, y_tr, epochs=8, batch_size=256, validation_split=0.2)

print("Training Kaiming")
h2 = model2.fit(X_tr, y_tr, epochs=8, batch_size=256, validation_split=0.2)

# Test Both Models
loss1, acc1 = model1.evaluate(X_te, y_te)
loss2, acc2 = model2.evaluate(X_te, y_te)

# Print Results
print("Xavier Accuracy:", acc1 * 100)
print("Xavier Loss:", loss1)

print("Kaiming Accuracy:", acc2 * 100)
print("Kaiming Loss:", loss2)

# Accuracy Graph
plt.plot(h1.history['accuracy'], label='Xavier')
plt.plot(h2.history['accuracy'], label='Kaiming')
plt.title('Training Accuracy')
plt.legend()
plt.show()

# Loss Graph
plt.plot(h1.history['loss'], label='Xavier')
plt.plot(h2.history['loss'], label='Kaiming')
plt.title('Training Loss')
plt.legend()
plt.show()

# Comparison
plt.bar(['Xavier', 'Kaiming'], [acc1*100, acc2*100])
plt.title('Test Accuracy Comparison')
plt.show()
