
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, layers, datasets, utils

# Load CIFAR-10 dataset
(X_tr, y_tr), (X_te, y_te) = datasets.cifar10.load_data()

# Preprocessing
X_tr = X_tr.reshape(-1, 3072) / 255.0
X_te = X_te.reshape(-1, 3072) / 255.0

y_tr = utils.to_categorical(y_tr, 10)
y_te = utils.to_categorical(y_te, 10)

# ReLU Model
relu_model = Sequential([
    layers.Input(shape=(3072,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

relu_model.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

relu_history = relu_model.fit(
    X_tr, y_tr, validation_split=0.2,
    epochs=10, batch_size=64
)

relu_loss, relu_acc = relu_model.evaluate(X_te, y_te)

# Sigmoid Model
sigmoid_model = Sequential([
    layers.Input(shape=(3072,)),
    layers.Dense(256, activation='sigmoid'),
    layers.Dense(128, activation='sigmoid'),
    layers.Dense(10, activation='softmax')
])

sigmoid_model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

sigmoid_history = sigmoid_model.fit(
    X_tr, y_tr, validation_split=0.2,
    epochs=10, batch_size=64
)

sigmoid_loss, sigmoid_acc = sigmoid_model.evaluate(X_te, y_te)

# Tanh Model
tanh_model = Sequential([
    layers.Input(shape=(3072,)),
    layers.Dense(256, activation='tanh'),
    layers.Dense(128, activation='tanh'),
    layers.Dense(10, activation='softmax')
])

tanh_model.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

tanh_history = tanh_model.fit(
    X_tr, y_tr, validation_split=0.2,
    epochs=10, batch_size=64
)

tanh_loss, tanh_acc = tanh_model.evaluate(X_te, y_te)

# Display Results
print("ReLU Test Loss:", relu_loss)
print("ReLU Test Accuracy:", relu_acc)

print("Sigmoid Test Loss:", sigmoid_loss)
print("Sigmoid Test Accuracy:", sigmoid_acc)

print("Tanh Test Loss:", tanh_loss)
print("Tanh Test Accuracy:", tanh_acc)

# Compare Accuracy
plt.bar(['ReLU', 'Sigmoid', 'Tanh'],
        [relu_acc, sigmoid_acc, tanh_acc])

plt.title('Activation Function Comparison')
plt.xlabel('Activation Function')
plt.ylabel('Test Accuracy')
plt.show()

# Compare Test Loss
plt.bar(['ReLU', 'Sigmoid', 'Tanh'],
        [relu_loss, sigmoid_loss, tanh_loss])

plt.title('Test Loss Comparison')
plt.xlabel('Activation Function')
plt.ylabel('Test Loss')
plt.show()
