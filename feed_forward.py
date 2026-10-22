import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, layers, datasets, utils

(X_tr, y_tr), (X_te, y_te) = datasets.cifar10.load_data()
X_tr = X_tr.reshape(-1, 3072) / 255.0
X_te = X_te.reshape(-1, 3072) / 255.0
y_tr = utils.to_categorical(y_tr, 10)
y_te = utils.to_categorical(y_te, 10)

model = Sequential([
    layers.Dense(256, activation='relu', input_shape=(3072,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
h = model.fit(X_tr, y_tr, validation_split=0.2, epochs=10, batch_size=64)
print("Test Accuracy:", model.evaluate(X_te, y_te)[1])

for i, m in enumerate(['loss', 'accuracy']):
    plt.subplot(1, 2, i + 1)
    plt.plot(h.history[m], label='Train ' + m)
    plt.plot(h.history['val_' + m], label='Val ' + m)
    plt.title(m.title())
    plt.legend()
plt.show()