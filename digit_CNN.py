import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, layers, datasets, utils

(x_tr, y_tr), (x_te, y_te) = datasets.mnist.load_data()
x_tr = x_tr.reshape(-1, 28, 28, 1) / 255.0
x_te = x_te.reshape(-1, 28, 28, 1) / 255.0
y_tr = utils.to_categorical(y_tr, 10)
y_te = utils.to_categorical(y_te, 10)

model = Sequential([
    layers.Conv2D(6, (5, 5), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(16, (5, 5), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(120, activation='relu'),
    layers.Dense(84, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
model.fit(x_tr, y_tr, epochs=10, batch_size=64, validation_split=0.2)
acc = model.evaluate(x_te, y_te)[1]
print("Test accuracy:", round(acc * 100, 2))

preds = model.predict(x_te[:10])
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_te[i].reshape(28, 28), cmap='gray')
    plt.title(f"P:{np.argmax(preds[i])} A:{np.argmax(y_te[i])}")
    plt.axis('off')
plt.show()