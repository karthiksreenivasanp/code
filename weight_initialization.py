import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, layers, datasets, utils

(X_tr, y_tr), (X_te, y_te) = datasets.cifar10.load_data()
X_tr, X_te = X_tr / 255.0, X_te / 255.0
y_tr = utils.to_categorical(y_tr, 10)
y_te = utils.to_categorical(y_te, 10)

inits = {"Baseline": None, "Xavier": "glorot_normal", "Kaiming": "he_normal"}
hist, res = {}, {}

for name, init in inits.items():
    model = Sequential([
        layers.Flatten(input_shape=(32, 32, 3)),
        layers.Dense(256, activation='relu', kernel_initializer=init),
        layers.Dense(128, activation='relu', kernel_initializer=init),
        layers.Dense(64, activation='relu', kernel_initializer=init),
        layers.Dense(10, activation='softmax')
    ])
    model.compile('sgd', 'categorical_crossentropy', metrics=['accuracy'])
    h = model.fit(X_tr, y_tr, epochs=20, batch_size=64, validation_split=0.2)
    hist[name] = h.history
    loss, acc = model.evaluate(X_te, y_te, verbose=0)
    res[name] = acc * 100
    print(name, "| Loss:", round(loss, 4), "| Acc:", round(acc * 100, 2))

for i, m in enumerate(['accuracy', 'loss']):
    plt.subplot(1, 3, i + 1)
    for name, h in hist.items():
        plt.plot(h[m], label=name + " Train")
        plt.plot(h['val_' + m], '--', label=name + " Val")
    plt.title(m.title())
    plt.legend()

plt.subplot(1, 3, 3)
plt.bar(res.keys(), res.values())
plt.title("Test Accuracy (%)")
plt.show()