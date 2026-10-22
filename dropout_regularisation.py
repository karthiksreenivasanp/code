import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, layers, datasets, utils, regularizers

(X_tr, y_tr), (X_te, y_te) = datasets.cifar10.load_data()
X_tr, X_te = X_tr / 255.0, X_te / 255.0
y_tr = utils.to_categorical(y_tr, 10)
y_te = utils.to_categorical(y_te, 10)

configs = {
    "Baseline": (0.0, None),
    "Dropout": (0.5, None),
    "L2": (0.0, regularizers.l2(0.001))
}
hist = {}

for name, (drop, reg) in configs.items():
    model = Sequential([
        layers.Flatten(input_shape=(32, 32, 3)),
        layers.Dense(256, activation='relu', kernel_regularizer=reg),
        layers.Dropout(drop),
        layers.Dense(128, activation='relu', kernel_regularizer=reg),
        layers.Dropout(drop),
        layers.Dense(64, activation='relu', kernel_regularizer=reg),
        layers.Dropout(drop),
        layers.Dense(10, activation='softmax')
    ])
    model.compile('sgd', 'categorical_crossentropy', metrics=['accuracy'])
    h = model.fit(X_tr, y_tr, validation_split=0.2, epochs=20, batch_size=64)
    hist[name] = h.history
    acc = model.evaluate(X_te, y_te, verbose=0)[1]
    print(name)
    print("Train Acc:", round(h.history['accuracy'][-1] * 100, 2))
    print("Val Acc  :", round(h.history['val_accuracy'][-1] * 100, 2))
    print("Test Acc :", round(acc * 100, 2))

for k in ['accuracy', 'val_accuracy', 'loss', 'val_loss']:
    for name, h in hist.items():
        plt.plot(h[k], label=name)
    plt.title(k)
    plt.legend()
    plt.show()