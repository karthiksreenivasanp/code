import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import Sequential, layers, datasets, preprocessing
from sklearn.metrics import confusion_matrix, classification_report

# Load data
(X_tr, y_tr), (X_te, y_te) = datasets.imdb.load_data(num_words=10000)
X_tr = preprocessing.sequence.pad_sequences(X_tr, maxlen=500)
X_te = preprocessing.sequence.pad_sequences(X_te, maxlen=500)

models = {
    'RNN': layers.SimpleRNN,
    'LSTM': layers.LSTM,
    'GRU': layers.GRU
}

hist, cms = {}, {}

# Train and test
for name, L in models.items():
    model = Sequential([
        layers.Embedding(10000, 32),
        L(32),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    h = model.fit(X_tr, y_tr, validation_split=0.2, epochs=5, batch_size=64)
    hist[name] = h.history

    loss, acc = model.evaluate(X_te, y_te, verbose=0)
    pred = (model.predict(X_te, verbose=0) >= 0.5).astype(int).flatten()
    cms[name] = confusion_matrix(y_te, pred)

    print(name, 'Loss:', round(loss, 4), 'Accuracy:', round(acc, 4))
    print('TN, FP, FN, TP:', cms[name].ravel())
    print(classification_report(y_te, pred, digits=4))

# Accuracy graph
for name, h in hist.items():
    plt.plot(h['val_accuracy'], label=name)

plt.title('Validation Accuracy')
plt.legend()
plt.show()

# Confusion matrices
for name, cm in cms.items():
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(name + ' Confusion Matrix')
    plt.show()
