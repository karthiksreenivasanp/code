import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import Sequential, layers, datasets, preprocessing
from sklearn.metrics import confusion_matrix, classification_report

(x_tr, y_tr), (x_te, y_te) = datasets.imdb.load_data(num_words=10000)
x_tr = preprocessing.sequence.pad_sequences(x_tr, maxlen=500)
x_te = preprocessing.sequence.pad_sequences(x_te, maxlen=500)

models = {'SimpleRNN': layers.SimpleRNN, 'LSTM': layers.LSTM, 'GRU': layers.GRU}
hist, cms = {}, {}

for name, Layer in models.items():
    model = Sequential([
        layers.Embedding(10000, 32),
        Layer(32),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    h = model.fit(x_tr, y_tr, epochs=5, batch_size=64, validation_split=0.2)
    hist[name] = h.history
    loss, acc = model.evaluate(x_te, y_te, verbose=0)
    y_pred = (model.predict(x_te, verbose=0) >= 0.5).astype(int).flatten()
    cms[name] = confusion_matrix(y_te, y_pred)
    print("\nModel:", name)
    print("Loss :", round(loss, 4), "| Acc:", round(acc, 4))
    print("TN, FP, FN, TP:", cms[name].ravel())
    print(classification_report(y_te, y_pred, digits=4))

for name, h in hist.items():
    plt.plot(h['val_accuracy'], label=name)
plt.title('Validation Accuracy')
plt.legend()
plt.show()

for name, cm in cms.items():
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(name + ' - Confusion Matrix')
    plt.show()