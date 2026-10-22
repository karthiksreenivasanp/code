import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import Sequential, layers, datasets, callbacks, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

(x1, y1), (x2, y2) = datasets.imdb.load_data(num_words=10000)
x = np.concatenate((x1, x2))
y = np.concatenate((y1, y2))
x = preprocessing.sequence.pad_sequences(x, maxlen=500)
x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=42)

model = Sequential([
    layers.Embedding(10000, 32, mask_zero=True),
    layers.SimpleRNN(64),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
cbs = [
    callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(factor=0.5, patience=1)
]
h = model.fit(
    x_tr, y_tr,
    epochs=12,
    batch_size=64,
    validation_split=0.2,
    callbacks=cbs
)

loss, acc = model.evaluate(x_te, y_te)
print("Test Loss:", round(loss, 4))
print("Test Acc :", round(acc, 4))

y_pred = (model.predict(x_te) >= 0.5).astype(int).flatten()
print(classification_report(y_te, y_pred, digits=4))

cm = confusion_matrix(y_te, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.show()

for m in ['accuracy', 'loss']:
    plt.plot(h.history[m], label='Train ' + m)
    plt.plot(h.history['val_' + m], label='Val ' + m)
    plt.legend()
    plt.show()