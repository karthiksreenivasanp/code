import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import Sequential, layers, datasets
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Load data
(X_tr, y_tr), (X_te, y_te) = datasets.imdb.load_data(num_words=10000)

X_tr = pad_sequences(X_tr, maxlen=500)
X_te = pad_sequences(X_te, maxlen=500)

# Build model
model = Sequential([
    layers.Embedding(10000, 32, mask_zero=True),
    layers.SimpleRNN(64),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

# Train
h = model.fit(X_tr, y_tr, validation_split=0.2, epochs=10, batch_size=64)

# Evaluate
print("Test Accuracy:", model.evaluate(X_te, y_te)[1])

# Predict
pred = (model.predict(X_te) >= 0.5).astype(int).flatten()
print(classification_report(y_te, pred))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_te, pred), annot=True, fmt='d')
plt.show()

# Graphs
for i, m in enumerate(['loss', 'accuracy']):
    plt.subplot(1, 2, i + 1)
    plt.plot(h.history[m], label='Train ' + m)
    plt.plot(h.history['val_' + m], label='Val ' + m)
    plt.title(m.title())
    plt.legend()
plt.show()
