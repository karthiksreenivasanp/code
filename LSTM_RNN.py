
import time
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, layers, datasets, preprocessing

# Load IMDB dataset
(X_tr, y_tr), (X_te, y_te) = datasets.imdb.load_data(num_words=10000)

# Preprocessing
X_tr = preprocessing.sequence.pad_sequences(X_tr, maxlen=500)
X_te = preprocessing.sequence.pad_sequences(X_te, maxlen=500)

# Define all three models in one dictionary
models = {
    'RNN': layers.SimpleRNN(32),
    'LSTM': layers.LSTM(32),
    'GRU': layers.GRU(32)
}

# RNN
model1 = Sequential([
    layers.Embedding(10000, 32),
    models['RNN'],
    layers.Dense(1, activation='sigmoid')
])

# LSTM
model2 = Sequential([
    layers.Embedding(10000, 32),
    models['LSTM'],
    layers.Dense(1, activation='sigmoid')
])

# GRU
model3 = Sequential([
    layers.Embedding(10000, 32),
    models['GRU'],
    layers.Dense(1, activation='sigmoid')
])

# Compile models
model1.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
model2.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
model3.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

# Train RNN
start = time.time()
h1 = model1.fit(X_tr, y_tr, epochs=5, batch_size=64, validation_split=0.2)
t1 = time.time() - start
loss1, acc1 = model1.evaluate(X_te, y_te)

# Train LSTM
start = time.time()
h2 = model2.fit(X_tr, y_tr, epochs=5, batch_size=64, validation_split=0.2)
t2 = time.time() - start
loss2, acc2 = model2.evaluate(X_te, y_te)

# Train GRU
start = time.time()
h3 = model3.fit(X_tr, y_tr, epochs=5, batch_size=64, validation_split=0.2)
t3 = time.time() - start
loss3, acc3 = model3.evaluate(X_te, y_te)

# Print Results
print("RNN  - Loss:", loss1, "Accuracy:", acc1, "Time:", t1)
print("LSTM - Loss:", loss2, "Accuracy:", acc2, "Time:", t2)
print("GRU  - Loss:", loss3, "Accuracy:", acc3, "Time:", t3)

# Accuracy Comparison
plt.bar(['RNN', 'LSTM', 'GRU'], [acc1, acc2, acc3])
plt.title('Test Accuracy Comparison')
plt.ylabel('Accuracy')
plt.show()

# Loss Comparison
plt.bar(['RNN', 'LSTM', 'GRU'], [loss1, loss2, loss3])
plt.title('Test Loss Comparison')
plt.ylabel('Loss')
plt.show()

# Training Time Comparison
plt.bar(['RNN', 'LSTM', 'GRU'], [t1, t2, t3])
plt.title('Training Time Comparison')
plt.ylabel('Time (seconds)')
plt.show()
