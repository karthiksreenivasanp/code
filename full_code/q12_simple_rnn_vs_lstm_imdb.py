"""
Q12. Simple RNN and LSTM on the same problem; compare accuracy and loss

Dataset: IMDb Movie Reviews
Architectures compared:
- Embedding(10000, 32) -> SimpleRNN(32) -> Dense(1, Sigmoid)
- Embedding(10000, 32) -> LSTM(32) -> Dense(1, Sigmoid)
Goal: Train both for 5 epochs, evaluate test loss & accuracy, and plot validation accuracy.

Algorithm:
1. Load IMDb (10,000 words); pad to 500.
2. Build SimpleRNN(32) and LSTM(32) models with the same Embedding(32).
3. Compile both (Adam, binary cross-entropy).
4. Train 5 epochs, batch 64, 20% validation.
5. Evaluate test loss/accuracy on the same test set.
6. Plot validation accuracy and compare.
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense

# 1. Load IMDb and pad to length 500
num_words, maxlen = 10000, 500
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

layers = {"SimpleRNN": SimpleRNN, "LSTM": LSTM}
histories = {}

# 2, 3, 4, 5. Build, compile, train, and evaluate both models
print("=" * 60)
print("Training and Evaluating SimpleRNN vs LSTM on IMDb")
print("=" * 60)
for name, layer in layers.items():
    print(f"Training {name} model...")
    model = Sequential([
        Embedding(num_words, 32),
        layer(32),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    histories[name] = model.fit(
        x_train, y_train, epochs=5, batch_size=64,
        validation_split=0.2, verbose=0
    )
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{name}: Test Loss={loss:.4f}, Test Accuracy={acc:.4f}")

# 6. Plot validation accuracy
for name, h in histories.items():
    plt.plot(h.history["val_accuracy"], label=name)
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("SimpleRNN vs LSTM")
plt.legend()
plt.grid(True)
plt.show()
