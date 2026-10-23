"""
Q13. RNN and LSTM on IMDb; compare training/validation accuracy and explain why LSTM differs

Dataset: IMDb Movie Reviews
Architectures compared: SimpleRNN(32) vs. LSTM(32)
Goal: Plot training and validation accuracy curves for both, and explain the architectural difference.

Algorithm:
1. Load IMDb; pad sequences to 500.
2. Build SimpleRNN and LSTM with the same embedding size and units.
3. Compile and train both for 5 epochs with 20% validation.
4. Plot training and validation accuracy for both.
5. Explain the difference.

Theoretical Explanation (Exam Ready):
SimpleRNN has a single hidden state, and gradients shrink or explode over long
sequences (vanishing/exploding gradient), so it forgets early words. LSTM adds a
cell state controlled by input, forget and output gates, which decide what to store,
discard and output. This lets it retain long-term dependencies and usually gives
more stable, higher accuracy on long reviews.
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense

# 1. Load IMDb and pad sequences to 500
num_words, maxlen = 10000, 500
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

layers = {"SimpleRNN": SimpleRNN, "LSTM": LSTM}
histories = {}

# 2 & 3. Build, compile, and train both
print("=" * 60)
print("Training SimpleRNN and LSTM (5 Epochs)")
print("=" * 60)
for name, layer in layers.items():
    print(f"Training {name}...")
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
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{name}: Test Accuracy={acc:.4f}")

# 4. Plot training and validation accuracy for both
plt.figure(figsize=(8, 5))
for name, h in histories.items():
    plt.plot(h.history["accuracy"], label=f"{name} Train")
    plt.plot(h.history["val_accuracy"], "--", label=f"{name} Val")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy: SimpleRNN and LSTM")
plt.legend()
plt.grid(True)
plt.show()

# 5. Explain the difference
print("\n" + "=" * 70)
print("Explanation (Why LSTM differs and outperforms SimpleRNN):")
print("=" * 70)
print("1. SimpleRNN has only a single recurrent hidden state. When backpropagating")
print("   over long sequences (e.g. 500 words), gradients vanish or explode, causing")
print("   it to forget earlier words.")
print("2. LSTM introduces a separate cell state (constant error carousel) regulated by")
print("   three gates:")
print("   - Forget Gate: decides what information to discard from the cell state.")
print("   - Input Gate: decides which new information to store in the cell state.")
print("   - Output Gate: decides what information to output based on the cell state.")
print("3. As a result, LSTM avoids vanishing gradients across long time steps, retains")
print("   long-term context, and produces more stable, superior validation accuracy.")
print("=" * 70)
