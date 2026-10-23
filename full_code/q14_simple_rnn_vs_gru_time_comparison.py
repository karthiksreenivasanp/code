"""
Q14. Simple RNN and GRU; compare accuracy, loss and training time

Dataset: IMDb Movie Reviews
Architectures compared: SimpleRNN(32) vs. GRU(32)
Goal: Benchmark training execution time (using time.perf_counter), test loss, and test accuracy.

Algorithm:
1. Load IMDb; pad to 500.
2. Build SimpleRNN(32) and GRU(32) with the same Embedding(32).
3. Compile (Adam, binary cross-entropy).
4. Time training with time.perf_counter().
5. Train 5 epochs, batch 64, 20% validation.
6. Evaluate test loss/accuracy; print training times and compare.
"""

import time
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, GRU, Dense

# 1. Load IMDb and pad to length 500
num_words, maxlen = 10000, 500
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

layers = {"SimpleRNN": SimpleRNN, "GRU": GRU}
histories, times = {}, {}

# 2, 3, 4, 5. Train and time each model
print("=" * 65)
print("Benchmarking SimpleRNN vs GRU: Accuracy, Loss, and Training Time")
print("=" * 65)
for name, layer in layers.items():
    print(f"\nTraining {name}...")
    model = Sequential([
        Embedding(num_words, 32),
        layer(32),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    start = time.perf_counter()
    histories[name] = model.fit(
        x_train, y_train, epochs=5, batch_size=64,
        validation_split=0.2, verbose=0
    )
    times[name] = time.perf_counter() - start
    
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{name}: Test Loss={loss:.4f}, Test Accuracy={acc:.4f}")

# 6. Print training time comparison
print("\n" + "=" * 50)
print("Training Time Comparison:")
print("=" * 50)
for name, sec in times.items():
    print(f"{name:<10} training time: {sec:.2f} seconds")
print("=" * 50)

# Plot validation accuracy
for name, h in histories.items():
    plt.plot(h.history["val_accuracy"], label=name)
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("SimpleRNN vs GRU")
plt.legend()
plt.grid(True)
plt.show()
