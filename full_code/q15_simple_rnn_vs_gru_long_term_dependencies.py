"""
Q15. Simple RNN and GRU; analyze how GRU handles long-term dependencies

Dataset: IMDb Movie Reviews
Goal: Analyze how GRU handles long-term dependencies compared to SimpleRNN via accuracy and loss subplots, plus gate-level theoretical explanation.

Algorithm:
1. Load IMDb; pad to 500 (long sequences stress memory).
2. Build equal-width SimpleRNN and GRU models.
3. Train under identical conditions (5 epochs).
4. Plot validation accuracy and loss.
5. Compare convergence and test accuracy; explain using GRU gates.

Analysis & Theory (Exam Ready):
GRU uses an update gate (how much of the old hidden state to keep) and a reset gate
(how much past information to use when forming the new candidate state). These gates create
a path along which information and gradients flow across many time steps with less decay,
so GRU retains long-range context better than SimpleRNN. It does this with fewer parameters than LSTM.
"""

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
histories = {}

# 2 & 3. Build equal-width models and train under identical conditions
print("=" * 65)
print("Analyzing Long-Term Dependencies: SimpleRNN vs GRU")
print("=" * 65)
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
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{name}: Test Loss={loss:.4f}, Test Accuracy={acc:.4f}")

# 4. Plot validation accuracy and loss side-by-side
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for name, h in histories.items():
    ax[0].plot(h.history["val_accuracy"], label=name)
    ax[1].plot(h.history["val_loss"], label=name)

ax[0].set_title("Validation Accuracy")
ax[1].set_title("Validation Loss")
for a in ax:
    a.set_xlabel("Epoch")
    a.legend()
    a.grid(True)
plt.tight_layout()
plt.show()

# 5. Analysis & explanation
print("\n" + "=" * 75)
print("Analysis (How GRU handles long-term dependencies):")
print("=" * 75)
print("1. Update Gate (z): Controls how much of the previous memory/hidden state")
print("   is preserved into the current step vs. how much new candidate state is accepted.")
print("2. Reset Gate (r): Decides how much of the past state to forget when computing")
print("   a new candidate hidden state.")
print("3. Why it handles long sequences well:")
print("   By allowing the update gate to remain close to 1, information can bypass")
print("   non-linear squashing and propagate unchanged across dozens or hundreds of")
print("   time steps without vanishing gradients. This lets GRU retain long-range context")
print("   substantially better than SimpleRNN, while being faster and computationally")
print("   cheaper than LSTM due to fewer parameters.")
print("=" * 75)
