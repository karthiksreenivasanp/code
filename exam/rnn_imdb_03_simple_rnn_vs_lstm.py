"""Q: Implement Simple RNN and LSTM for binary sentiment classification on IMDb. Compare
training time, validation accuracy and test accuracy, and predict the sentiment of two
custom movie reviews.

Algorithm:
1. Load IMDb (10,000 words), pad to 500.
2. Build two models with identical Embedding(32) and 32 units: SimpleRNN and LSTM.
3. Compile (Adam, binary cross-entropy); time training with time.perf_counter().
4. Train 5 epochs, batch 64, 20% validation; evaluate on test set.
5. Print comparison table, plot validation accuracy/loss, predict two custom reviews.
"""
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, SimpleRNN, GRU, LSTM, Dropout, Dense

tf.keras.utils.set_random_seed(42)
NUM_WORDS, MAXLEN = 10000, 500

# ---- Load IMDB (top 10,000 words) and pad to fixed length ----
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=NUM_WORDS)
x_train = pad_sequences(x_train, maxlen=MAXLEN)
x_test = pad_sequences(x_test, maxlen=MAXLEN)

# ---- Helper to encode custom reviews (Keras reserves 0,1,2 -> pad, start, unknown) ----
word_index = imdb.get_word_index()
def encode(text):
    ids = [1]
    for w in text.lower().replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").split():
        i = word_index.get(w, 0) + 3
        ids.append(i if 3 < i < NUM_WORDS else 2)
    return pad_sequences([ids], maxlen=MAXLEN)

REVIEWS = [
    "This movie was fantastic. The acting was brilliant and the story was wonderful and moving",
    "Terrible film. The plot was boring, the acting was awful and I wasted my time",
]
def predict_reviews(model, name):
    for r in REVIEWS:
        p = float(model.predict(encode(r), verbose=0)[0][0])
        print(f"[{name}] {'Positive' if p >= 0.5 else 'Negative'} ({p:.3f}) -> {r[:60]}...")

LAYERS = {"SimpleRNN": SimpleRNN, "LSTM": LSTM}
histories, times, results, models = {}, {}, {}, {}

for name, layer in LAYERS.items():
    print(f"\n=== {name} ===")
    tf.keras.utils.set_random_seed(42)
    model = Sequential([
        Input(shape=(MAXLEN,)),
        Embedding(NUM_WORDS, 32),
        layer(32),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    t0 = time.perf_counter()
    histories[name] = model.fit(x_train, y_train, validation_split=0.2, epochs=5, batch_size=64, verbose=1)
    times[name] = time.perf_counter() - t0
    results[name] = model.evaluate(x_test, y_test, verbose=0)
    models[name] = model

print("\n" + "=" * 70)
print(f"{'Model':10s} | {'Train time(s)':>13s} | {'Val Acc':>8s} | {'Test Acc':>8s} | {'Test Loss':>9s}")
for n in LAYERS:
    h = histories[n].history
    print(f"{n:10s} | {times[n]:13.1f} | {h['val_accuracy'][-1]:8.4f} | {results[n][1]:8.4f} | {results[n][0]:9.4f}")
print("=" * 70)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for n, hh in histories.items():
    ax[0].plot(hh.history["val_accuracy"], label=n); ax[1].plot(hh.history["val_loss"], label=n)
ax[0].set_title("Validation Accuracy"); ax[1].set_title("Validation Loss")
for a in ax: a.set_xlabel("Epoch"); a.legend(); a.grid(True)
plt.tight_layout(); plt.show()

print("\nCustom review predictions:")
for n, m in models.items():
    predict_reviews(m, n)

