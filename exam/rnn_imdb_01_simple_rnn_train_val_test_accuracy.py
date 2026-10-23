"""Q: Design/implement and train a Simple RNN for IMDb sentiment classification and
evaluate with training, validation and test accuracy.

Algorithm:
1. Load IMDb (10,000 words), pad each review to 500 tokens.
2. Build Embedding(32) -> SimpleRNN(64) -> Dropout(0.2) -> Dense(1, sigmoid).
3. Compile (Adam, binary cross-entropy); train with 20% validation and EarlyStopping.
4. Print train/val/test accuracy; plot curves; predict two custom reviews.
"""
from tensorflow.keras.callbacks import EarlyStopping
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

model = Sequential([
    Input(shape=(MAXLEN,)),
    Embedding(NUM_WORDS, 32),
    SimpleRNN(64),
    Dropout(0.2),
    Dense(1, activation="sigmoid"),
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
              loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=64,
                    callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)], verbose=1)
h = history.history
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print(f"\nTraining Accuracy  : {h['accuracy'][-1]*100:.2f}%")
print(f"Validation Accuracy: {h['val_accuracy'][-1]*100:.2f}%")
print(f"Test Accuracy      : {test_acc*100:.2f}%  (Test loss {test_loss:.4f})")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(h["accuracy"], label="Training"); ax[0].plot(h["val_accuracy"], label="Validation"); ax[0].set_title("Accuracy")
ax[1].plot(h["loss"], label="Training"); ax[1].plot(h["val_loss"], label="Validation"); ax[1].set_title("Loss")
for a in ax: a.set_xlabel("Epoch"); a.legend(); a.grid(True)
plt.tight_layout(); plt.show()

print()
predict_reviews(model, "SimpleRNN")
