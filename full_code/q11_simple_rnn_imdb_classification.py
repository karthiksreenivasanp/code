"""
Q11. Simple RNN for sequence classification; train and evaluate

Dataset: IMDb Movie Reviews (sentiment classification)
Architecture: Embedding(10000, 32) -> SimpleRNN(64) -> Dropout(0.2) -> Dense(1, Sigmoid)
Evaluation Metrics: Accuracy, Confusion Matrix, Classification Report, Precision, Recall, F1-Score

Algorithm:
1. Set random seed; load IMDb (top 10,000 words).
2. Combine train/test, shuffle, split 70:30.
3. Pad sequences to length 500.
4. Build Embedding(32) -> SimpleRNN(64) -> Dropout(0.2) -> Dense(1, Sigmoid).
5. Compile (Adam, binary cross-entropy); use EarlyStopping and ReduceLROnPlateau.
6. Train, evaluate test accuracy.
7. Compute confusion matrix, precision, recall, F1; plot accuracy.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, SimpleRNN, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_score, recall_score, f1_score)

# 1. Set random seeds for reproducibility
tf.keras.utils.set_random_seed(42)
np.random.seed(42)

# Load IMDb dataset
num_words, maxlen = 10000, 500
(xa, ya), (xb, yb) = imdb.load_data(num_words=num_words)

# 2. Combine and split 70:30
x = np.concatenate((xa, xb))
y = np.concatenate((ya, yb))
idx = np.random.permutation(len(x))
x, y = x[idx], y[idx]
split = int(0.70 * len(x))
x_train, y_train = x[:split], y[:split]
x_test, y_test = x[split:], y[split:]

# 3. Pad sequences to length 500
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# 4. Build SimpleRNN model
model = Sequential([
    Input(shape=(maxlen,)),
    Embedding(num_words, 32, mask_zero=True),
    SimpleRNN(64),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

# 5. Compile with Adam, gradient clipping, binary crossentropy
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5)
]

# 6. Train model
history = model.fit(
    x_train, y_train, epochs=12, batch_size=64,
    validation_split=0.2, callbacks=callbacks, verbose=1
)

# Evaluate test accuracy
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")

# 7. Compute confusion matrix, precision, recall, F1
y_pred = (model.predict(x_test, verbose=0) >= 0.5).astype(int).flatten()

print("\n" + "=" * 50)
print("Classification Evaluation Metrics")
print("=" * 50)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Negative", "Positive"], zero_division=0))
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred, zero_division=0):.4f}")
print("=" * 50)

# Plot accuracy curve
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("SimpleRNN Accuracy")
plt.legend()
plt.grid(True)
plt.show()
