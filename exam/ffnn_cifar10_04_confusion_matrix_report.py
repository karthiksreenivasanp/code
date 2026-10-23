"""Q: FFNN (3 hidden layers) on CIFAR-10. Plot and analyse confusion matrix and
classification report.

Algorithm:
1. Load/preprocess CIFAR-10, train the 3-hidden-layer FFNN.
2. Predict on test set, take argmax of probabilities.
3. Build confusion matrix and classification report.
4. Plot heatmap and print best / worst classes.
"""
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.utils import to_categorical

tf.keras.utils.set_random_seed(42)

# ---- Load & preprocess CIFAR-10 ----
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
y_train_int, y_test_int = y_train.flatten(), y_test.flatten()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

model = Sequential([
    Input(shape=(32, 32, 3)), Flatten(),
    Dense(512, activation="relu"), Dense(256, activation="relu"), Dense(128, activation="relu"),
    Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, validation_split=0.2, epochs=15, batch_size=64, verbose=1)
_, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {acc*100:.2f}%")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
cm = confusion_matrix(y_test_int, y_pred)
print("\nConfusion Matrix:\n", cm)
report = classification_report(y_test_int, y_pred, target_names=CLASSES, output_dict=True)
print("\nClassification Report:\n", classification_report(y_test_int, y_pred, target_names=CLASSES))

fig, ax = plt.subplots(figsize=(9, 8))
ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=True)
ax.set_title("CIFAR-10 FFNN Confusion Matrix"); plt.tight_layout(); plt.show()

f1 = {c: report[c]["f1-score"] for c in CLASSES}
print(f"Best class : {max(f1, key=f1.get)} (F1={max(f1.values()):.2f})")
print(f"Worst class: {min(f1, key=f1.get)} (F1={min(f1.values()):.2f})")
off = cm.copy()
np.fill_diagonal(off, 0)
i, j = np.unravel_index(np.argmax(off), off.shape)
print(f"Most confused pair: true {CLASSES[i]} predicted as {CLASSES[j]} ({off[i, j]} times)")
