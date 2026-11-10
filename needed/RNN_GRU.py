import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Embedding, SimpleRNN, LSTM, GRU, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report

tf.keras.utils.set_random_seed(42)
VOCAB_SIZE = 10000
MAX_LEN = 200  # Change to 500 to match the full Experiment 7 setting
EPOCHS = 5    # Increase to 12 for the full lab setting
BATCH_SIZE = 64

# Use the SAME train/test split for every model in comparisons.
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
x_train = pad_sequences(x_train, maxlen=MAX_LEN)
x_test = pad_sequences(x_test, maxlen=MAX_LEN)


def build_model(layer_type):
    layers = {'SimpleRNN': SimpleRNN, 'LSTM': LSTM, 'GRU': GRU}
    model = Sequential([
        Input(shape=(MAX_LEN,)),
        Embedding(input_dim=VOCAB_SIZE, output_dim=32, mask_zero=True),
        layers[layer_type](64),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_evaluate(name):
    print('\nTraining', name)
    model = build_model(name)
    model.summary()
    start = time.perf_counter()
    history = model.fit(x_train, y_train, epochs=EPOCHS,
                        batch_size=BATCH_SIZE, validation_split=0.2,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=2,
                                                 restore_best_weights=True)], verbose=1)
    seconds = time.perf_counter() - start
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f'{name}: test accuracy={accuracy:.4f}, test loss={loss:.4f}, training seconds={seconds:.1f}')
    return model, history, loss, accuracy, seconds


def plot_history(histories, metric):
    plt.figure(figsize=(9, 5))
    for name, history in histories.items():
        plt.plot(history.history[metric], label=f'{name} train')
        plt.plot(history.history['val_' + metric], '--', label=f'{name} validation')
    plt.title('Training and Validation ' + metric.title())
    plt.xlabel('Epoch')
    plt.ylabel(metric.title())
    plt.legend()
    plt.tight_layout()
    plt.show()


def show_classification(model, name):
    pred = (model.predict(x_test, verbose=0).ravel() >= 0.5).astype(int)
    cm = confusion_matrix(y_test, pred)
    print('\nConfusion matrix for', name, '\n', cm)
    print(classification_report(y_test, pred, target_names=['Negative', 'Positive']))
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap='Blues')
    plt.title(name + ' Confusion Matrix')
    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.yticks([0, 1], ['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    plt.colorbar()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results = {}
    for name in ['SimpleRNN', 'GRU']:
        model, history, loss, accuracy, seconds = train_evaluate(name)
        results[name] = (history, loss, accuracy, seconds)
    print('\nMODEL COMPARISON')
    for name, (_, loss, accuracy, seconds) in results.items():
        print(f'{name}: accuracy={accuracy:.4f}, loss={loss:.4f}, training time={seconds:.1f}s')
    plot_history({name: v[0] for name, v in results.items()}, 'accuracy')
    plot_history({name: v[0] for name, v in results.items()}, 'loss')

