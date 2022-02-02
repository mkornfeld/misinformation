import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.text import one_hot
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
dataset = pd.read_excel('newsfinal.xlsx', sheet_name=0)
dataset=dataset.astype(str)

titles = dataset['Title'].tolist()
articles = dataset['Article'].tolist()
labels = dataset['Type'].tolist()

new_labels = []
new_titles = []
for i in range(0, 6335):
    new_label = int(float(labels[i]))
    new_labels.append(new_label)
    new_titles.append(titles[i])

titles = new_titles
labels = new_labels
print("commence data processing")
vocab_size = 1000
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(titles, vocab_size, max_subword_length = 5)

for i, article in enumerate(titles):
    titles[i] = tokenizer.encode(article)

max_length = 50
trunc_type = "post"
padding_type = "post"

sequences_padded = pad_sequences(titles, maxlen=max_length,
                                 padding=padding_type, truncating=trunc_type)

training_size = int(len(titles) * 0.8)

training_sequences = sequences_padded[0:training_size]
testing_sequences = sequences_padded[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

print(training_sequences[0])
print(testing_sequences[0])
print(training_labels_final[0])
print(testing_labels_final[0])

embedding_dim = 16
print("commence machine learning")
model = tf.keras.Sequential([
    # tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),
    # tf.keras.layers.GlobalAveragePooling1D(),
    # tf.keras.layers.Dense(6, activation='relu'),
    # tf.keras.layers.Dense(1, activation='sigmoid')
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, return_sequences = True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')

])

model.summary()

num_epochs = 30
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(training_sequences, training_labels_final, epochs = num_epochs, validation_data=(testing_sequences, testing_labels_final))

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.xlabel("Epochs")
# plt.ylabel(string)
# plt.legend(['accuracy', 'val_accuracy'])
# plt.show()

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
