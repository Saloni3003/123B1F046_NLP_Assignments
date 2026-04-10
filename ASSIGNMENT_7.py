import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# dataset
sentences = [
    ["ram", "lives", "in", "pune"],
    ["she", "likes", "python"],
    ["he", "is", "happy"]
]

labels = [
    ["NNP", "VBZ", "IN", "NNP"],
    ["PRP", "VBZ", "NN"],
    ["PRP", "VBZ", "JJ"]
]

# tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
X = tokenizer.texts_to_sequences(sentences)

# label encoding
tag2index = {"NNP":1, "VBZ":2, "IN":3, "PRP":4, "NN":5, "JJ":6}
y = [[tag2index[tag] for tag in sent] for sent in labels]

# padding
max_len = max(len(s) for s in X)
X = pad_sequences(X, maxlen=max_len, padding='post')
y = pad_sequences(y, maxlen=max_len, padding='post')
y = np.expand_dims(y, -1)

# model
model = Sequential()
model.add(Embedding(input_dim=50, output_dim=8))
model.add(LSTM(32, return_sequences=True))
model.add(Dense(8, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# training
model.fit(X, y, epochs=200, verbose=0)

# test input
test_sentence = ["This", "is", "cat"]
test_seq = tokenizer.texts_to_sequences([test_sentence])
test_pad = pad_sequences(test_seq, maxlen=max_len, padding='post')

# prediction
pred = model.predict(test_pad)
pred_labels = np.argmax(pred, axis=-1)

# reverse mapping
index2tag = {v:k for k,v in tag2index.items()}

# output
print("\n--- Input Sentence ---")
print(test_sentence)

print("\n--- Predicted Labels ---")
for i, word in enumerate(test_sentence):
    tag = index2tag.get(pred_labels[0][i], "O")
    print(word, "->", tag)
