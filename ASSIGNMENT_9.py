import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical

# text data
text = "I love machine learning and I love deep learning"

# preprocessing
text = text.lower()
text = re.sub(r'[^a-z\s]', '', text)
words = text.split()

# tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
seq = tokenizer.texts_to_sequences([text])[0]

# create sequences
seq_len = 3
X = []
y = []

for i in range(len(seq) - seq_len):
    X.append(seq[i:i+seq_len])
    y.append(seq[i+seq_len])

X = np.array(X)
y = to_categorical(y, num_classes=len(tokenizer.word_index) + 1)

# model
model = Sequential()
model.add(Embedding(input_dim=50, output_dim=8))
model.add(SimpleRNN(32))
model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

# training
model.fit(X, y, epochs=200, verbose=0)

# text generation
def generate_text(seed, n_words):
    for _ in range(n_words):
        seq = tokenizer.texts_to_sequences([seed])[0]
        seq = pad_sequences([seq], maxlen=seq_len, truncating='pre')
        pred = model.predict(seq, verbose=0)
        next_word = np.argmax(pred)
        
        for word, index in tokenizer.word_index.items():
            if index == next_word:
                seed += " " + word
                break
    return seed

# output
seed_text = "machine learning and"
generated = generate_text(seed_text, 3)

print("\n--- Seed Text ---")
print(seed_text)

print("\n--- Generated Text ---")
print(generated)
