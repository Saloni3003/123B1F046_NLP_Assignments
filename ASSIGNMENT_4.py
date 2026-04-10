import re
from gensim.models import Word2Vec

# input text
text = "I love machine learning and I love natural language processing"

# preprocessing
text = text.lower()
text = re.sub(r'[^a-z\s]', '', text)
words = text.split()

# prepare data (list of sentences)
data = [words]

# train Word2Vec model
model = Word2Vec(data, vector_size=50, window=2, min_count=1, sg=1)

# vocabulary
vocab = list(model.wv.index_to_key)

# output
print("\n--- Vocabulary ---")
print(vocab)

print("\n--- Word Vectors (sample) ---")
for word in vocab[:5]:
    print(word, "->", model.wv[word][:5])   # showing first 5 values only

# similarity
print("\n--- Similarity ---")
w1 = "machine"
w2 = "learning"

similarity = model.wv.similarity(w1, w2)

print(f"{w1} <-> {w2} =", round(similarity, 3))
