import re
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# dataset (sample sentences)
sentences = [
    "best laptop for coding",
    "top programming laptops for developers",
    "how to reset password",
    "payment failed refund issue",
    "application crashes frequently"
]

# preprocessing
def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.split()

clean_data = [clean(s) for s in sentences]

# train Word2Vec
model = Word2Vec(clean_data, vector_size=50, window=2, min_count=1, sg=1)

# sentence embedding
def sent_vector(sentence):
    vec = []
    for word in sentence:
        if word in model.wv:
            vec.append(model.wv[word])
    if len(vec) == 0:
        return np.zeros(50)
    return np.mean(vec, axis=0)

# convert dataset to vectors
data_vectors = [sent_vector(s) for s in clean_data]

# user input
query = input("\nEnter your query: ")
query_clean = clean(query)
query_vec = sent_vector(query_clean).reshape(1, -1)

# similarity
scores = []
for vec in data_vectors:
    sim = cosine_similarity(query_vec, vec.reshape(1, -1))[0][0]
    scores.append(sim)

# get best match
best_index = np.argmax(scores)

# output
print("\n--- User Query ---")
print(query)

print("\n--- Most Relevant Result ---")
print(sentences[best_index])

print("\n--- Similarity Score ---")
print(round(scores[best_index], 3))
