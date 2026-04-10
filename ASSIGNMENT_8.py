import re
import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression

# dataset (customer queries)
texts = [
    "forgot my password cannot login",
    "payment failed money deducted",
    "application crashes on opening",
    "unable to login account",
    "refund not received",
    "app is not working properly"
]

labels = [
    "account",
    "billing",
    "technical",
    "account",
    "billing",
    "technical"
]

# preprocessing
def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.split()

texts = [clean(t) for t in texts]

# train Word2Vec
model_w2v = Word2Vec(texts, vector_size=50, window=2, min_count=1, sg=1)

# sentence embedding (average of word vectors)
def get_vector(sentence):
    vec = []
    for word in sentence:
        if word in model_w2v.wv:
            vec.append(model_w2v.wv[word])
    if len(vec) == 0:
        return np.zeros(50)
    return np.mean(vec, axis=0)

X = np.array([get_vector(s) for s in texts])

# classifier
clf = LogisticRegression()
clf.fit(X, labels)

# user input
query = input("\nEnter customer query: ")
query_clean = clean(query)

query_vec = get_vector(query_clean).reshape(1, -1)
prediction = clf.predict(query_vec)

# output
print("\n--- User Query ---")
print(query)

print("\n--- Predicted Category ---")
print(prediction[0])
