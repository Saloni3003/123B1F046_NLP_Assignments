import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# dataset
texts = [
    "book a flight",
    "cancel my ticket",
    "what is the weather",
    "book a hotel",
    "cancel my order",
    "what is temperature today",
    "reserve a seat",
    "book train ticket",
    "cancel booking",
    "refund my order",
    "is it raining today",
    "weather forecast"
]

labels = [
    "booking",
    "cancellation",
    "weather",
    "booking",
    "cancellation",
    "weather",
    "booking",
    "booking",
    "cancellation",
    "cancellation",
    "weather",
    "weather"
]

# preprocessing
def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

texts = [clean(t) for t in texts]

# TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# model training
model = MultinomialNB()
model.fit(X, labels)

# user input
query = input("\nEnter your query: ")
query_clean = clean(query)

# prediction
query_vec = vectorizer.transform([query_clean])
prediction = model.predict(query_vec)

# output
print("\n--- User Query ---")
print(query)

print("\n--- Predicted Intent ---")
print(prediction[0])
