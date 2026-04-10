import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# dataset
texts = [
    "I love this product",
    "This is amazing",
    "I am very happy",
    "I hate this",
    "This is bad",
    "Very disappointing",
    "Excellent service",
    "Worst experience",
    "I like it",
    "Not good"
]

labels = [
    "positive",
    "positive",
    "positive",
    "negative",
    "negative",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative"
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

# model
model = MultinomialNB()
model.fit(X, labels)

# simple evaluation (on same data)
pred = model.predict(X)

print("\n--- Model Evaluation ---")
correct = sum([1 for i in range(len(labels)) if labels[i] == pred[i]])
accuracy = correct / len(labels)
print("Accuracy:", round(accuracy, 2))

# user input
query = input("\nEnter text to classify: ")
query_clean = clean(query)

query_vec = vectorizer.transform([query_clean])
prediction = model.predict(query_vec)

print("\n--- Input Text ---")
print(query)

print("\n--- Predicted Sentiment ---")
print(prediction[0])
