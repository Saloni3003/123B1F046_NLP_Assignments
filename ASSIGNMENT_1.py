import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

text = "I am learning Natural Language Processing and studying different techniques like tokenization, stemming, and lemmatization."

text = text.lower()

tokens = word_tokenize(text)

stemmer = PorterStemmer()
stemmed = [stemmer.stem(word) for word in tokens]

lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word) for word in tokens]

# formatted output
print("\n--- Tokenization ---")
print(tokens)
 
print("\n--- Stemming ---")
for i in range(len(tokens)):
    print(tokens[i], "->", stemmed[i])

print("\n--- Lemmatization ---")
for i in range(len(tokens)):
    print(tokens[i], "->", lemmatized[i])
