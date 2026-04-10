import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import re

# download required data
nltk.download('punkt')
nltk.download('punkt_tab')   # needed sometimes
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')  # ✅ add this

text = "The quick brown fox jumps over the lazy dog."

text = text.lower()
text = re.sub(r'[^a-z\s]', '', text)

tokens = word_tokenize(text)

pos_tags = pos_tag(tokens)

print("\n--- Tokens ---")
print(tokens)

print("\n--- POS Tagging ---")
for word, tag in pos_tags:
    print(word, "->", tag)
