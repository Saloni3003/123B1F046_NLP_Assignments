import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, RegexpParser

# download required data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# input paragraph
text = """Natural language processing techniques are widely used in artificial intelligence applications.
The intelligent system extracts meaningful information from large datasets."""

# grammar for noun phrase
grammar = "NP: {<DT>?<JJ>*<NN.*>+}"
chunk_parser = RegexpParser(grammar)

noun_phrases = []

# processing
sentences = sent_tokenize(text)

for sent in sentences:
    words = word_tokenize(sent)
    tagged = pos_tag(words)
    tree = chunk_parser.parse(tagged)
    
    for subtree in tree.subtrees():
        if subtree.label() == "NP":
            np = " ".join(word for word, tag in subtree)
            noun_phrases.append(np)

# output
print("\n--- Input Paragraph ---")
print(text)

print("\n--- Extracted Noun Phrases ---")
for np in noun_phrases:
    print(np)
