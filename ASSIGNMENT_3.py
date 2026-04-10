import re
from collections import defaultdict

# input text
text = "I love machine learning and I love natural language processing"

# preprocessing
text = text.lower()
text = re.sub(r'[^a-z\s]', '', text)
words = text.split()

# choose N (2 = bigram, 3 = trigram)
N = 2

# generate n-grams
model = defaultdict(lambda: defaultdict(int))

for i in range(len(words) - N + 1):
    context = tuple(words[i:i+N-1])
    next_word = words[i+N-1]
    model[context][next_word] += 1

# calculate probabilities
prob_model = {}

for context in model:
    total = sum(model[context].values())
    prob_model[context] = {}
    for word in model[context]:
        prob_model[context][word] = model[context][word] / total

# prediction function
def predict_next(input_text):
    input_text = input_text.lower().split()
    context = tuple(input_text[-(N-1):])
    
    if context in prob_model:
        next_word = max(prob_model[context], key=prob_model[context].get)
        return next_word
    else:
        return "No prediction"

# output
print("\n--- Words ---")
print(words)

print("\n--- N-gram Model ---")
for context in prob_model:
    print(context, "->", prob_model[context])

# test prediction
input_seq = "i love"
print("\n--- Prediction ---")
print("Input:", input_seq)
print("Next word:", predict_next(input_seq))
