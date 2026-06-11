from nltk.tokenize import word_tokenize
# Import the function for POS tagging
from nltk import pos_tag
import pandas as pd
import nltk
nltk.download('punkt_tab')
# Download the model needed for NLTK's POS tagging
nltk.download('averaged_perceptron_tagger_eng')
text = "The central bank raised interest rates by 0.5 percentage points yesterday."
text_lower = text.lower()
tokens = word_tokenize(text_lower)
# Perform POS tagging
tagged_tokens = pos_tag(tokens)
# Convert to DataFrame
print(pd.DataFrame(tagged_tokens, columns=['Token', 'POS tag']))