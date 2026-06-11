import nltk
# Import other necessary components
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt_tab')
nltk.download('stopwords')
text = """
The Natural Language Toolkit, or NLTK for short, provides a suite of libraries and programs for symbolic and statistical natural language processing in Python. It includes  a diverse set of linguistic data and tools to work with, making it easier for developers and researchers to build 
NLP applications without having to worry about data gathering and preprocessing.
"""
# Convert the text to lowercase
text_lower = text.lower()
# Load the English stop words list from NLTK convert it to set
stop_words = set(stopwords.words('english'))
# Tokenize the text
tokens = word_tokenize(text_lower)
# Filter out the stop words
tokens_clean = [token for token in tokens if token not in stop_words]
print("Filtered Tokens:", tokens_clean)