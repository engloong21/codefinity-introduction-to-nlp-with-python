from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# Import Porter Stemmer
from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
text = "Despite the pouring rain, the overwhelming sense of joy and accomplishment made the day unforgettable!"
# Convert the text to lowercase
text_lower = text.lower()
# Tokenization
tokens = word_tokenize(text_lower)
# Load English stop words
stop_words = set(stopwords.words('english'))
# Remove stop words (use list comprehension)
filtered_tokens = [token for token in tokens if token not in stop_words]
# Create a stemmer object
stemmer = PorterStemmer()
# Stemming (use list comprehension)
stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
print("Stemmed tokens:", stemmed_tokens)