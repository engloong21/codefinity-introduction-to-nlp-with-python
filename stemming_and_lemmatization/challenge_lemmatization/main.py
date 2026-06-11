from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# Import the WordNet lemmatizer
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
# Download the WordNet corpus
nltk.download('wordnet')
text = "The leaves on the tree were turning a bright red, indicating that fall was leaving its mark."
text_lower = text.lower()
tokens = word_tokenize(text_lower)
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token not in stop_words]
# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()
# Lemmatize the tokens using list comprehension
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
print("Lemmatized tokens:", lemmatized_tokens)