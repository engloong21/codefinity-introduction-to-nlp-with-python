import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN


text = """Late in the evening, as the stars began to twinkle in the clear night sky, 
an old man sat quietly on the porch, reminiscing about his youthful adventures."""
# Convert the text to lowercase
text_lower = text.lower()
# Tokenize the text
tokens = word_tokenize(text_lower)
# Load the list of English stop words and convert it to set
stop_words = set(stopwords.words('english'))
# Filter out the stop words
filtered_tokens = [token for token in tokens if token not in stop_words]
# Perform POS tagging
tagged_tokens = pos_tag(filtered_tokens)
# Initialize a lemmatizer
lemmatizer = WordNetLemmatizer()
# Lemmatize the tokens
lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in tagged_tokens]
print(' '.join(lemmatized_tokens))