# Import the class for creating a Word2Vec model
from gensim.models import Word2Vec
import pandas as pd
corpus = pd.read_csv(
    'https://content-media-cdn.codefinity.com/courses/c68c1f2e-2c90-4d5d-8db9-1e97ca89d15e/section_3/chapter_4/example_corpus.csv')
# Tokenize each of the sentence
sentences = corpus['Document'].str.split()
# Initialize the model
model = Word2Vec(sentences, vector_size=50, window=2, min_count=1, sg=1)
# Print top-3 most similar words to 'bowl'
print(model.wv.most_similar('bowl', topn=3))