# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
corpus = pd.read_csv(
    'https://content-media-cdn.codefinity.com/courses/c68c1f2e-2c90-4d5d-8db9-1e97ca89d15e/section_3/chapter_4/example_corpus.csv')
# Instantiate TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
# Generate a TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus['Document'])
# Convert the resulting matrix to a DataFrame
tfidf_matrix_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
# Print the vector for "medical"
print(tfidf_matrix_df['medical'].values)