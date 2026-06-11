# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
corpus = pd.read_csv(
    'https://content-media-cdn.codefinity.com/courses/c68c1f2e-2c90-4d5d-8db9-1e97ca89d15e/section_3/chapter_4/example_corpus.csv')
# Instantiate CountVectorizer
count_vectorizer = CountVectorizer(ngram_range=(1, 2))
# Generate a BoW matrix
bow_matrix = count_vectorizer.fit_transform(corpus['Document'])
# Convert the resulting matrix to a DataFrame
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=count_vectorizer.get_feature_names_out())
# Print the vector for "graphic design"
print(bow_df['graphic design'].values)